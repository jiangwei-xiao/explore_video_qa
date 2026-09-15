"""CPU selection logic; no access to options or reference answers."""
from bisect import bisect_right
from fractions import Fraction
import numpy as np
import pywt
from scipy.signal import find_peaks
from videoqa_runtime.baseline_selection import ProtocolError


def segment_scores(scores, candidates, video, cfg):
    """接口：输入初始相关性、候选元数据、视频信息和固定配置，返回片段及小波诊断信号。分段只读取初始候选；不做片段过滤或配额分配。"""
    # 步骤1：校验初始分数，并按序列长度与小波滤波器上限确定分解层数。
    values = np.asarray(scores, dtype=np.float64)
    if len(values) != len(candidates) or not len(values) or not np.isfinite(values).all():
        raise ProtocolError('Invalid segmentation signal')
    maximum = pywt.dwt_max_level(len(values), pywt.Wavelet(cfg['wavelet']).dec_len)
    level = 1 if maximum <= 1 else int(np.clip(np.floor(np.log2(len(values)) - cfg['drift']), 1, maximum))
    # 步骤2：只重建最粗尺度的细节系数；不改变原始ITM分数的尺度。
    coefficients = pywt.wavedec(values, cfg['wavelet'], mode=cfg['wavelet_mode'], level=level)
    detail_coefficients = [np.zeros_like(c) for c in coefficients]
    detail_coefficients[1] = coefficients[1]
    detail = pywt.waverec(detail_coefficients, cfg['wavelet'], mode=cfg['wavelet_mode'])[:len(values)]
    signal = np.abs(detail)
    distance = max(cfg['min_peak_distance_absolute'], int(len(values) * cfg['min_peak_distance_ratio']))
    # 步骤3：在绝对细节信号上检测峰值，再形成全覆盖的连续候选片段。
    peaks = find_peaks(signal, height=signal.mean() + cfg['peak_height_factor'] * signal.std(),
                       prominence=cfg['peak_prominence_factor'] * np.ptp(signal), distance=distance)[0]
    boundaries = [0] + [int(p) for p in peaks if 0 < p < len(values)] + [len(values)]
    segments = []
    for a, b in zip(boundaries, boundaries[1:]):
        if b <= a:
            continue
        segments.append(dict(segment_id=len(segments), start=a, end=b,
                             start_pts=video['start_pts'] if a == 0 else candidates[a]['source_pts'],
                             start_seconds=0.0 if a == 0 else candidates[a]['timestamp_seconds'],
                             end_seconds=video['duration_seconds'] if b == len(values) else candidates[b]['timestamp_seconds'],
                             representative_score=float(values[a:b].max())))
    return segments, dict(level=level, peaks=[int(p) for p in peaks], detail_signal=detail.tolist())


def membership(candidates, segments):
    """接口：按真实源PTS把候选映射到初始片段；内部边界采用左闭右开，边界帧归右段。"""
    boundaries = [s['start_pts'] for s in segments[1:]]
    return np.array([bisect_right(boundaries, r['source_pts']) for r in candidates], dtype=np.int64)


def check_features(features, n):
    """校验纯视觉特征的数量、有限性和L2单位范数，返回FP32矩阵；异常时停止，不能静默换特征。"""
    f = np.asarray(features, dtype=np.float32)
    if f.ndim != 2 or len(f) != n or not np.isfinite(f).all():
        raise ProtocolError('Invalid pure visual feature matrix')
    if not np.allclose(np.linalg.norm(f, axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ProtocolError('Visual CLS features must have unit L2 norm')
    return f


def best(indices, gains, candidates):
    """按增益最大、源PTS最早、候选编号最小的固定顺序返回一个候选索引。"""
    return min(indices, key=lambda i: (-float(gains[i]), candidates[i]['source_pts'], candidates[i]['candidate_index']))


def compete(candidates, scores, features, segments, lam, cfg, count=16):
    """核心接口：在固定总帧数下执行逐帧跨段竞争，返回时间有序的候选索引、片段配额和每步追踪。即使所有剩余增益为负，也必须选满预算。"""
    # 步骤1：建立片段归属、固定代表分数与零配额/零重复度初始状态。
    if len(candidates) < count:
        raise ProtocolError('Not enough unique candidates')
    r = np.asarray(scores, dtype=np.float64)
    f = check_features(features, len(r))
    groups = membership(candidates, segments)
    representatives = np.array([s['representative_score'] for s in segments])
    quotas = np.zeros(len(segments), dtype=np.int64)
    duplicate = np.zeros(len(r), dtype=np.float32)
    chosen, trace = [], []
    remaining = set(range(len(r)))
    # 步骤2：每轮按原公式评分；每段先提名，再在所有提名中选出一帧。
    for step in range(count):
        gains = (1 - lam) * r + lam * representatives[groups] / (1 + quotas[groups]) - cfg['redundancy_coefficient'] * duplicate
        proposals = []
        for sid in range(len(segments)):
            available = [i for i in remaining if groups[i] == sid]
            if available:
                i = best(available, gains, candidates)
                proposals.append(dict(segment_id=sid, candidate_index=i, gain=float(gains[i])))
        i = best([p['candidate_index'] for p in proposals], gains, candidates)
        sid = int(groups[i])
        trace.append(dict(step=step, candidate_index=i, segment_id=sid, gain=float(gains[i]),
                          relevance=float(r[i]), representative=float(representatives[sid]),
                          segment_count_before=int(quotas[sid]), duplicate=float(duplicate[i]), proposals=proposals))
        # 步骤3：只更新获胜片段的已选数量与最大相似度，避免跨段去重。
        chosen.append(i)
        remaining.remove(i)
        quotas[sid] += 1
        local = np.flatnonzero(groups == sid)
        duplicate[local] = np.maximum(duplicate[local], np.clip(f[local] @ f[i], 0, 1))
    return sorted(chosen, key=lambda i: (candidates[i]['source_pts'], candidates[i]['candidate_index'])), quotas.tolist(), trace


def hotspot_candidates(candidates, scores, features, selected, segments, quotas, video, cfg):
    """核心接口：根据粗选配额和初始特征计算热点与未覆盖度，返回全部判定记录及至多两个不重叠窗口。边界判断使用有理数时间。"""
    # 步骤1：构造精确时间和75百分位阈值；每个有配额片段只提名一个热点。
    f = check_features(features, len(candidates))
    r = np.asarray(scores)
    groups = membership(candidates, segments)
    tb, origin = Fraction(video['time_base']), video['start_pts']
    times = [(c['source_pts'] - origin) * tb for c in candidates]
    duration = Fraction(str(video['duration_seconds']))
    radius = Fraction(str(cfg['hotspot_window_seconds']))
    threshold = float(np.percentile(r, cfg['hotspot_percentile'], method='linear'))
    records = []
    for sid, quota in enumerate(quotas):
        if quota == 0:
            continue
        local = [i for i in selected if groups[i] == sid]
        h = min(local, key=lambda i: (-r[i], candidates[i]['source_pts'], candidates[i]['candidate_index']))
        left, right = max(Fraction(0), times[h] - radius), min(duration, times[h] + radius)
        record = dict(segment_id=sid, candidate_index=h, source_pts=candidates[h]['source_pts'],
                      relevance=float(r[h]), threshold=threshold, left_fraction=str(left), right_fraction=str(right),
                      left_seconds=float(left), right_seconds=float(right), center_fraction=str(times[h]), selected=False)
        if r[h] < threshold:
            record.update(reason='below_percentile', uncovered=None, priority=None)
        else:
            # 步骤2：用窗口内所有粗候选对窗口内粗选帧计算未覆盖度。
            inside = [i for i, t in enumerate(times) if left <= t <= right]
            selected_inside = [i for i in selected if left <= times[i] <= right]
            similarities = np.clip(f[inside] @ f[selected_inside].T, 0, 1).max(axis=1)
            uncovered = float(np.mean(r[inside] * (1 - similarities)))
            record.update(uncovered=uncovered, priority=float(r[h] * uncovered),
                          reason='eligible' if uncovered > cfg['uncovered_epsilon'] else 'already_covered')
        records.append(record)
    # 步骤3：按热点相关性×未覆盖度排序；相接端点也算重叠，最多接受两个窗口。
    eligible = sorted([x for x in records if x['reason'] == 'eligible'], key=lambda x: (-x['priority'], x['source_pts'], x['candidate_index']))
    selected_windows = []
    for record in eligible:
        if len(selected_windows) >= cfg['maximum_hotspots']:
            record['reason'] = 'hotspot_limit'
        elif any(Fraction(record['left_fraction']) <= Fraction(other['right_fraction']) and
                 Fraction(other['left_fraction']) <= Fraction(record['right_fraction']) for other in selected_windows):
            record['reason'] = 'overlap'
        else:
            record.update(selected=True, reason='selected')
            selected_windows.append(record)
    return records, selected_windows


def reselect(candidates, scores, features, segments, quotas, coarse_selected, initial_count, lam, cfg):
    """核心接口：仅在收到新增候选且已有配额的片段中从空集合重选，返回最终索引和重选追踪；初始边界、代表分数和配额保持固定。"""
    r = np.asarray(scores, dtype=np.float64)
    f = check_features(features, len(r))
    groups = membership(candidates, segments)
    # 步骤1：确定收到新候选的片段；其余片段保留原粗选结果。
    changed = set(groups[initial_count:].tolist())
    selected, trace = [], []
    for sid, quota in enumerate(quotas):
        if not quota:
            continue
        if sid not in changed:
            selected.extend(i for i in coarse_selected if groups[i] == sid)
            continue
        # 步骤2：有配额片段从空集合重选。覆盖项段内相同，但仍保留在增益追踪中。
        remaining = set(np.flatnonzero(groups == sid).tolist())
        local = []
        for _ in range(quota):
            duplicate = np.zeros(len(r)) if not local else np.clip(f @ f[local].T, 0, 1).max(axis=1)
            gains = ((1 - lam) * r + lam * segments[sid]['representative_score'] / (1 + len(local))
                     - cfg['redundancy_coefficient'] * duplicate)
            i = best(remaining, gains, candidates)
            trace.append(dict(segment_id=sid, candidate_index=i, gain=float(gains[i]),
                              relevance=float(r[i]), duplicate=float(duplicate[i]), segment_count_before=len(local)))
            remaining.remove(i)
            local.append(i)
        selected.extend(local)
    selected.sort(key=lambda i: (candidates[i]['source_pts'], candidates[i]['candidate_index']))
    # 步骤3：恢复时间顺序并核对原配额，防止零配额区域获得新名额。
    if np.bincount(groups[selected], minlength=len(segments)).tolist() != quotas:
        raise ProtocolError('Refinement changed coarse segment quotas')
    return selected, trace
