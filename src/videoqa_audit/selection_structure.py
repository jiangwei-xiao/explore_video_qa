"""同粗池选择组织对照：固定官方MMR/WFS，不接收答案或历史对错。"""
from dataclasses import asdict
from fractions import Fraction
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'third_party/WFS-SB-reference/core.py'
REFERENCE_SHA256 = '86614d63415abea4d65578c65710d32f68e422206d9da9b80381e9f2812106f0'
PARENT = 'RD-1.2-exp20'
GLOBAL = 'CTRL-GlobalMMR-S1'
WFS = 'REF-WFS-SB-S1'
CONTROLS = (GLOBAL, WFS)
PAPER_CONFIG = dict(wavelet='db4', lambda_param=.5, prominence_factor=.05,
    height_factor=.5, w_duration=.4, w_mean=.2, w_max=.3, w_var=.1,
    strictness_factor=1.2, temperature=1.)
REFERENCE_HELPERS = dict(drift=3, min_distance_ratio=.02, min_distance_absolute=5,
    padding_mode='symmetric', min_importance=.05, detail_level=1)


def reference_core(path=None):
    """接口：只加载字节匹配固定提交的官方核心；缺依赖不得降级到自写替代实现。"""
    path = Path(path or REFERENCE)
    # 步骤1先验源码身份；步骤2加载原文件，所有数值分支沿用参考代码。
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError('Fixed WFS core fingerprint differs')
    name = '_selection_structure_fixed_wfs'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_coarse(pool, features, frame_budget):
    """接口：核验完整粗前缀、原分数、单位视觉特征和真实时钟，拒绝细池混入。"""
    if isinstance(frame_budget, bool) or not isinstance(frame_budget, int) or frame_budget < 1:
        raise ValueError('Positive integer frame budget required')
    rows, video = pool['candidates'], pool['video']
    n = pool['initial_count']
    if isinstance(n, bool) or not isinstance(n, int) or not n or len(rows) != n:
        raise ValueError('Only the entire initial coarse pool is admissible')
    # 步骤1检查候选编号及双重源身份；步骤2只使用真实PTS验证时间，不反推平均FPS。
    if [r['candidate_index'] for r in rows] != list(range(n)):
        raise ValueError('Noncanonical coarse candidate indices')
    keys = [(r['source_pts'], r['source_frame_index']) for r in rows]
    if keys != sorted(keys) or len({k[0] for k in keys}) != n or len({k[1] for k in keys}) != n:
        raise ValueError('Sources must be unique and chronological')
    for row in rows:
        t = (int(row['source_pts']) - int(video['start_pts'])) * Fraction(video['time_base'])
        if not 0 <= t < Fraction(video['duration_fraction']):
            raise ValueError('Source outside video clock')
        if abs(float(t) - row['timestamp_seconds']) > 1e-8:
            raise ValueError('Timestamp differs from exact PTS')
    scores = np.asarray(pool['initial_scores'], dtype=np.float64)
    f = np.asarray(features)
    if scores.shape != (n,) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Invalid original-question ITM scores')
    if (f.dtype != np.float32 or f.ndim != 2 or f.shape[0] != n or f.shape[1] < 1
            or not np.isfinite(f).all()
            or not np.allclose(np.linalg.norm(f, axis=1), 1., atol=1e-5, rtol=0)):
        raise ValueError('Invalid frozen normalized FP32 features')
    return scores, f, min(frame_budget, n)


def _mmr_steps(core, selected, scores, features):
    """接口：从官方获选顺序记录每步赢家分量，额外计算只涉及获选帧，不重选。"""
    result = []
    for step, i in enumerate(selected):
        maximum = None if step == 0 else float(np.max(core.cosine_similarity(
            features[i:i+1], features[selected[:step]])[0]))
        result.append(dict(step=step, candidate_index=int(i), relevance=float(scores[i]),
            maximum_selected_cosine=maximum,
            mmr_score=None if maximum is None else .5*float(scores[i])-.5*maximum))
    return result


def select_coarse_control(pool, features, frame_budget, control_id, *, _reference=None):
    """接口：粗池、原特征、K及对照ID→最终索引和轨迹；无评分、解码、模型调用。

    _reference仅供CPU验收指定同SHA参考文件，正式入口固定使用仓库参考位置。
    本比较涉及目标和组织整层变化，不把它解释成单独删除区域或前缀的因果效应。
    """
    if control_id not in CONTROLS:
        raise ValueError('Unknown fixed structure control')
    scores, f, target = validate_coarse(pool, features, frame_budget)
    core = reference_core(_reference)
    trace = dict(control_id=control_id, frame_budget=frame_budget, actual_frame_count=target,
        coarse_count=len(scores), reference_sha256=REFERENCE_SHA256, new_model_calls=0,
        similarity='official sklearn cosine; no clipping', score_mapping='2*r-1',
        limitation='Whole-selection contrast on coarse pool; no isolated causal attribution')
    # 步骤1 G直接执行官方单段MMR，去掉分段、过滤、跨段配额及RD锚点。
    if control_id == GLOBAL:
        normalized = 2.*scores-1.
        raw = core.WFSFrameSelector(lambda_param=.5).select_from_segment(
            (0, len(scores)), target, normalized, f)
        trace.update(lambda_param=.5, selection_order=list(map(int, raw)),
            steps=_mmr_steps(core, list(raw), normalized, f))
    else:
        # 步骤2 W显式传入论文/CLI参数；观察实例方法的返回，不修改官方源码或回退。
        cfg = core.WFSConfig(**PAPER_CONFIG)
        if asdict(cfg) != PAPER_CONFIG:
            raise ValueError('Actual WFS paper configuration differs')
        selector = core.WFS(cfg)
        observed = {}
        def capture(obj, method, name):
            original = getattr(obj, method)
            def wrapped(*args, **kwargs):
                value = original(*args, **kwargs)
                if name == 'local_selections':
                    observed.setdefault(name, []).append(dict(segment=list(map(int,args[0] if args else kwargs['segment'])),
                        selected=list(map(int, value))))
                elif name == 'filtered_segments':
                    observed[name] = dict(segments=[list(map(int,x)) for x in value[0]], importance=list(map(float, value[1])))
                elif name == 'budget_adjustment':
                    observed[name] = dict(before=list(map(int, kwargs.get('selected_frames', args[0] if args else []))), after=list(map(int, value)))
                elif isinstance(value, np.ndarray):
                    observed[name] = value.tolist()
                elif isinstance(value, dict):
                    observed[name] = {str(k):int(v) for k,v in value.items()}
                else:
                    observed[name] = [list(map(int,x)) for x in value]
                return value
            setattr(obj, method, wrapped)
        capture(selector.event_detector, 'detect_peaks', 'peaks')
        capture(selector.event_detector, 'create_segments', 'segments')
        capture(selector.budget_allocator, 'filter_segments', 'filtered_segments')
        capture(selector.budget_allocator, 'allocate_budget', 'allocation')
        capture(selector.frame_selector, 'select_from_segment', 'local_selections')
        capture(selector.frame_selector, 'adjust_to_budget', 'budget_adjustment')
        level = core.compute_dwt_level(len(scores), wavelet='db4', drift=3)
        distance = core.compute_min_peak_distance(len(scores), ratio=.02, absolute_min=5)
        raw = selector.select_keyframes(scores, frame_budget, level, distance, f)
        trace.update(config=asdict(cfg), helpers=REFERENCE_HELPERS, dwt_level=int(level),
            min_peak_distance=int(distance), observed=observed,
            fallback='all_coarse' if len(scores) < frame_budget else
                'official_uniform_topk' if 'allocation' not in observed else 'official_segment_mmr')
    # 步骤3只整理源时间顺序并拒绝越界/重复/不足；不自定义补帧或更弱回退。
    raw = list(map(int, raw))
    if len(raw) != target or len(set(raw)) != target or not all(0 <= i < len(scores) for i in raw):
        raise ValueError('Official selection violated source budget')
    selected = sorted(raw, key=lambda i:(pool['candidates'][i]['source_pts'], i))
    trace['selected_indices'] = selected
    return selected, trace
