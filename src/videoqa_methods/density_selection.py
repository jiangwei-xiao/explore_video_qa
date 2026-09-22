"""检索密度v1纯选择逻辑：区域构建、补查目标与锚点保护竞争。"""
from fractions import Fraction
import math
import numpy as np
from videoqa_runtime.baseline_selection import ProtocolError


def source_time(row, video):
    """接口：由源PTS与视频起点得到精确相对时间，不使用平均FPS或展示小数。"""
    return (int(row['source_pts']) - int(video['start_pts'])) * Fraction(video['time_base'])


def duration(video):
    """接口：正式输入必须携带由流duration/time_base得到的有理数时长。"""
    value = Fraction(video['duration_fraction'])
    if value <= 0:
        raise ProtocolError('Non-positive exact stream duration')
    return value


def frame_key(row):
    """源时间及稳定候选ID提供所有同分情况的统一次序。"""
    return int(row['source_pts']), int(row['candidate_index'])


def validate_candidates(rows, video):
    """核对候选ID/源身份及相对时间；新帧可追加编号，不要求整个列表按PTS排序。"""
    if not rows or [r['candidate_index'] for r in rows] != list(range(len(rows))):
        raise ProtocolError('Empty pool or non-canonical candidate IDs')
    if len({r['source_pts'] for r in rows}) != len(rows) or len({r['source_frame_index'] for r in rows}) != len(rows):
        raise ProtocolError('Duplicate source identity')
    for row in rows:
        if not 0 <= source_time(row, video) <= duration(video):
            raise ProtocolError('Candidate outside video clock')
        if abs(float(source_time(row, video)) - row['timestamp_seconds']) > 1e-8:
            raise ProtocolError('Relative timestamp does not match source PTS')


def build_regions(candidates, scores, video, cfg):
    """接口：完整粗池与分数→种子、原核心、最终区域；不读取答案或新增候选。"""
    # 步骤1：核对完整初始池，稳定选择实际M_eff个原种子。
    validate_candidates(candidates, video)
    if len(scores) != len(candidates) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ProtocolError('Invalid coarse ITM scores')
    if cfg['seed_budget'] < 1 or cfg['frame_budget'] < 1:
        raise ProtocolError('Invalid configured budgets')
    ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], *frame_key(candidates[i])))
    seeds = ranking[:min(cfg['seed_budget'], len(candidates))]
    timed = sorted(seeds, key=lambda i: frame_key(candidates[i]))
    cores = []
    gap, radius = Fraction(cfg['core_gap_seconds']), Fraction(cfg['window_radius_seconds'])
    if gap < 0 or radius < 0:
        raise ProtocolError('Negative temporal configuration')
    # 步骤2：连接使用原1FPS目标时间；扩窗使用真实相对源时间。
    for i in timed:
        target = Fraction(str(candidates[i]['requested_seconds']))
        if not cores or target - Fraction(str(candidates[cores[-1]['seed_indices'][-1]]['requested_seconds'])) > gap:
            cores.append(dict(core_id=len(cores), seed_indices=[]))
        cores[-1]['seed_indices'].append(i)
    regions = []
    for core in cores:
        ids = core['seed_indices']
        left = max(Fraction(0), source_time(candidates[ids[0]], video) - radius)
        right = min(duration(video), source_time(candidates[ids[-1]], video) + radius)
        core.update(seed_count=len(ids), left_fraction=str(left), right_fraction=str(right))
        if regions and left <= Fraction(regions[-1]['right_fraction']):
            regions[-1]['right_fraction'] = str(max(right, Fraction(regions[-1]['right_fraction'])))
            regions[-1]['core_ids'].append(core['core_id'])
        else:
            regions.append(dict(region_id=len(regions), left_fraction=str(left), right_fraction=str(right), core_ids=[core['core_id']]))
    # 步骤3：最终区域只派生实际需要的成员数和触发依据，不预分帧配额。
    for region in regions:
        region_seeds = [i for c in region['core_ids'] for i in cores[c]['seed_indices']]
        sizes = [cores[c]['seed_count'] for c in region['core_ids']]
        left, right = Fraction(region['left_fraction']), Fraction(region['right_fraction'])
        coarse = [i for i, row in enumerate(candidates) if left <= source_time(row, video) <= right]
        region.update(seed_indices=region_seeds, core_seed_counts=sizes, m=len(region_seeds), c=len(sizes),
                      weight=1 + len(region_seeds) - len(sizes),
                      refine=max(sizes) >= cfg['refinement_min_core_seeds'], coarse_indices=coarse,
                      anchor_index=min(region_seeds, key=lambda i: (-scores[i], *frame_key(candidates[i]))))
    if len(regions) > cfg['frame_budget']:
        raise ProtocolError('Protected region anchors exceed frame budget')
    return dict(seed_indices=seeds, effective_seed_count=len(seeds), cores=cores, regions=regions)


def refinement_targets(region, video, cfg):
    """接口：生成闭窗口内全局2FPS新增网格，原1FPS目标不重复读取。"""
    if cfg['fine_fps'] != 2 or cfg['coarse_fps'] != 1 or Fraction(cfg['phase_fraction']) != Fraction(1, 4):
        raise ProtocolError('This version requires aligned 1/2 FPS grids')
    left, right = Fraction(region['left_fraction']), min(Fraction(region['right_fraction']), duration(video))
    first = max(0, math.ceil(left - Fraction(3, 4)))
    return [Fraction(3, 4) + k for k in range(first, math.floor(right - Fraction(3, 4)) + 1)]


def select_joint(candidates, features, plan, video, cfg):
    """接口：统一新旧池与视觉特征→最终索引、逐轮提议及实际区域配额。"""
    # 步骤1：每区初始化一个已保护锚点，并核对特征与候选身份。
    validate_candidates(candidates, video)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(candidates) or not np.isfinite(features).all():
        raise ProtocolError('Invalid feature matrix')
    norms = np.linalg.norm(features, axis=1)
    if not np.allclose(norms, 1, atol=1e-5, rtol=0):
        raise ProtocolError('Visual features must already be L2-normalized')
    selected, memberships, similarity, references = {}, {}, {}, {}
    for region in plan['regions']:
        rid = region['region_id']
        left, right = Fraction(region['left_fraction']), Fraction(region['right_fraction'])
        ids = sorted((i for i, row in enumerate(candidates) if left <= source_time(row, video) <= right), key=lambda i: frame_key(candidates[i]))
        anchor = region['anchor_index']
        if anchor not in ids or any(set(ids) & set(v) for v in memberships.values()):
            raise ProtocolError('Invalid or overlapping region membership')
        memberships[rid], selected[rid] = ids, [anchor]
        similarity[rid] = np.clip(features[ids] @ features[anchor], 0, 1)
        references[rid] = np.full(len(ids), anchor, dtype=np.int64)
    eligible = [i for ids in memberships.values() for i in ids]
    budget = min(cfg['frame_budget'], len(eligible))
    if len(selected) > budget or not eligible:
        raise ProtocolError('Protected anchor capacity failure')
    trace = []
    # 步骤2：区域内最大差异提议，区域间以固定聚集权重乘差异竞争。
    while sum(map(len, selected.values())) < budget:
        proposals = []
        for region in plan['regions']:
            rid = region['region_id']
            remaining = [j for j, i in enumerate(memberships[rid]) if i not in selected[rid]]
            if not remaining:
                continue
            position = min(remaining, key=lambda j: (float(similarity[rid][j]), *frame_key(candidates[memberships[rid][j]])))
            i = memberships[rid][position]
            redundancy = float(similarity[rid][position])
            proposals.append(dict(region_id=rid, candidate_index=i, redundancy=redundancy,
                                  diversity=1 - redundancy, weight=region['weight'],
                                  priority=region['weight'] * (1 - redundancy),
                                  nearest_selected_index=int(references[rid][position])))
        winner = min(proposals, key=lambda p: (-p['priority'], *frame_key(candidates[p['candidate_index']])))
        rid, i = winner['region_id'], winner['candidate_index']
        trace.append(dict(step=len(trace), proposals=proposals, winner=dict(winner), zero_diversity_fill=winner['priority'] == 0))
        selected[rid].append(i)
        # 步骤3：只更新获选区域，与全已选参照重算等价；不构造全片两两矩阵。
        values = np.clip(features[memberships[rid]] @ features[i], 0, 1)
        improved = values > similarity[rid]
        references[rid][improved] = i
        similarity[rid] = np.maximum(similarity[rid], values)
    indices = sorted((i for ids in selected.values() for i in ids), key=lambda i: frame_key(candidates[i]))
    if len(indices) != budget or len(set(indices)) != budget:
        raise ProtocolError('Final budget/uniqueness mismatch')
    return indices, dict(protected_indices=[r['anchor_index'] for r in plan['regions']],
                         eligible_indices=sorted(eligible), effective_frame_budget=budget, rounds=trace,
                         region_selected={str(k): v for k, v in selected.items()},
                         region_budgets={str(k): len(v) for k, v in selected.items()})
