"""exp07/exp12等价选择执行链：同一局部规则、所选覆盖目标、一次DP。"""
import math
from fractions import Fraction

import numpy as np

from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_methods.budget_contract import budget_config
from videoqa_methods.coarse_eligibility import coarse_selection_cells
from videoqa_methods.density_selection import build_regions, frame_key, source_time, validate_candidates
from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget
from videoqa_methods.weighted_seed_support import weighted_support

IMPLEMENTATION_ID = 'rd_selection_equivalent_v1'
METHODS = ('RD-1.2-exp07', 'RD-1.2-exp12')


def _selection_regions(plan, video):
    """接口：从已核验计划投影选择必需字段；不复制完整核心或历史审计字段。"""
    # 步骤1只传递时间边界与选择依据；步骤2沿用原整数PTS切分和同距归属。
    fields = ('region_id', 'left_fraction', 'right_fraction', 'seed_indices', 'anchor_index', 'weight')
    projected = {'regions': [{key: r[key] for key in fields} for r in plan['regions']]}
    return coarse_selection_cells(projected, video)['regions']


def _coverage_value(covered, weights, uniform, previous):
    """接口：保持原均值/ITM加权分支和单调截断，不引入新阈值或近似。"""
    if uniform:
        return float(np.mean(covered, dtype=np.float64))
    return min(1., max(previous, float(weights @ covered)))


def select_rd(pool, features, frame_budget, method_id, trace_mode='summary'):
    """接口：合法冻结池、归一化视觉特征、K和规范方法ID→源索引及当前选择轨迹。

    不读取历史配额、旧选择或旧轨迹；trace_mode只控制记录详细程度。
    本函数不采样、不编码、不生成答案。补查合法性沿用上游冻结池契约。
    """
    if method_id not in METHODS or trace_mode not in ('summary', 'full'):
        raise ProtocolError('Unsupported method_id or trace_mode')
    cfg = budget_config(frame_budget)
    c, video = pool['candidates'], pool['video']
    validate_candidates(c, video)
    ncoarse = pool['initial_count']
    if isinstance(ncoarse, bool) or not isinstance(ncoarse, int) or not 1 <= ncoarse <= len(c):
        raise ProtocolError('Invalid initial_count')
    scores = np.asarray(pool['initial_scores'], dtype=np.float64)
    if scores.shape != (ncoarse,) or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ProtocolError('Invalid coarse scores')
    f = np.asarray(features, dtype=np.float32)
    if (f.ndim != 2 or len(f) != len(c) or not np.isfinite(f).all()
            or not np.allclose(np.linalg.norm(f, axis=1), 1, atol=1e-5, rtol=0)):
        raise ProtocolError('Invalid normalized features')
    # 步骤1用原目标时间连接/真实PTS扩窗核验K对应计划，避免误用其他预算的细池。
    plan = build_regions(c[:ncoarse], pool['initial_scores'], video, cfg)
    if plan != pool['plan']:
        raise ProtocolError('Frozen pool does not follow the single-budget seed plan')
    cells = _selection_regions(plan, video)
    target = min(frame_budget, len(c))
    if not cells or len(cells) > target:
        raise ProtocolError('Protected anchor capacity failure')
    all_members, records = set(), []
    # 步骤2局部次序和所选目标的覆盖曲线同步构建，保持原matvec形状与累积顺序。
    for r in cells:
        left, right = Fraction(r['left_fraction']), Fraction(r['right_fraction'])
        ids = sorted([i for i, candidate in enumerate(c) if left <= source_time(candidate, video) <= right],
                     key=lambda i: frame_key(c[i]))
        refs = [i for i in ids if i < ncoarse]
        if all_members & set(ids) or not refs or r['anchor_index'] not in ids or not set(r['seed_indices']) <= set(ids):
            raise ProtocolError('Invalid region membership')
        original_left, original_right = map(Fraction, r['original_window'])
        for i in ids:
            if i >= ncoarse and not original_left <= source_time(c[i], video) <= original_right:
                raise ProtocolError('Fine candidate outside original window')
        all_members.update(ids)
        support, _, _ = weighted_support(f, ids, r['seed_indices'], pool['initial_scores'])
        if method_id == METHODS[0]:
            weights = np.full(len(refs), 1 / len(refs), dtype=np.float64)
            uniform, fallback = True, False
        else:
            raw = scores[refs];mass = float(raw.sum())
            uniform, fallback = bool(np.all(raw == raw[0])), mass == 0
            weights = np.full(len(refs), 1 / len(refs), dtype=np.float64) if fallback else raw / mass
        chosen = [r['anchor_index']]
        redundancy = np.clip(f[ids] @ f[chosen[0]], 0, 1)
        covered = np.clip(f[refs] @ f[chosen[0]], 0, 1)
        curve = [0., _coverage_value(covered, weights, uniform, 0.)]
        steps = []
        while len(chosen) < min(target, len(ids)):
            proposals = []
            for j, i in enumerate(ids):
                if i in chosen:
                    continue
                score = float(support[j]) * math.sqrt(1 - float(redundancy[j]))
                proposals.append(dict(candidate_index=i, support=float(support[j]),
                                      redundancy=float(redundancy[j]), local_score=score))
            winner = min(proposals, key=lambda p: (-p['local_score'], *frame_key(c[p['candidate_index']])))
            i = winner['candidate_index']; chosen.append(i)
            redundancy = np.maximum(redundancy, np.clip(f[ids] @ f[i], 0, 1))
            covered = np.maximum(covered, np.clip(f[refs] @ f[i], 0, 1))
            curve.append(_coverage_value(covered, weights, uniform, curve[-1]))
            if trace_mode == 'full':
                steps.append(winner)
        record = dict(region_id=r['region_id'], prefix=chosen, coverage_curve=curve, density_weight=r['weight'])
        if trace_mode == 'full':
            record.update(members=ids, reference_coarse_indices=refs, seed_indices=list(r['seed_indices']),
                anchor_index=r['anchor_index'], left_fraction=r['left_fraction'], right_fraction=r['right_fraction'],
                original_window=r['original_window'], local_steps=steps, reference_weights=weights.tolist(),
                uniform_fallback=fallback, exact_uniform_equivalence=uniform)
        records.append(record)
    if all_members != set(range(len(c))):
        raise ProtocolError('Candidate eligibility lost')
    # 步骤3只求解当前方法一次；直接闭合实际预算，不依赖历史参照配额提供总数。
    quotas, objective = allocate_prefix_budget([r['coverage_curve'] for r in records],
                                               [r['density_weight'] for r in records], target)
    selected = sorted([i for r, n in zip(records, quotas) for i in r['prefix'][:n]], key=lambda i: frame_key(c[i]))
    protected = [r['anchor_index'] for r in plan['regions']]
    if len(selected) != target or len(set(selected)) != target or not set(protected) <= set(selected):
        raise ProtocolError('Final budget or anchor identity mismatch')
    # 步骤4摘要不承载历史反事实；完整模式只导出当前计算已有的量，不触发额外求解。
    trace = dict(trace_schema='rd_selection_v1', implementation_id=IMPLEMENTATION_ID, method_id=method_id,
        frame_budget=frame_budget, seed_count=len(plan['seed_indices']), actual_frame_count=target,
        region_budgets={str(r['region_id']): n for r, n in zip(records, quotas)}, objective=objective,
        protected_indices=protected, coverage_policy='equal' if method_id == METHODS[0] else 'original_itm',
        limitation='Exact only within the original deterministic prefixes; visual coverage is not answer sufficiency')
    if trace_mode == 'full':
        trace.update(seed_indices=list(plan['seed_indices']), eligible_indices=sorted(all_members), regions=records)
    return selected, trace
