"""exp28：仅触发补查核心之间合并，原核心距离确定唯一候选归属。"""
from fractions import Fraction
import math
import numpy as np
from videoqa_methods.budget_contract import budget_config
from videoqa_methods.density_selection import source_time, frame_key, validate_candidates
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget


def conditional_groups(plan, frame_budget):
    """接口：原核心→选择组；触发窗口只与触发窗口合并，稀疏核心不能成为桥梁。

    组可能在时间上交错，不能把组首尾包围区间重新解释成新的补查窗口。
    """
    cfg = budget_config(frame_budget)
    hot = sorted((c for c in plan['cores'] if c['seed_count'] >= cfg['refinement_min_core_seeds']),
                 key=lambda c: (Fraction(c['left_fraction']), c['core_id']))
    groups = []
    # 步骤1：仅触发窗口按闭区间并集连接，保留每个真实成员核心的身份。
    for core in hot:
        left, right = Fraction(core['left_fraction']), Fraction(core['right_fraction'])
        if groups and left <= Fraction(groups[-1]['right_fraction']):
            groups[-1]['right_fraction'] = str(max(right, Fraction(groups[-1]['right_fraction'])))
            groups[-1]['core_ids'].append(core['core_id'])
        else:
            groups.append(dict(left_fraction=str(left), right_fraction=str(right), core_ids=[core['core_id']], refine=True))
    # 步骤2：未触发核心各自保留，即使与触发组或其他稀疏窗口重叠也不合并。
    for core in plan['cores']:
        if core['seed_count'] < cfg['refinement_min_core_seeds']:
            groups.append(dict(left_fraction=core['left_fraction'], right_fraction=core['right_fraction'],
                               core_ids=[core['core_id']], refine=False))
    groups.sort(key=lambda g: (min(g['core_ids']), Fraction(g['left_fraction'])))
    cores = {c['core_id']: c for c in plan['cores']}
    for index, group in enumerate(groups):
        sizes = [cores[i]['seed_count'] for i in group['core_ids']]
        group.update(region_id=index, seed_indices=[j for i in group['core_ids'] for j in cores[i]['seed_indices']],
                     weight=1 + sum(n - 1 for n in sizes))
    if len(groups) > frame_budget or sorted(i for g in groups for i in g['core_ids']) != sorted(cores):
        raise ValueError('Invalid conditional core partition')
    return groups


def core_ownership(candidates, plan):
    """接口：真实源PTS→最近未扩窗核心；重叠窗口不重复拥有候选，同距归较早核心。

    原核心种子区间定义时间归属，不声称语义事件。整数PTS比较与有理秒等价。
    """
    intervals = []
    for core in plan['cores']:
        pts = [int(candidates[i]['source_pts']) for i in core['seed_indices']]
        intervals.append((min(pts), max(pts), core['core_id']))
    # 步骤1：先给原核心唯一归属，再聚合到选择组，支持不连续的组成员。
    owners = [min(intervals, key=lambda x: (max(x[0] - c['source_pts'], 0, c['source_pts'] - x[1]),
                                          x[0], x[2]))[2] for c in candidates]
    for core in plan['cores']:
        if any(owners[i] != core['core_id'] for i in core['seed_indices']):
            raise ValueError('Core seed ownership changed')
    return owners


def select_conditional_merge(pool, features, frame_budget):
    """接口：exp27合法池→条件合并后的exp07前缀与等权DP；不采样或调用模型。

    改变组会连带改变锚点、支持归一化、去重域、粗参考和密度；这些均完整追踪，
    不将该组合干预称为纯补查效应或免费精度改进。
    """
    candidates, plan, video = pool['candidates'], pool['plan'], pool['video']
    validate_candidates(candidates, video)
    f = np.asarray(features, dtype=np.float32)
    if f.ndim != 2 or len(f) != len(candidates) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f, axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError('Invalid normalized feature bank')
    groups = conditional_groups(plan, frame_budget)
    owners = core_ownership(candidates, plan)
    core_group = {c: g['region_id'] for g in groups for c in g['core_ids']}
    target = min(frame_budget, len(candidates)); ncoarse = pool['initial_count']; records = []
    # 步骤1：候选全部保留且恰属一组，粗参照、支持和锚点按新组重新派生。
    for group in groups:
        rid = group['region_id']
        ids = sorted((i for i, c in enumerate(owners) if core_group[c] == rid), key=lambda i: frame_key(candidates[i]))
        refs = [i for i in ids if i < ncoarse]
        seeds = group['seed_indices']
        if not refs or not set(seeds) <= set(ids):raise ValueError('Conditional group lost coarse references or seeds')
        anchor = min(seeds, key=lambda i: (-pool['initial_scores'][i], *frame_key(candidates[i])))
        support, weights, fallback = weighted_support(f, ids, seeds, pool['initial_scores'])
        chosen = [anchor]; redundancy = np.clip(f[ids] @ f[anchor], 0, 1)
        covered = np.clip(f[refs] @ f[anchor], 0, 1)
        curve = [0.0, float(np.mean(covered, dtype=np.float64))]; steps = []
        # 步骤2：保持exp07局部L×sqrt(1-D)，新旧来源不产生优先级。
        while len(chosen) < min(target, len(ids)):
            proposals = [dict(candidate_index=i, support=float(support[j]), redundancy=float(redundancy[j]),
                              local_score=float(support[j]) * math.sqrt(1 - float(redundancy[j])))
                         for j, i in enumerate(ids) if i not in chosen]
            winner = min(proposals, key=lambda p: (-p['local_score'], *frame_key(candidates[p['candidate_index']])))
            i = winner['candidate_index']; chosen.append(i); steps.append(winner)
            redundancy = np.maximum(redundancy, np.clip(f[ids] @ f[i], 0, 1))
            covered = np.maximum(covered, np.clip(f[refs] @ f[i], 0, 1))
            curve.append(float(np.mean(covered, dtype=np.float64)))
        records.append(dict(**group, members=ids, reference_coarse_indices=refs, prefix=chosen,
                            anchor_index=anchor, coverage_curve=curve, density_weight=group['weight'],
                            seed_weights=weights, all_zero_fallback=fallback, local_steps=steps))
    # 步骤3：原正配额等权覆盖DP，固定同分规则；组数增加的必留成本如实保存。
    allocation, value = allocate_prefix_budget([r['coverage_curve'] for r in records],
                                               [r['density_weight'] for r in records], target)
    selected = sorted((i for r, n in zip(records, allocation) for i in r['prefix'][:n]), key=lambda i: frame_key(candidates[i]))
    if len(selected) != target or len(set(selected)) != target:raise ValueError('Conditional budget mismatch')
    return selected, dict(regions=records, region_budgets={str(r['region_id']): n for r, n in zip(records, allocation)},
                         protected_indices=[r['anchor_index'] for r in records], eligible_indices=list(range(len(candidates))),
                         candidate_core_owners=owners, candidate_group_owners=[core_group[c] for c in owners],
                         original_region_count=len(plan['regions']), selection_group_count=len(groups),
                         objective=value, rule='Only overlapping triggering core windows merge; nearest original-core ownership; exp07 equal coverage DP',
                         limitation='Group restructuring changes several downstream quantities; temporal groups are not semantic events')
