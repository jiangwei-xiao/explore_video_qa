"""离线候选：描述检索资格覆盖已有粗帧，补查位置及其他exp17规则不变。"""
import math
from fractions import Fraction
import numpy as np
from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_methods.density_selection import validate_candidates, source_time, frame_key
from videoqa_methods.coarse_eligibility import coarse_selection_cells
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget
from videoqa_methods.local_query_coverage import strict_rank_counts, select_local_query_coverage


def select_coarse_description_access(pool, features, cfg, reference_quotas, option_scores, enabled=True):
    """接口：四描述等权覆盖已有单元全部候选；唯一变化是允许窗外粗帧参与描述竞争。

    不移动种子或锚点、不增加细帧、不选择正确选项。扩大查询排名比较域并非只给
    某些帧加资格，也可能改变原窗内帧的序数增益，必须在离线对照中如实报告。
    """
    if not enabled:
        return select_local_query_coverage(pool, features, cfg, reference_quotas, option_scores, False)
    candidates, video, plan = pool['candidates'], pool['video'], pool['plan']
    n = pool['initial_count']
    validate_candidates(candidates, video)
    f = np.asarray(features, dtype=np.float32)
    scores = np.asarray(option_scores, dtype=np.float64)
    if f.ndim != 2 or len(f) != len(candidates) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f, axis=1), 1, atol=1e-5, rtol=0):
        raise ProtocolError('Invalid feature identity')
    if scores.shape != (4, len(candidates)) or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ProtocolError('Expected four complete cached score vectors')
    if n != len(pool['initial_scores']):
        raise ProtocolError('Coarse prefix mismatch')
    cells = coarse_selection_cells(plan, video)
    target = min(cfg['frame_budget'], len(candidates))
    records, all_members = [], set()
    # 步骤1：单元和原窗口分别保留；细帧仍必须来自旧补查窗，只开放已有粗帧描述资格。
    for pos, r in enumerate(cells['regions']):
        original = plan['regions'][pos]
        ids = sorted([i for i, c in enumerate(candidates) if Fraction(r['left_fraction']) <= source_time(c, video) <= Fraction(r['right_fraction'])],
                     key=lambda i: frame_key(candidates[i]))
        refs = [i for i in ids if i < n]
        window_ids = [i for i in ids if Fraction(original['left_fraction']) <= source_time(candidates[i], video) <= Fraction(original['right_fraction'])]
        anchor = r['anchor_index']
        if all_members.intersection(ids) or not refs or anchor not in window_ids or not set(r['seed_indices']) <= set(ids):
            raise ProtocolError('Region identity mismatch')
        if any(i >= n and i not in window_ids for i in ids):
            raise ProtocolError('Fine frame outside original window')
        all_members.update(ids)
        ranks = strict_rank_counts(scores[:, ids])
        chosen = [anchor]
        support, _, _ = weighted_support(f, ids, r['seed_indices'], pool['initial_scores'])
        redundancy = np.clip(f[ids] @ f[anchor], 0, 1)
        covered = np.clip(f[refs] @ f[anchor], 0, 1)
        curve = [0., float(np.mean(covered, dtype=np.float64))]
        best = ranks[:, ids.index(anchor)].copy()
        query_steps, active, steps = 0, True, []
        # 步骤2：沿用exp17四步上限、贪心描述增益与原视觉同分规则，不增加阈值或参数。
        while len(chosen) < min(target, len(ids)):
            proposals = []
            for j, i in enumerate(ids):
                if i in chosen:
                    continue
                gain = int(np.sum(np.maximum(best, ranks[:, j]) - best))
                visual = float(support[j]) * math.sqrt(1 - float(redundancy[j]))
                proposals.append(dict(candidate_index=i, position=j, query_gain=gain, support=float(support[j]),
                                      redundancy=float(redundancy[j]), local_score=visual))
            if active and query_steps < 4 and max(p['query_gain'] for p in proposals) > 0:
                winner = min(proposals, key=lambda p: (-p['query_gain'], -p['local_score'], *frame_key(candidates[p['candidate_index']])))
                query_steps += 1
                mode = 'description_coverage'
            else:
                active = False
                winner = min(proposals, key=lambda p: (-p['local_score'], *frame_key(candidates[p['candidate_index']])))
                mode = 'original_visual_rule'
            i = winner['candidate_index']
            chosen.append(i)
            best = np.maximum(best, ranks[:, winner['position']])
            redundancy = np.maximum(redundancy, np.clip(f[ids] @ f[i], 0, 1))
            covered = np.maximum(covered, np.clip(f[refs] @ f[i], 0, 1))
            curve.append(float(np.mean(covered, dtype=np.float64)))
            steps.append(dict(winner, mode=mode, query_coverage_count=int(best.sum())))
        records.append(dict(region_id=r['region_id'], members=ids, reference_coarse_indices=refs,
                            original_window_indices=window_ids, description_indices=ids, prefix=chosen,
                            coverage_curve=curve, density_weight=r['weight'], local_steps=steps, query_steps=query_steps))
    # 步骤3：保持原密度加权粗观测覆盖DP，最终预算与源时间排序不变。
    if all_members != set(range(len(candidates))):
        raise ProtocolError('Candidate eligibility changed')
    allocation, value = allocate_prefix_budget([r['coverage_curve'] for r in records], [r['density_weight'] for r in records], target)
    selected = sorted([i for r, k in zip(records, allocation) for i in r['prefix'][:k]], key=lambda i: frame_key(candidates[i]))
    baseline = sum(r['density_weight'] * r['coverage_curve'][reference_quotas[str(r['region_id'])]] for r in records)
    if len(selected) != len(set(selected)) or len(selected) != target or value + 1e-12 < baseline:
        raise ProtocolError('Budget or DP invariant failed')
    return selected, dict(regions=records, region_budgets={str(r['region_id']): k for r, k in zip(records, allocation)},
                         protected_indices=[r['anchor_index'] for r in plan['regions']], eligible_indices=sorted(all_members),
                         objective=value, reference_objective=baseline,
                         limitation='Expanded ordinal description access is not evidence verification or answer confidence')
