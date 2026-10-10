"""选项支持只改变局部排序：原L、视觉D、锚点与exp07预算完整保留。"""
import math
import numpy as np
from videoqa_methods.density_selection import frame_key, validate_candidates
from videoqa_methods.weighted_seed_support import weighted_support


def select_option_evidence(pool, features, parent, evidence, budget=32):
    """接口：完整合法池/旧配额/逐候选E→有序源索引与局部轨迹，不接收gold或模型意见。"""
    candidates = pool['candidates']; validate_candidates(candidates, pool['video'])
    f = np.asarray(features, dtype=np.float32)
    if f.ndim != 2 or len(f) != len(candidates) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f, axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError('Original visual feature population differs')
    values = {int(i): float(v) for i, v in evidence.items()}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError('Invalid evidence proxy')
    selected = []; traces = []; owned = []
    # 步骤1：全部粗细候选保持原归属；配额1无需评分，其他位置缺失E必须暂停。
    for region, plan in zip(parent['regions'], pool['plan']['regions']):
        members = region['members']; owned += members; anchor = plan['anchor_index']
        quota = parent['region_budgets'][str(region['region_id'])]
        if region['region_id'] != plan['region_id'] or anchor not in members or not 1 <= quota <= len(members):
            raise ValueError('Frozen region allocation differs')
        eligible = [i for i in members if i != anchor]
        if quota > 1 and not set(eligible) <= set(values):
            raise ValueError('Unscored legal candidate; no silent E fallback')
        fallback = quota == 1 or not any(values[i] > 0 for i in eligible)
        support, _, _ = weighted_support(f, members, plan['seed_indices'], pool['initial_scores'])
        support = dict(zip(members, support)); chosen = [anchor]; steps = []
        # 步骤2：原视觉重复度用于近重复抑制；E只乘局部增益，不修改覆盖目标或配额。
        while len(chosen) < quota:
            redundant = np.maximum.reduce([np.clip(f[members] @ f[j], 0, 1) for j in chosen])
            proposals = []
            for position, i in enumerate(members):
                if i in chosen:
                    continue
                e = 1.0 if fallback else values[i]; d = float(redundant[position])
                proposals.append(dict(candidate_index=i, L=float(support[i]), D=d, E=e,
                    score=float(support[i]) * e * math.sqrt(1 - d)))
            winner = min(proposals, key=lambda v: (-v['score'], *frame_key(candidates[v['candidate_index']])))
            steps.append(dict(selected_before=list(chosen), winner=winner)); chosen.append(winner['candidate_index'])
        selected += chosen; traces.append(dict(region_id=region['region_id'], quota=quota,
            prefix=chosen, no_signal_fallback=fallback, steps=steps))
    # 步骤3：帧预算是上限；沿用父实际预算，不重复源帧补齐，恢复真实源时间排序。
    expected = sum(parent['region_budgets'].values())
    if sorted(owned) != list(range(len(candidates))) or len(set(selected)) != expected or len(selected) != expected or expected > budget:
        raise ValueError('Ownership, uniqueness or budget mismatch')
    selected = sorted(selected, key=lambda i: frame_key(candidates[i]))
    return selected, dict(regions=traces, region_budgets=parent['region_budgets'], selected_indices=selected,
        rule='Original L * option-evidence E * sqrt(1 - original visual D); frozen exp07 quota')
