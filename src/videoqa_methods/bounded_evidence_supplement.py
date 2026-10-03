"""离线有界证据补充候选：保留exp07主体，至多四次增删配对，不修改上游。"""
import math
import numpy as np
from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_methods.density_selection import validate_candidates, frame_key


def conditional_gain(original_scores, joined_scores):
    """接口：对称使用四个问题加选项响应；log-odds差仅为启发式，不是条件概率。"""
    q, a = np.asarray(original_scores, dtype=np.float64), np.asarray(joined_scores, dtype=np.float64)
    if q.ndim != 1 or a.shape != (4, len(q)) or not np.isfinite(q).all() or not np.isfinite(a).all():
        raise ProtocolError('Invalid score identity')
    if (q < 0).any() or (q > 1).any() or (a < 0).any() or (a > 1).any():
        raise ProtocolError('Score outside probability range')
    eps = np.finfo(np.float64).eps
    def logits(x):
        """中文说明：转换匹配分数，机器精度裁剪只避免0或1处对数无定义。"""
        x = np.clip(x, eps, 1 - eps)
        return np.log(x) - np.log1p(-x)
    return np.maximum(0., (logits(a) - logits(q)[None, :]).max(axis=0))


def coverage_and_removal_cost(features, regions, selected):
    """接口：计算原密度加权粗观测覆盖，以及逐个删除当前帧的精确边际损失。"""
    total, costs = 0., {}
    for r in regions:
        local = sorted(set(r['members']).intersection(selected))
        refs = r['reference_coarse_indices']
        if not local or not refs:
            raise ProtocolError('Empty covered region')
        sims = np.clip(features[refs] @ features[local].T, 0, 1)
        best = sims.max(axis=1)
        second = np.partition(sims, -2, axis=1)[:, -2] if len(local) > 1 else np.zeros(len(refs))
        owner = sims.argmax(axis=1)
        weight = r['density_weight']
        total += weight * float(np.mean(best, dtype=np.float64))
        # 同分时第二大值等于最大值，删除任一重复见证的损失正确为零。
        for j, i in enumerate(local):
            costs[i] = weight * float(np.sum(np.where(owner == j, best - second, 0), dtype=np.float64)) / len(refs)
    return total, costs


def supplement(pool, features, original_indices, original_trace, joined_scores):
    """接口：固定最多4次替换；输入无答案，候选只来自已有粗池，旧细帧仍可保留。

    提议优先级为正匹配增量×sqrt(1−同单元最大相似度)；删除以加入后的原覆盖
    损失最小为准。这是资源受限的启发式组合，不宣称优化答题正确率或无损压缩。
    """
    candidates = pool['candidates']
    validate_candidates(candidates, pool['video'])
    f = np.asarray(features, dtype=np.float32)
    if f.ndim != 2 or len(f) != len(candidates) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f, axis=1), 1, atol=1e-5, rtol=0):
        raise ProtocolError('Invalid normalized features')
    original = set(original_indices)
    if len(original) != len(original_indices) or len(original) != min(32, len(candidates)):
        raise ProtocolError('Expected frozen K32 input')
    regions = original_trace['regions']
    owners = {i: r['region_id'] for r in regions for i in r['members']}
    if len(owners) != sum(len(r['members']) for r in regions) or set(owners) != set(range(len(candidates))):
        raise ProtocolError('Invalid region partition')
    anchors = set(original_trace['protected_indices'])
    if not anchors <= original:
        raise ProtocolError('Missing protected anchor')
    gain = conditional_gain(pool['initial_scores'], joined_scores)
    if len(gain) != pool['initial_count']:
        raise ProtocolError('Coarse score prefix mismatch')
    selected, protected, steps = original.copy(), anchors.copy(), []
    baseline, _ = coverage_and_removal_cost(f, regions, selected)
    # 步骤1：主体固定；只从原输入之外的粗帧提议，不重加已移除旧帧，不新增采样或评分。
    for _ in range(4):
        removable = (selected & original) - protected
        if not removable:
            break
        proposals = []
        for r in regions:
            ids = [i for i in r['members'] if i < len(gain) and i not in original and i not in selected and gain[i] > 0]
            if not ids:
                continue
            local = sorted(set(r['members']) & selected)
            redundancy = np.clip(f[ids] @ f[local].T, 0, 1).max(axis=1)
            for i, d in zip(ids, redundancy):
                score = float(gain[i]) * math.sqrt(max(0., 1 - float(d)))
                if score > 0:
                    proposals.append(dict(index=i, region_id=r['region_id'], gain=float(gain[i]), redundancy=float(d), score=score))
        if not proposals:
            break
        add = min(proposals, key=lambda p: (-p['score'], *frame_key(candidates[p['index']])))
        before, _ = coverage_and_removal_cost(f, regions, selected)
        # 步骤2：加入一张才删一张；最小化原覆盖损失，同分先删晚帧，锚点与已加帧不删除。
        enlarged = selected | {add['index']}
        enlarged_value, costs = coverage_and_removal_cost(f, regions, enlarged)
        remove = min(removable, key=lambda i: (costs[i], -candidates[i]['source_pts'], -i))
        selected = enlarged - {remove}
        protected.add(add['index'])
        after, _ = coverage_and_removal_cost(f, regions, selected)
        if abs((enlarged_value - costs[remove]) - after) > 1e-6:
            raise ProtocolError('Removal cost mismatch')
        steps.append(dict(addition=add, removed_index=remove, removal_cost=costs[remove],
                          coverage_before=before, coverage_with_addition=enlarged_value, coverage_after=after))
    # 步骤3：源时间排序并闭合预算；只记录事实上的配额变化，不假称仍是原DP解。
    if len(selected) != len(original) or not anchors <= selected or len(selected - original) > 4:
        raise ProtocolError('Bounded intervention invariant failed')
    final, _ = coverage_and_removal_cost(f, regions, selected)
    return sorted(selected, key=lambda i: frame_key(candidates[i])), dict(steps=steps, protected_indices=sorted(anchors),
        added_indices=sorted(selected - original), removed_indices=sorted(original - selected),
        baseline_coverage=baseline, final_coverage=final,
        region_budgets={str(r['region_id']): len(set(r['members']) & selected) for r in regions},
        scope='Fixed four-swap offline candidate; no QA guarantee; visual coverage is not semantic evidence')
