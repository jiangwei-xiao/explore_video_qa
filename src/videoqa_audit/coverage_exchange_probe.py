"""固定配额下一次覆盖交换的能力探针，不读取问题答案或人工见证。"""
import numpy as np


def best_single_exchange(similarity, weights, selected, anchor):
    """接口：输入粗参考×候选相似度、参考权重、局部选择及锚点，返回最佳一次交换。

    候选列须预先按源时间稳定排序；保持锚点、配额及原覆盖定义，不多轮迭代。
    返回的目标增益是视觉代理值，不是问答收益或证据充分性。
    """
    # 步骤1：检查定义域，禁止空参考、重复选择或不合法权重悄悄改变目标。
    sim = np.asarray(similarity)
    w = np.asarray(weights, dtype=np.float64)
    chosen = list(selected)
    if sim.ndim != 2 or not sim.shape[0] or len(w) != sim.shape[0]:
        raise ValueError('Reference shape mismatch')
    if not np.isfinite(sim).all() or np.any(sim < 0) or np.any(sim > 1):
        raise ValueError('Similarity must be finite in [0,1]')
    if not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1):
        raise ValueError('Reference weights must sum to one')
    if not chosen or len(set(chosen)) != len(chosen) or anchor not in chosen:
        raise ValueError('Invalid selected set or anchor')
    if any(i < 0 or i >= sim.shape[1] for i in chosen):
        raise ValueError('Selected index outside candidates')
    before = float(w @ sim[:, chosen].max(axis=1))
    free = [i for i in range(sim.shape[1]) if i not in set(chosen)]
    best = None
    # 步骤2：枚举移除对象，批量计算加入各候选后的同一个覆盖目标。
    for removed in sorted(i for i in chosen if i != anchor):
        remaining = [i for i in chosen if i != removed]
        covered = sim[:, remaining].max(axis=1)
        for start in range(0, len(free), 128):
            added = free[start:start + 128]
            values = w @ np.maximum(covered[:, None], sim[:, added])
            for candidate, value in zip(added, values):
                proposal = (float(value), -candidate, -removed)
                if float(value) > before and (best is None or proposal > best):
                    best = proposal
    # 步骤3：只执行一次严格改善，平局按加入源时间、移除源时间顺序决定。
    if best is None:
        return sorted(chosen), dict(changed=False, before=before, after=before, delta=0.)
    after, added, removed = best[0], -best[1], -best[2]
    result = sorted([i for i in chosen if i != removed] + [added])
    return result, dict(changed=True, before=before, after=after, delta=after-before,
                        added_local=added, removed_local=removed)
