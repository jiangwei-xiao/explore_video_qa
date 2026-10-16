"""RD-1.2-exp34：固定exp33邻接提议，仅选择粗帧覆盖损失最小的删除对象。"""
from __future__ import annotations

import math
import numpy as np

from videoqa_methods.adjacent_context_repair import select_adjacent_repair, source_key

TOLERANCE = 1e-6
REFERENCE_BLOCK = 1024


def _stable_key(candidates, index):
    """接口：按真实源PTS、候选编号及数组编号稳定决定数值并列。"""
    frame = candidates[index]
    return (int(frame['source_pts']), int(frame.get('candidate_index', index)), index)


def _coarse_coverage_losses(features, coarse_count, selected, added):
    """接口：返回F(S)、F(S+r)及每个已选帧的删除损失；粗参考等权。

    分块只构造粗参考×(K+1)矩阵。对每个参考，只有其唯一最佳代表被删除
    时才产生“最佳-次佳”的损失；并列最佳的损失自然为零。
    """
    pool_indices = list(selected) + [added]
    losses = np.zeros(len(pool_indices), dtype=np.float64)
    old_sum = augmented_sum = 0.0
    for start in range(0, coarse_count, REFERENCE_BLOCK):
        ref = features[start:min(coarse_count, start + REFERENCE_BLOCK)]
        sim = np.clip(ref @ features[pool_indices].T, 0.0, 1.0)
        if not np.isfinite(sim).all():
            raise ValueError('Non-finite coarse-reference similarity')
        old_sum += float(np.max(sim[:, :-1], axis=1).sum(dtype=np.float64))
        best_columns = np.argmax(sim, axis=1)
        best_values = sim[np.arange(len(sim)), best_columns]
        augmented_sum += float(best_values.sum(dtype=np.float64))
        # 步骤1：仅有唯一最佳代表被移除时，覆盖值才会降低。
        sim[np.arange(len(sim)), best_columns] = -1.0
        runner_up = np.max(sim, axis=1)
        deltas = np.asarray(best_values - runner_up, dtype=np.float64)
        losses += np.bincount(best_columns, weights=deltas,
                              minlength=len(pool_indices))
    scale = float(coarse_count)
    return old_sum / scale, augmented_sum / scale, losses / scale


def select_low_loss_replacement(pool, features, selected_indices, exp33_trace, K):
    """接口：复用冻结exp33请求，返回同预算最终编号和完整删除追踪。

    输入不包含问题答案、人工事实或历史对错；本函数不采样、不调用模型。
    当原规则无提议时，输出与父输入相同。
    """
    candidates = pool['candidates']
    f = np.asarray(features, dtype=np.float32)
    selected = list(selected_indices)
    if not isinstance(K, int) or isinstance(K, bool) or K < 1:
        raise ValueError('Invalid frame budget')
    if f.ndim != 2 or len(f) != len(candidates) or not np.isfinite(f).all():
        raise ValueError('Candidate/feature identity differs')
    coarse_count = int(pool['initial_count'])
    if not 0 < coarse_count <= len(f):
        raise ValueError('Invalid coarse prefix')

    # 步骤1：用原选择器重放提议，验证源文件记录的请求帧、补帧及删除槽位。
    old_final, replay = select_adjacent_repair(pool, f, selected, K)
    old_winner, saved_winner = replay['winner'], exp33_trace.get('winner')
    if replay['parent_indices'] != exp33_trace.get('parent_indices'):
        raise ValueError('Frozen exp33 parent selection differs')
    if (old_winner is None) != (saved_winner is None):
        raise ValueError('Frozen exp33 trigger differs')
    if old_winner is not None:
        for key in ('anchor_index', 'added_index', 'replaced_index',
                    'anchor_pts', 'added_pts', 'replaced_pts', 'replaced_slot'):
            if old_winner[key] != saved_winner.get(key):
                raise ValueError('Frozen exp33 proposal differs: ' + key)
    if old_winner is None:
        return selected, dict(changed=False, reason='no_frozen_exp33_proposal',
                              exp33_indices=old_final, final_indices=selected,
                              original_exp33_winner=None)

    anchor = old_winner['anchor_index']
    added = old_winner['added_index']
    previous_donor = old_winner['replaced_index']
    protected = set(replay['protected_indices']) | {anchor}
    deletable = [i for i in selected if i not in protected]
    if previous_donor not in deletable or added in selected or not deletable:
        raise ValueError('Frozen proposal has no legal deletion candidate')

    # 步骤2：新增帧固定，只比较删除后粗候选的等权视觉覆盖损失。
    parent_cover, augmented_cover, all_losses = _coarse_coverage_losses(
        f, coarse_count, selected, added)
    losses = {i: float(all_losses[pos]) for pos, i in enumerate(selected)
              if i in deletable}
    if any(not math.isfinite(x) or x < -TOLERANCE for x in losses.values()):
        raise ValueError('Invalid coverage deletion loss')
    minimum = min(losses.values())
    tied = [i for i in deletable if losses[i] <= minimum + TOLERANCE]
    donor = previous_donor if previous_donor in tied else min(
        tied, key=lambda i: _stable_key(candidates, i))
    selected_cover = augmented_cover - losses[donor]
    exp33_cover = augmented_cover - losses[previous_donor]
    if selected_cover + TOLERANCE + 1e-12 < exp33_cover:
        raise ValueError('Coverage below frozen exp33 within tolerance')

    # 步骤3：只增删各一张，恢复真实时间顺序并检查所有保护帧仍在。
    final = sorted([i for i in selected if i != donor] + [added],
                   key=lambda i: (*source_key(candidates[i]), i))
    if len(final) != len(selected) or len(set(final)) != len(selected):
        raise ValueError('Source-frame budget or uniqueness changed')
    if not protected <= set(final):
        raise ValueError('Protected retrieval frame was removed')
    donor_records = [dict(index=i, source_pts=int(candidates[i]['source_pts']),
                          candidate_number=int(candidates[i].get('candidate_index', i)),
                          coverage_loss=losses[i])
                     for i in sorted(deletable, key=lambda i: _stable_key(candidates, i))]
    trace = dict(changed=True, exp33_indices=old_final,
                 parent_indices=selected, final_indices=final,
                 original_exp33_winner=old_winner,
                 protected_indices=sorted(protected),
                 added_index=added, previous_donor_index=previous_donor,
                 chosen_donor_index=donor, donor_changed=donor != previous_donor,
                 donor_losses=donor_records, minimum_coverage_loss=minimum,
                 original_donor_loss=losses[previous_donor],
                 chosen_donor_loss=losses[donor], tie_tolerance=TOLERANCE,
                 coarse_references=coarse_count,
                 coverage_parent=parent_cover,
                 coverage_with_added=augmented_cover,
                 coverage_exp33=exp33_cover, coverage_final=selected_cover)
    return final, trace
