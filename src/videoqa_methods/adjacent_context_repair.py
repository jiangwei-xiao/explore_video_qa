"""RD-1.2-exp33：在冻结父输入上只修复一个未受保护的后继槽位。"""
from collections import Counter
from fractions import Fraction
import math
import numpy as np

TOLERANCE = 1e-6
TARGET_STEP = Fraction(1, 2)


def source_key(frame):
    """接口：源PTS与源编号共同确定身份，候选数组编号不作为跨池身份。"""
    return int(frame['source_pts']), int(frame['source_frame_index'])


def requested_time(frame):
    """接口：使用原采样目标的有理数，不用实际解码时间近似0.5秒网格。"""
    value = frame.get('requested_fraction', frame.get('requested_seconds'))
    if value is None:
        return None
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None


def propose_adjacent_repair(pool, features, selected_indices, K):
    """接口：返回全部合格提议及拒绝原因，不接收题面、答案或人工见证。

    输入为合法池、对应归一化视觉特征、父版有序候选编号及唯一公开预算K。
    提议按原问题分数、相对连续性和真实源身份稳定排序；来源不赋予替换优先级。
    """
    candidates = pool['candidates']; n = len(candidates); coarse = pool['initial_count']
    f = np.asarray(features, dtype=np.float32); selected = list(selected_indices)
    if not isinstance(K, int) or isinstance(K, bool) or K < 1 or not 1 <= len(selected) <= K:
        raise ValueError('Invalid public budget or parent selection')
    if f.ndim != 2 or len(f) != n or not np.isfinite(f).all():
        raise ValueError('Candidate/feature identity differs')
    if not np.allclose(np.linalg.norm(f, axis=1), 1, rtol=0, atol=1e-5):
        raise ValueError('Features must already be normalized')
    if not 0 < coarse <= n or len(pool['initial_scores']) != coarse:
        raise ValueError('Coarse prefix or scores differ')
    scores = pool['initial_scores']
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
        raise ValueError('Invalid frozen ITM probability')
    keys = [source_key(c) for c in candidates]
    if len({x[0] for x in keys}) != n or len({x[1] for x in keys}) != n:
        raise ValueError('Duplicate source frame in legal pool')
    if len(set(selected)) != len(selected) or any(not 0 <= i < n for i in selected):
        raise ValueError('Duplicate or invalid parent indices')
    if selected != sorted(selected, key=lambda i: keys[i]):
        raise ValueError('Parent selection is not in source order')
    seeds = set(pool['plan']['seed_indices'])
    anchors = {r['anchor_index'] for r in pool['plan']['regions']}
    if any(not 0 <= i < coarse for i in seeds) or not anchors <= set(selected):
        raise ValueError('Seed identity or protected parent anchor differs')
    protected = anchors | (seeds & set(selected))
    # 步骤1：建立真实时间邻接；粗细候选的存储顺序不等于源时间顺序。
    order = sorted(range(n), key=lambda i: (*keys[i], i))
    positions = {i: j for j, i in enumerate(order)}
    chosen = set(selected); decisions = []; proposals = []; reasons = Counter()
    for slot, a in enumerate(selected):
        decision = dict(anchor_slot=slot, anchor_index=a)
        reason = None
        if a >= coarse:
            reason = 'request_frame_has_no_original_itm'
        elif slot+1 == len(selected):
            reason = 'no_following_input_slot'
        else:
            b = selected[slot+1]; pos = positions[a]
            decision.update(replaced_index=b, replaced_slot=slot+1)
            before = order[pos-1] if pos else None
            after = order[pos+1] if pos+1 < n else None
            if (before is not None and before in chosen) or (after is not None and after in chosen):
                reason = 'direct_time_neighbour_already_selected'
            elif b in protected:
                reason = 'following_slot_is_protected'
            elif after is None:
                reason = 'no_legal_successor'
            else:
                r = after; qa, qr = requested_time(candidates[a]), requested_time(candidates[r])
                decision['added_index'] = r
                if qa is None or qr is None:
                    reason = 'missing_original_sampling_target'
                elif qr-qa != TARGET_STEP:
                    reason = 'successor_not_next_half_second_target'
                elif not keys[a][0] < keys[r][0] < keys[b][0] or r in chosen:
                    reason = 'successor_outside_fixed_slot_or_repeated'
                else:
                    # 步骤2：只有两次向量点积，不构造全片N×N矩阵，也不重算视觉特征。
                    ar = float(np.clip(f[a] @ f[r], 0, 1))
                    ab = float(np.clip(f[a] @ f[b], 0, 1))
                    delta = ar-ab
                    decision.update(itm=float(scores[a]), similarity_adjacent=ar,
                                    similarity_previous_partner=ab, adjacency_gain=delta)
                    if delta <= TOLERANCE:
                        reason = 'no_positive_relative_visual_continuity'
                    else:
                        proposal = dict(decision, anchor_pts=keys[a][0], added_pts=keys[r][0],
                                        replaced_pts=keys[b][0])
                        proposals.append(proposal)
        decision['reason'] = reason or 'eligible'
        reasons[decision['reason']] += 1; decisions.append(decision)
    proposals.sort(key=lambda p: (-p['itm'], -p['adjacency_gain'], p['anchor_pts'], p['added_pts'],
                                 p['anchor_index'], p['added_index']))
    return dict(proposals=proposals, decisions=decisions, reason_counts=dict(reasons),
                protected_indices=sorted(protected), target_step_fraction=str(TARGET_STEP),
                numerical_tolerance=TOLERANCE, policy='one highest-ITM eligible retained coarse frame; no iterative repair')


def select_adjacent_repair(pool, features, selected_indices, K):
    """接口：返回最终编号与追踪；最多换一个槽，不重排、补预算或重跑父版DP。"""
    trace = propose_adjacent_repair(pool, features, selected_indices, K)
    selected = list(selected_indices); output = list(selected)
    # 步骤1：只应用事前固定排序的第一项；无可行提议自然返回完整父输入。
    winner = trace['proposals'][0] if trace['proposals'] else None
    if winner:
        output[winner['replaced_slot']] = winner['added_index']
    # 步骤2：保护集合、单槽位和源时间严格闭合；跨区配额变化交由外层单独记录。
    changed = [i for i, (a, b) in enumerate(zip(selected, output)) if a != b]
    if len(changed) != int(winner is not None) or len(set(output)) != len(output):
        raise ValueError('Single-slot source budget invariant violated')
    if not set(trace['protected_indices']) <= set(output):
        raise ValueError('A protected retrieval frame was removed')
    if output != sorted(output, key=lambda i: source_key(pool['candidates'][i])):
        raise ValueError('Replacement would reorder unchanged slots')
    trace.update(changed=bool(winner), winner=winner, changed_slots=changed,
                 parent_indices=selected, final_indices=output)
    return output, trace
