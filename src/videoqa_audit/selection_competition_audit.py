"""冻结前缀竞争补审：仅从缓存特征恢复L、D与实际胜者，不执行新选帧。"""
import math
import numpy as np


def audit_competition(pool, features, prefix_trace, weighted_trace, targets, tolerance=1e-6):
    """接口：返回指定粗见证的竞争轨迹、两配额终点和视觉代表；不接收答案。

    features已按当前池映射，targets为当前池编号。轨迹中的winner来自冻结前缀，
    只核验数值与已保存步骤一致，不把最高余弦帧标成语义等价或算法替换建议。
    """
    candidates = pool['candidates']
    f = np.asarray(features, dtype=np.float32)
    ids = sorted(set(targets))
    if f.ndim != 2 or len(f) != len(candidates) or any(i < 0 or i >= len(f) for i in ids):
        raise ValueError('Target/feature identity differs')
    region_map = {}
    for region in prefix_trace['regions']:
        for i in region['reference_coarse_indices']:
            if i in region_map:
                raise ValueError('Coarse witness has multiple owners')
            region_map[i] = region
    if any(i not in region_map for i in ids):
        raise ValueError('This audit accepts initial coarse references only')
    # 步骤1：固定种子支持与前缀；每区只做目标/前缀的小矩阵，不建立全片N×N矩阵。
    records = []
    max_error = 0.0
    for region in prefix_trace['regions']:
        current = [i for i in ids if region_map[i]['region_id'] == region['region_id']]
        if not current:
            continue
        plan = next(r for r in pool['plan']['regions'] if r['region_id'] == region['region_id'])
        seeds = plan['seed_indices']
        prefix = region['prefix']
        score = pool['initial_scores']
        peak = max(score[j] for j in seeds)
        # 沿用原GEMV形状；改成小矩阵GEMM会在cos≈1时放大sqrt的舍入差异。
        rows = region.get('members', sorted(set(current + prefix)))
        pos = {i: j for j, i in enumerate(rows)}
        supports = np.maximum.reduce([
            (score[j] / peak if peak > 0 else 1.) * np.clip(f[rows] @ f[j], 0, 1)
            for j in seeds])
        dots = np.stack([np.clip(f[rows] @ f[i], 0, 1) for i in prefix], axis=1)
        before = np.maximum.accumulate(dots, axis=1)
        steps = []
        for step, winner in enumerate(prefix[1:], 1):
            j = pos[winner]
            L, D = float(supports[j]), float(before[j, step-1])
            actual = dict(candidate_index=winner, support=L, redundancy=D,
                          local_score=L*math.sqrt(max(0., 1-D)))
            stored = region['local_steps'][step-1]
            if stored['candidate_index'] != winner:
                raise ValueError('Stored step and prefix disagree')
            for key in ('support', 'redundancy', 'local_score'):
                error = abs(actual[key] - stored[key])
                max_error = max(max_error, error)
                if error > tolerance:
                    raise ValueError('Frozen feature/competition component mismatch')
            steps.append(actual)
        # 步骤2：每个目标只在尚未入选时比较，保护锚点和零分填额必须分别解释。
        for i in current:
            j = pos[i]
            selected_position = prefix.index(i)+1 if i in prefix else None
            trajectory = []
            for step, winner in enumerate(steps, 1):
                if i in prefix[:step]:
                    break
                L, D = float(supports[j]), float(before[j, step-1])
                target_score = L*math.sqrt(max(0., 1-D))
                if target_score > winner['local_score'] + tolerance:
                    raise ValueError('Unselected target exceeds recorded actual winner')
                trajectory.append(dict(selected_before=prefix[:step], target_support=L,
                    target_redundancy=D, target_local_score=target_score,
                    actual_winner=winner, score_margin=winner['local_score']-target_score))
            parents = {}
            for name, trace in [('exp07', prefix_trace), ('exp12', weighted_trace)]:
                quota = trace['region_budgets'][str(region['region_id'])]
                chosen = prefix[:quota]
                sims = np.clip(f[chosen] @ f[i], 0, 1)
                representative = chosen[int(np.argmax(sims))]
                retained = i in chosen
                if selected_position == 1:
                    terminal = dict(mode='protected_anchor', competition=None)
                elif retained:
                    terminal = dict(mode='selected_at_prefix_position', competition=trajectory[selected_position-2])
                elif quota == 1:
                    terminal = dict(mode='protected_anchor_only_no_supplement_slot', competition=None)
                else:
                    terminal = dict(mode='lost_at_last_available_slot', competition=trajectory[quota-2])
                parents[name] = dict(quota=quota, retained=retained, terminal=terminal,
                    representative_index=representative,
                    representative_source_frame=candidates[representative]['source_frame_index'],
                    representative_source_pts=candidates[representative]['source_pts'],
                    representative_seconds=candidates[representative]['timestamp_seconds'],
                    representative_cosine=float(np.max(sims)), semantic_equivalence='not_inferred')
            records.append(dict(candidate_index=i, source_frame_index=candidates[i]['source_frame_index'],
                source_pts=candidates[i]['source_pts'], seconds=candidates[i]['timestamp_seconds'],
                itm=score[i], region_id=region['region_id'], prefix_position=selected_position,
                support=float(supports[j]), trajectory=trajectory, parents=parents))
    # 步骤3：完整轨迹保留在独立补充产物，原池、前缀、配额与选择文件不写入。
    if len(records) != len(ids):
        raise ValueError('Target trajectory population differs')
    return dict(records=records, max_component_error=max_error, new_model_calls=0,
                limitation='Competition of frozen visual proxies, not semantic evidence sufficiency')
