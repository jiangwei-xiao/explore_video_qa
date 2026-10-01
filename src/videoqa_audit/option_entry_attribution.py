"""入口对照的零调用归因：区分既有前缀优先级与区域名额，不将新帧当成真值。"""
import numpy as np


def prefix_gate(index, trace, budget):
    """接口：在冻结exp07轨迹中定位候选，判断单独调整配额能否选到它。"""
    owners = [r for r in trace['regions'] if index in r['members']]
    if len(owners) != 1:
        raise ValueError('Expected exactly one frozen selection cell')
    region = owners[0]; quota = trace['region_budgets'][str(region['region_id'])]
    # 步骤1：其余区域各至少一个锚点，因此本区最多得到K-(R-1)，不是K。
    maximum = min(len(region['prefix']), budget-len(trace['regions'])+1)
    if not 1 <= quota <= maximum:
        raise ValueError('Infeasible recorded allocation')
    position = region['prefix'].index(index)+1 if index in region['prefix'] else None
    # 步骤2：只判单个候选的机械可达性；不宣称全部候选可同时纳入或能改善答案。
    if position is not None and position <= quota:
        status = 'selected'
    elif position is not None and position <= maximum:
        status = 'quota_reachable'
    else:
        status = 'local_prefix_blocked'
    return dict(region_id=region['region_id'],prefix_position=position,quota=quota,
                maximum_feasible_quota=maximum,status=status)


def nearest_similarities(features, selected):
    """接口：每帧与其他已选帧的最大截断CLS余弦，仅为外观相似代理。"""
    f = np.asarray(features,dtype=np.float32)
    if len(selected) != len(set(selected)) or not selected:
        raise ValueError('Expected unique nonempty selection')
    # 步骤1：排除自身；单帧集合没有重复参照，约定为0。
    if len(selected) == 1:
        return [0.]
    scores = np.clip(f[selected]@f[selected].T,0,1)
    np.fill_diagonal(scores,0)
    # 步骤2：不设过滤阈值，也不将微小文字差异当作无用画面。
    return scores.max(axis=1).astype(float).tolist()
