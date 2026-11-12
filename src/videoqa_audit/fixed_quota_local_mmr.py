"""固定父区域、锚点和实际配额的官方局部MMR；不读取答案。"""
from numbers import Integral
import numpy as np

from .selection_structure import (ROOT, PARENT, GLOBAL, WFS, REFERENCE_SHA256,
    reference_core, validate_coarse, _mmr_steps)

CONTROL = 'CTRL-RegionMMR-FixedQuota-S1'
CONTROLS = (CONTROL,)


def parent_partition(coarse_pool, parent_trace, frame_budget):
    """接口：只读取最终region_budgets，校验粗成员分区并重构父前缀实际输入。"""
    # 步骤1禁止用reference_quotas等可行参照代替最终配额。
    if 'region_budgets' not in parent_trace:
        raise ValueError('Actual region_budgets required; feasible references are not quotas')
    regions = parent_trace['coarse_prefix_trace']['regions']
    quotas = parent_trace['region_budgets']
    if set(quotas) != {str(r['region_id']) for r in regions}:
        raise ValueError('Actual quota region keys differ')
    n = coarse_pool['initial_count']; seen = set(); old = []; result = []
    for region in regions:
        rid = str(region['region_id']); members = list(region['members'])
        refs = region['reference_coarse_indices']; prefix = list(region['prefix'])
        if (not members or len(members) != len(set(members)) or set(members) != set(refs)
                or any(not isinstance(i, Integral) or not 0 <= i < n for i in members)
                or seen.intersection(members)):
            raise ValueError('Region member mapping differs')
        seen.update(members); q = quotas[rid]
        if (isinstance(q, bool) or not isinstance(q, Integral) or not 1 <= q <= len(members)
                or len(prefix) < q or len(prefix) != len(set(prefix))
                or not set(prefix) <= set(members)):
            raise ValueError('Actual quota or parent prefix is invalid')
        anchor = prefix[0]
        if anchor not in parent_trace['protected_indices']:
            raise ValueError('Parent prefix anchor is not protected')
        old.extend(prefix[:q])
        result.append(dict(region_id=rid, members=sorted(members,
            key=lambda i:(coarse_pool['candidates'][i]['source_pts'], i)),
            quota=int(q), anchor=int(anchor)))
    # 步骤2区域必须完整划分粗池，实际预算与保护集合必须闭合。
    if (seen != set(range(n)) or len(old) != min(frame_budget,n)
            or len(set(old)) != len(old)
            or {r['anchor'] for r in result} != set(parent_trace['protected_indices'])):
        raise ValueError('Parent partition, anchors or actual budget differs')
    return result, sorted(old,key=lambda i:(coarse_pool['candidates'][i]['source_pts'],i))


def select_region_mmr_fixed_quota(coarse_pool, features, parent_trace, frame_budget, *, _reference=None):
    """接口：合法粗池、FP32视觉特征、父实际轨迹、K→源索引及局部MMR轨迹。

    仅替换区内规则；不计算覆盖曲线、不求解DP、不接收标签。_reference仅供同SHA测试。
    """
    scores, f, target = validate_coarse(coarse_pool, features, frame_budget)
    regions, parent_ids = parent_partition(coarse_pool,parent_trace,frame_budget)
    core = reference_core(_reference); normalized = 2.*scores-1.
    selected = []; traces = []
    # 步骤1每区独立调用固定官方实现，直接选到父实际配额，保留负余弦和原生同分。
    for region in regions:
        ids = region['members']; local_f = f[ids]; local_r = normalized[ids]
        raw = list(map(int,core.WFSFrameSelector(lambda_param=.5).select_from_segment(
            (0,len(ids)),region['quota'],local_r,local_f)))
        if (len(raw) != region['quota'] or len(set(raw)) != len(raw)
                or any(not 0 <= i < len(ids) for i in raw)):
            raise ValueError('Official local selection violated quota')
        chosen = [ids[i] for i in raw]
        if chosen[0] != region['anchor']:
            raise ValueError('Official first item differs from parent anchor; pause')
        steps = _mmr_steps(core,raw,local_r,local_f)
        for step in steps:
            step['local_index'] = step['candidate_index']
            step['candidate_index'] = ids[step['local_index']]
        selected.extend(chosen)
        traces.append(dict(region,selection_order=chosen,steps=steps))
    # 步骤2只恢复源时间顺序；不跨区去重、不追加锚点或补位回退。
    selected.sort(key=lambda i:(coarse_pool['candidates'][i]['source_pts'],i))
    if len(selected) != target or len(set(selected)) != target:
        raise ValueError('Final actual budget not closed')
    return selected, dict(control_id=CONTROL,parent_method=PARENT,frame_budget=frame_budget,
        actual_frame_count=target,reference_sha256=REFERENCE_SHA256,lambda_param=.5,
        score_mapping='2*r-1',similarity='official sklearn cosine; no clipping',
        regions=traces,region_budgets=dict(parent_trace['region_budgets']),
        parent_selected_indices=parent_ids,selected_indices=selected,dp_calls=0,
        new_model_calls=0,limitation='Local-rule replacement conditional on parent actual quotas; coarse only')
