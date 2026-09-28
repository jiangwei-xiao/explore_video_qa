"""RD-1.2-exp05：预算与最终选择共享同一检索支持信号，其他公式保持不变。"""
from fractions import Fraction
import numpy as np
from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_methods.density_selection import validate_candidates,source_time,frame_key
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.coarse_eligibility import coarse_selection_cells,select_global_coarse


def allocate_weighted_support(candidates,features,plan,video,cfg,initial_scores):
    """接口：仅把第一遍L替换为第二遍已有的L_r，保留u、密度递减和稳定排序。"""
    validate_candidates(candidates,video);features=np.asarray(features,dtype=np.float32)
    if features.ndim!=2 or len(features)!=len(candidates) or not np.isfinite(features).all():raise ProtocolError('Invalid features')
    if not np.allclose(np.linalg.norm(features,axis=1),1,atol=1e-5,rtol=0):raise ProtocolError('Features not normalized')
    selected={};members={};support={};redundancy={};nearest={};weights={}
    # 步骤1：支持参照固定为原种子；与最终选择调用同一weighted_support接口。
    for region in plan['regions']:
        rid=region['region_id'];left=Fraction(region['left_fraction']);right=Fraction(region['right_fraction'])
        ids=sorted([i for i,r in enumerate(candidates) if left<=source_time(r,video)<=right],key=lambda i:frame_key(candidates[i]))
        if region['anchor_index'] not in ids or not set(region['seed_indices'])<=set(ids):raise ProtocolError('Lost seeds')
        if any(set(ids)&set(v) for v in members.values()):raise ProtocolError('Overlapping regions')
        members[rid]=ids;selected[rid]=[region['anchor_index']]
        support[rid],weights[rid],_=weighted_support(features,ids,region['seed_indices'],initial_scores)
        redundancy[rid]=np.clip(features[ids]@features[region['anchor_index']],0,1)
        nearest[rid]=np.full(len(ids),region['anchor_index'],dtype=np.int64)
    target=min(cfg['frame_budget'],sum(map(len,members.values())))
    if len(selected)>target:raise ProtocolError('Anchor capacity exceeded')
    trace=[]
    # 步骤2：仍按(1+(w-1)/n)×L_r×u分配，唯一替换项是支持度，不增加参数。
    while sum(map(len,selected.values()))<target:
        proposals=[]
        for region in plan['regions']:
            rid=region['region_id'];remaining=[j for j,i in enumerate(members[rid]) if i not in selected[rid]]
            if not remaining:continue
            utility=lambda j:float(support[rid][j])*(1-float(redundancy[rid][j]))
            j=min(remaining,key=lambda j:(-utility(j),*frame_key(candidates[members[rid][j]])))
            count=len(selected[rid]);effective=1+(region['weight']-1)/count
            if region['weight']<1:raise ProtocolError('Invalid density weight')
            proposals.append(dict(region_id=rid,candidate_index=members[rid][j],support=float(support[rid][j]),
                redundancy=float(redundancy[rid][j]),diversity=1-float(redundancy[rid][j]),utility=utility(j),
                weight=region['weight'],selected_before=count,effective_weight=effective,priority=effective*utility(j),nearest_selected_index=int(nearest[rid][j])))
        if not proposals:raise ProtocolError('No candidates before budget filled')
        winner=min(proposals,key=lambda p:(-p['priority'],*frame_key(candidates[p['candidate_index']])))
        rid=winner['region_id'];i=winner['candidate_index'];selected[rid].append(i)
        trace.append(dict(step=len(trace),proposals=proposals,winner=winner))
        # 步骤3：只更新本区域重复度；最终第二遍仍使用L_r×sqrt(u)，不改距离尺度。
        values=np.clip(features[members[rid]]@features[i],0,1);changed=values>redundancy[rid]
        nearest[rid][changed]=i;redundancy[rid]=np.maximum(redundancy[rid],values)
    indices=sorted([i for group in selected.values() for i in group],key=lambda i:frame_key(candidates[i]))
    return indices,dict(protected_indices=[r['anchor_index'] for r in plan['regions']],rounds=trace,
        eligible_indices=sorted(i for ids in members.values() for i in ids),region_selected={str(k):v for k,v in selected.items()},
        region_budgets={str(k):len(v) for k,v in selected.items()},effective_frame_budget=target,
        local_rule='L_r*u',global_rule='(1+(w-1)/n)*L_r*u',
        supports={str(rid):dict(zip(map(str,members[rid]),map(float,support[rid]))) for rid in members})


def select_consistent_support(candidates,features,plan,video,cfg,reference_quotas,initial_scores):
    """接口：复用exp04候选资格，第一遍支持统一后重算配额，第二遍逐行复用旧实现。"""
    cells=coarse_selection_cells(plan,video)
    # 步骤1：选择域与exp04完全一致，原细采样与特征资产不变。
    _,allocation=allocate_weighted_support(candidates,features,cells,video,cfg,initial_scores)
    # 步骤2：改变支持信号后由原过程产生新配额，不手工设置各题或各区域预算。
    selected,trace=select_global_coarse(candidates,features,plan,video,cfg,allocation['region_budgets'],initial_scores)
    trace.update(budget_rule='shared weighted seed support across both passes; density decay and distance scales unchanged',
        budget_trace=allocation,reference_quotas=reference_quotas,
        quota_changes={rid:allocation['region_budgets'][rid]-reference_quotas[rid] for rid in reference_quotas})
    return selected,trace
