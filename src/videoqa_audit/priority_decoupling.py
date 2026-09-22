"""P1离线归因：解耦局部候选排序和跨区域预算；不改正式两方法。"""
from fractions import Fraction
import numpy as np
from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_methods.density_selection import validate_candidates,source_time,frame_key


def select_decoupled(candidates, features, plan, video, cfg):
    """接口：区域内以L×u提出候选，跨区以w×u竞争；只作单因素离线诊断。"""
    validate_candidates(candidates,video)
    features=np.asarray(features,dtype=np.float32)
    if features.ndim!=2 or features.shape[0]!=len(candidates) or not np.isfinite(features).all():
        raise ProtocolError('Invalid feature matrix')
    if not np.allclose(np.linalg.norm(features,axis=1),1,atol=1e-5,rtol=0):
        raise ProtocolError('Features not normalized')
    selected={};members={};support={};redundancy={};nearest={}
    # 步骤1：参照集合来自原始种子，固定不扩张；新旧来源不参与评分。
    for region in plan['regions']:
        rid=region['region_id'];left=Fraction(region['left_fraction']);right=Fraction(region['right_fraction'])
        ids=sorted([i for i,r in enumerate(candidates) if left<=source_time(r,video)<=right],key=lambda i:frame_key(candidates[i]))
        if region['anchor_index'] not in ids or not set(region['seed_indices'])<=set(ids):
            raise ProtocolError('Region lost its original seeds')
        if any(set(ids)&set(v) for v in members.values()):
            raise ProtocolError('Overlapping final regions')
        members[rid]=ids;selected[rid]=[region['anchor_index']]
        support[rid]=np.maximum.reduce([np.clip(features[ids]@features[j],0,1) for j in region['seed_indices']])
        redundancy[rid]=np.clip(features[ids]@features[region['anchor_index']],0,1)
        nearest[rid]=np.full(len(ids),region['anchor_index'],dtype=np.int64)
    target=min(cfg['frame_budget'],sum(map(len,members.values())))
    if len(selected)>target:raise ProtocolError('Anchor capacity exceeded')
    trace=[]
    # 步骤2：区域内仍最大L*u；唯一改变是跨区优先级w*u，不再次使用L决定区域名额。
    while sum(map(len,selected.values()))<target:
        proposals=[]
        for region in plan['regions']:
            rid=region['region_id']
            remaining=[j for j,i in enumerate(members[rid]) if i not in selected[rid]]
            if not remaining:continue
            utility=lambda j:float(support[rid][j])*(1-float(redundancy[rid][j]))
            j=min(remaining,key=lambda j:(-utility(j),*frame_key(candidates[members[rid][j]])))
            proposals.append(dict(region_id=rid,candidate_index=members[rid][j],support=float(support[rid][j]),
                redundancy=float(redundancy[rid][j]),diversity=1-float(redundancy[rid][j]),utility=utility(j),
                weight=region['weight'],priority=region['weight']*(1-float(redundancy[rid][j])),nearest_selected_index=int(nearest[rid][j])))
        if not proposals:raise ProtocolError('No proposal before budget completed')
        winner=min(proposals,key=lambda p:(-p['priority'],*frame_key(candidates[p['candidate_index']])))
        rid=winner['region_id'];i=winner['candidate_index'];selected[rid].append(i)
        trace.append(dict(step=len(trace),proposals=proposals,winner=winner))
        # 步骤3：只更新已选集合的重复度；L保持固定，不让选中离群画面扩张支持集合。
        values=np.clip(features[members[rid]]@features[i],0,1);changed=values>redundancy[rid]
        nearest[rid][changed]=i;redundancy[rid]=np.maximum(redundancy[rid],values)
    indices=sorted([i for group in selected.values() for i in group],key=lambda i:frame_key(candidates[i]))
    return indices,dict(protected_indices=[r['anchor_index'] for r in plan['regions']],rounds=trace,
        eligible_indices=sorted(i for ids in members.values() for i in ids),
        region_selected={str(k):v for k,v in selected.items()},region_budgets={str(k):len(v) for k,v in selected.items()},
        effective_frame_budget=target,local_rule='L*u',global_rule='w*u',
        supports={str(rid):dict(zip(map(str,members[rid]),map(float,support[rid]))) for rid in members})
