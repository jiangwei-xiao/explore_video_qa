"""E1：固定区域名额，将局部平方距离尺度换为距离尺度；仅用于离线归因。"""
import math
from fractions import Fraction
import numpy as np
from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_methods.density_selection import validate_candidates,source_time,frame_key


def select_local_distance(candidates,features,plan,video,cfg,quotas):
    """接口：外部固定配额下按L*sqrt(u)选帧；不重新决定跨区预算，不读取答案。"""
    validate_candidates(candidates,video);f=np.asarray(features,dtype=np.float32)
    if f.ndim!=2 or len(f)!=len(candidates) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5,rtol=0):
        raise ProtocolError('Invalid normalized features')
    if set(quotas)!={str(r['region_id']) for r in plan['regions']}:raise ProtocolError('Quota scope mismatch')
    picked={};eligible=[];trace=[]
    # 步骤1：只读取冻结区域资格，名额包含原保护锚点，不能借用或合并预算。
    for region in plan['regions']:
        rid=region['region_id'];left,right=Fraction(region['left_fraction']),Fraction(region['right_fraction'])
        members=sorted([i for i,r in enumerate(candidates) if left<=source_time(r,video)<=right],key=lambda i:frame_key(candidates[i]))
        budget=quotas[str(rid)];anchor=region['anchor_index']
        if anchor not in members or not set(region['seed_indices'])<=set(members) or set(members)&set(eligible):raise ProtocolError('Invalid region membership')
        if not isinstance(budget,int) or not 1<=budget<=len(members):raise ProtocolError('Invalid fixed quota')
        eligible.extend(members);selected=[anchor];picked[str(rid)]=selected
        support=np.maximum.reduce([np.clip(f[members]@f[j],0,1) for j in region['seed_indices']])
        # 步骤2：每次完整重算本区域重复度，唯一变化为sqrt(1-D)，不增加指数搜索。
        while len(selected)<budget:
            redundant=np.maximum.reduce([np.clip(f[members]@f[j],0,1) for j in selected])
            proposals=[]
            for position,i in enumerate(members):
                if i in selected:continue
                L=float(support[position]);D=float(redundant[position]);u=1-D
                proposals.append(dict(candidate_index=i,support=L,redundancy=D,squared_distance=u,
                    distance=math.sqrt(u),local_score=L*math.sqrt(u),old_local_score=L*u))
            winner=min(proposals,key=lambda p:(-p['local_score'],*frame_key(candidates[p['candidate_index']])))
            trace.append(dict(region_id=rid,selected_before=list(selected),winner=winner,proposals=proposals))
            selected.append(winner['candidate_index'])
    # 步骤3：恢复真实源时间顺序，预算和锚点均须保持外部固定条件。
    indices=sorted([i for group in picked.values() for i in group],key=lambda i:frame_key(candidates[i]))
    if len(indices)!=min(cfg['frame_budget'],len(eligible)) or len(set(indices))!=len(indices):raise ProtocolError('Fixed total budget mismatch')
    return indices,dict(local_rule='L*sqrt(u)',budget_rule='externally frozen D1 quotas, not reallocated',
        protected_indices=[r['anchor_index'] for r in plan['regions']],region_selected=picked,
        region_budgets={k:len(v) for k,v in picked.items()},eligible_indices=sorted(eligible),rounds=trace)
