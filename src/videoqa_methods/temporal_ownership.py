"""RD-1.2-exp06：跨原窗口间隙的时间归属支持，候选资格仍全局开放。"""
from fractions import Fraction
import math
import numpy as np
from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_methods.density_selection import validate_candidates,source_time,frame_key
from videoqa_methods.coarse_eligibility import coarse_selection_cells
from videoqa_methods.weighted_seed_support import weighted_support


def temporal_ownership(time,region_position,original_regions):
    """接口：相邻区间端点间线性分摊支持；是时间归属代理，不是答题证据概率。"""
    t=Fraction(time);r=original_regions[region_position]
    left,right=Fraction(r['left_fraction']),Fraction(r['right_fraction'])
    # 步骤1：原窗口内支持为1；前后只有一个可归属区域时不附加人为惩罚。
    if left<=t<=right:return 1.0
    if t<left:
        if region_position==0:return 1.0
        previous=Fraction(original_regions[region_position-1]['right_fraction'])
        value=(t-previous)/(left-previous)
    else:
        if region_position==len(original_regions)-1:return 1.0
        following=Fraction(original_regions[region_position+1]['left_fraction'])
        value=(following-t)/(following-right)
    # 步骤2：最近区域取得较大的重心坐标；中点同距时为1/2，不将任何帧硬剔除。
    if not Fraction(1,2)<=value<=1:raise ProtocolError('Candidate not owned by nearest original region')
    return float(value)


def select_temporal_ownership(candidates,features,plan,video,cfg,reference_quotas,initial_scores):
    """接口：仅加入几何支持T；保留exp04候选域、种子、密度、两遍L差异与u尺度。"""
    validate_candidates(candidates,video);f=np.asarray(features,dtype=np.float32)
    if f.ndim!=2 or len(f)!=len(candidates) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5,rtol=0):raise ProtocolError('Invalid normalized features')
    cells=coarse_selection_cells(plan,video);n=len(initial_scores)
    if not 1<=n<=len(candidates):raise ProtocolError('Invalid coarse prefix')
    members={};ownership={};unit={};weighted={};selected={};all_members=set();support_trace={}
    # 步骤1：原细帧保持原区域，只对区域间隙中的已有粗帧计算T∈[0.5,1]。
    for pos,r in enumerate(cells['regions']):
        rid=r['region_id'];left,right=Fraction(r['left_fraction']),Fraction(r['right_fraction'])
        ids=sorted([i for i,c in enumerate(candidates) if left<=source_time(c,video)<=right],key=lambda i:frame_key(candidates[i]))
        if all_members&set(ids) or r['anchor_index'] not in ids or not set(r['seed_indices'])<=set(ids):raise ProtocolError('Invalid selection cell')
        original=plan['regions'][pos];a,b=Fraction(original['left_fraction']),Fraction(original['right_fraction'])
        for i in ids:
            if i>=n and not a<=source_time(candidates[i],video)<=b:raise ProtocolError('Fine frame escaped original window')
        all_members.update(ids);members[rid]=ids;selected[rid]=[r['anchor_index']]
        ownership[rid]=np.array([temporal_ownership(source_time(candidates[i],video),pos,plan['regions']) for i in ids],dtype=np.float64)
        raw_unit=np.maximum.reduce([np.clip(f[ids]@f[j],0,1) for j in r['seed_indices']])
        raw_weighted,weights,_=weighted_support(f,ids,r['seed_indices'],initial_scores)
        unit[rid]=ownership[rid]*raw_unit;weighted[rid]=ownership[rid]*raw_weighted
        support_trace[str(rid)]=dict(ownership=dict(zip(map(str,ids),map(float,ownership[rid]))),
            unit_visual=dict(zip(map(str,ids),map(float,raw_unit))),weighted_visual=dict(zip(map(str,ids),map(float,raw_weighted))))
    if all_members!=set(range(len(candidates))):raise ProtocolError('Candidate eligibility lost')
    target=min(cfg['frame_budget'],len(candidates))
    if len(selected)>target:raise ProtocolError('Anchor capacity exceeded')
    allocation=[]
    # 步骤2：原第一遍仅乘T，不合并exp05的支持替换，避免同时改变两个因素。
    while sum(map(len,selected.values()))<target:
        proposals=[]
        for r in cells['regions']:
            rid=r['region_id'];ids=members[rid]
            D=np.maximum.reduce([np.clip(f[ids]@f[s],0,1) for s in selected[rid]])
            remaining=[j for j,i in enumerate(ids) if i not in selected[rid]]
            if not remaining:continue
            value=lambda j:float(unit[rid][j])*(1-float(D[j]))
            j=min(remaining,key=lambda j:(-value(j),*frame_key(candidates[ids[j]])))
            count=len(selected[rid]);effective=1+(r['weight']-1)/count
            if r['weight']<1:raise ProtocolError('Invalid density weight')
            proposals.append(dict(region_id=rid,candidate_index=ids[j],temporal_support=float(ownership[rid][j]),
                combined_support=float(unit[rid][j]),redundancy=float(D[j]),utility=value(j),selected_before=count,
                effective_weight=effective,priority=effective*value(j)))
        if not proposals:raise ProtocolError('No proposal before budget filled')
        winner=min(proposals,key=lambda p:(-p['priority'],*frame_key(candidates[p['candidate_index']])))
        allocation.append(dict(proposals=proposals,winner=winner));selected[winner['region_id']].append(winner['candidate_index'])
    quotas={str(r):len(s) for r,s in selected.items()};rounds=[];picked={}
    # 步骤3：原第二遍同样只乘T，仍在固定新配额下使用L_r×sqrt(u)。
    for r in cells['regions']:
        rid=r['region_id'];ids=members[rid];chosen=[r['anchor_index']];picked[str(rid)]=chosen
        while len(chosen)<quotas[str(rid)]:
            D=np.maximum.reduce([np.clip(f[ids]@f[s],0,1) for s in chosen]);proposals=[]
            for j,i in enumerate(ids):
                if i in chosen:continue
                score=float(weighted[rid][j])*math.sqrt(1-float(D[j]))
                proposals.append(dict(candidate_index=i,temporal_support=float(ownership[rid][j]),combined_support=float(weighted[rid][j]),redundancy=float(D[j]),local_score=score))
            winner=min(proposals,key=lambda p:(-p['local_score'],*frame_key(candidates[p['candidate_index']])))
            rounds.append(dict(region_id=rid,selected_before=list(chosen),winner=winner,proposals=proposals));chosen.append(winner['candidate_index'])
    result=sorted([i for group in picked.values() for i in group],key=lambda i:frame_key(candidates[i]))
    if len(result)!=target or len(set(result))!=target:raise ProtocolError('Invalid final budget')
    return result,dict(rule='barycentric temporal ownership on original-window gaps; exterior/single-region weight=1',
        eligible_indices=sorted(all_members),protected_indices=[r['anchor_index'] for r in plan['regions']],
        region_budgets=quotas,region_selected=picked,supports=support_trace,budget_rounds=allocation,rounds=rounds,
        reference_quotas=reference_quotas,quota_changes={r:quotas[r]-reference_quotas[r] for r in quotas})
