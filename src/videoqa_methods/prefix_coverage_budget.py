"""RD-1.2-exp07：依据实际局部选择前缀的粗帧覆盖，用动态规划分配总预算。"""
import math
from fractions import Fraction
import numpy as np
from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_methods.density_selection import validate_candidates,source_time,frame_key
from videoqa_methods.coarse_eligibility import coarse_selection_cells
from videoqa_methods.weighted_seed_support import weighted_support


def allocate_prefix_budget(curves,weights,budget):
    """接口：精确求解固定前缀族内的预算组合；每区至少保留一帧，不声称QA最优。"""
    if not curves or len(curves)!=len(weights) or budget<len(curves):raise ProtocolError('Invalid DP scope')
    for curve,w in zip(curves,weights):
        if len(curve)<2 or not math.isfinite(w) or w<=0 or not all(math.isfinite(x) and 0<=x<=1 for x in curve):raise ProtocolError('Invalid coverage curve')
        if any(b<a for a,b in zip(curve,curve[1:])):raise ProtocolError('Coverage must be non-decreasing')
    # 步骤1：D[r,b]枚举前r区用b帧的所有前缀长度；不假定各曲线具有凹性。
    states={0:(0.0,())}
    for curve,w in zip(curves,weights):
        updated={}
        for used,(value,allocation) in states.items():
            for n in range(1,min(len(curve)-1,budget-used)+1):
                proposal=(value+w*curve[n],allocation+(n,))
                # 同目标值时优先较早区域更多名额，对应嵌套前缀下更早的源帧集合。
                if used+n not in updated or proposal>updated[used+n]:updated[used+n]=proposal
        states=updated
    # 步骤2：必须闭合实际帧预算，不通过少选或重复源帧规避约束。
    if budget not in states:raise ProtocolError('Infeasible prefix budget')
    value,allocation=states[budget]
    return list(allocation),value


def select_prefix_coverage(candidates,features,plan,video,cfg,reference_quotas,initial_scores):
    """接口：局部L_r×sqrt(u)次序不变，只把预算评估换为实际前缀的视觉覆盖。"""
    validate_candidates(candidates,video);f=np.asarray(features,dtype=np.float32)
    if f.ndim!=2 or len(f)!=len(candidates) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5,rtol=0):raise ProtocolError('Invalid normalized features')
    ncoarse=len(initial_scores);cells=coarse_selection_cells(plan,video);target=min(cfg['frame_budget'],len(candidates))
    if not 1<=ncoarse<=len(candidates):raise ProtocolError('Invalid coarse prefix')
    all_members=set();records=[]
    # 步骤1：全局粗资格与原细帧不变；参考点仅是真实粗帧，不按虚构时间区间加权。
    for pos,r in enumerate(cells['regions']):
        rid=r['region_id'];left,right=Fraction(r['left_fraction']),Fraction(r['right_fraction'])
        ids=sorted([i for i,c in enumerate(candidates) if left<=source_time(c,video)<=right],key=lambda i:frame_key(candidates[i]))
        refs=[i for i in ids if i<ncoarse];original=plan['regions'][pos]
        if all_members&set(ids) or not refs or r['anchor_index'] not in ids or not set(r['seed_indices'])<=set(ids):raise ProtocolError('Invalid region membership')
        for i in ids:
            if i>=ncoarse and not Fraction(original['left_fraction'])<=source_time(candidates[i],video)<=Fraction(original['right_fraction']):raise ProtocolError('Fine candidate outside original window')
        all_members.update(ids);support,_,_=weighted_support(f,ids,r['seed_indices'],initial_scores)
        chosen=[r['anchor_index']];D=np.clip(f[ids]@f[chosen[0]],0,1)
        covered=np.clip(f[refs]@f[chosen[0]],0,1)
        curve=[0.0,float(np.mean(covered,dtype=np.float64))];steps=[]
        # 步骤2：枚举同一局部规则的嵌套前缀，不重新选择一套有利于覆盖指标的区域内帧。
        while len(chosen)<min(target,len(ids)):
            proposals=[]
            for j,i in enumerate(ids):
                if i in chosen:continue
                score=float(support[j])*math.sqrt(1-float(D[j]))
                proposals.append(dict(candidate_index=i,support=float(support[j]),redundancy=float(D[j]),local_score=score))
            winner=min(proposals,key=lambda p:(-p['local_score'],*frame_key(candidates[p['candidate_index']])))
            i=winner['candidate_index'];chosen.append(i)
            D=np.maximum(D,np.clip(f[ids]@f[i],0,1))
            covered=np.maximum(covered,np.clip(f[refs]@f[i],0,1))
            curve.append(float(np.mean(covered,dtype=np.float64)));steps.append(winner)
        records.append(dict(region_id=rid,members=ids,reference_coarse_indices=refs,prefix=chosen,
            coverage_curve=curve,density_weight=r['weight'],local_steps=steps))
    if all_members!=set(range(len(candidates))):raise ProtocolError('Candidate eligibility lost')
    # 步骤3：目标Σw_r C_r(n_r)，C为粗观测的平均最大视觉相似度；不包含答案、时长或细帧计数奖励。
    allocation,value=allocate_prefix_budget([r['coverage_curve'] for r in records],[r['density_weight'] for r in records],target)
    selected=sorted([i for r,k in zip(records,allocation) for i in r['prefix'][:k]],key=lambda i:frame_key(candidates[i]))
    parent_selected=sorted([i for r in records for i in r['prefix'][:reference_quotas[str(r['region_id'])]]],key=lambda i:frame_key(candidates[i]))
    baseline=sum(r['density_weight']*r['coverage_curve'][reference_quotas[str(r['region_id'])]] for r in records)
    if value+1e-12<baseline:raise ProtocolError('DP worse than feasible reference allocation')
    if len(selected)!=target or len(set(selected))!=target:raise ProtocolError('Final budget mismatch')
    return selected,dict(rule='exact DP over unchanged L_r*sqrt(u) local prefixes; empirical coarse-frame coverage',
        region_budgets={str(r['region_id']):k for r,k in zip(records,allocation)},regions=records,
        protected_indices=[r['anchor_index'] for r in plan['regions']],eligible_indices=sorted(all_members),
        reference_quotas=reference_quotas,reference_prefix_selection=parent_selected,
        objective=value,reference_objective=baseline,normalized_objective=value/sum(r['density_weight'] for r in records),
        limitation='Optimal only over these deterministic prefixes; visual coverage is not answer sufficiency')
