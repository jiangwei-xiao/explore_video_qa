"""原问题窗口内的候选描述覆盖前缀；保持原锚点与exp07预算目标。"""
import math
from fractions import Fraction
import numpy as np
from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_methods.density_selection import validate_candidates,source_time,frame_key
from videoqa_methods.coarse_eligibility import coarse_selection_cells
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget
from videoqa_methods.option_seed_entry import select_option_pool


def strict_rank_counts(scores):
    """接口：每行返回严格低于当前分数的候选数；同分同值，常量查询不产生增益。"""
    a=np.asarray(scores,dtype=np.float64)
    if a.ndim!=2 or not a.shape[0] or not a.shape[1] or not np.isfinite(a).all():
        raise ValueError('Invalid local score matrix')
    result=np.zeros(a.shape,dtype=np.int64)
    # 步骤1：用经验序数比较同一查询，不把跨查询概率当作已校准置信度。
    for k,values in enumerate(a):
        _,inverse,counts=np.unique(values,return_inverse=True,return_counts=True)
        result[k]=(np.cumsum(counts)-counts)[inverse]
    return result


def select_local_query_coverage(pool,features,cfg,reference_quotas,option_scores,enabled=True):
    """接口：保留问题锚点，至多四次正增益描述覆盖，再接原L×sqrt(u)前缀及DP。

    窗内F(S)=Σq max(i∈S) rank_q(i)，以最多四个描述各自最优帧即可达到代理上限，
    因而采用四步资源上限，不声称贪心必达该上限或代理等于真实证据充分性。
    """
    if not enabled:
        selected,trace=select_option_pool(pool,features,cfg)
        return selected,dict(trace,local_query_coverage_enabled=False)
    candidates=pool['candidates']
    video=pool['video']
    plan=pool['plan']
    n=pool['initial_count']
    validate_candidates(candidates,video)
    f=np.asarray(features,dtype=np.float32)
    scores=np.asarray(option_scores,dtype=np.float64)
    if f.ndim!=2 or len(f)!=len(candidates) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5,rtol=0):
        raise ProtocolError('Invalid feature identity')
    if scores.shape!=(4,len(candidates)) or not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():
        raise ProtocolError('Expected four complete option-only score vectors')
    if n!=len(pool['initial_scores']):raise ProtocolError('Coarse prefix mismatch')
    cells=coarse_selection_cells(plan,video)
    target=min(cfg['frame_budget'],len(candidates))
    records=[]
    all_members=set()
    # 步骤1：原窗口限定描述检索；原全局粗资格、细帧资格和保护锚点保持不变。
    for pos,r in enumerate(cells['regions']):
        original=plan['regions'][pos]
        ids=sorted([i for i,c in enumerate(candidates) if Fraction(r['left_fraction'])<=source_time(c,video)<=Fraction(r['right_fraction'])],key=lambda i:frame_key(candidates[i]))
        refs=[i for i in ids if i<n]
        local_positions=[j for j,i in enumerate(ids) if Fraction(original['left_fraction'])<=source_time(candidates[i],video)<=Fraction(original['right_fraction'])]
        anchor=r['anchor_index']
        if all_members.intersection(ids) or not refs or anchor not in ids or ids.index(anchor) not in local_positions:
            raise ProtocolError('Region identity mismatch')
        if not set(r['seed_indices'])<=set(ids):raise ProtocolError('Seed outside cell')
        if any(i>=n and j not in local_positions for j,i in enumerate(ids)):raise ProtocolError('Fine frame outside original window')
        all_members.update(ids)
        local_ids=[ids[j] for j in local_positions]
        ranks=np.zeros((4,len(ids)),dtype=np.int64)
        ranks[:,local_positions]=strict_rank_counts(scores[:,local_ids])
        chosen=[anchor]
        support,_,_=weighted_support(f,ids,r['seed_indices'],pool['initial_scores'])
        D=np.clip(f[ids]@f[anchor],0,1)
        covered=np.clip(f[refs]@f[anchor],0,1)
        curve=[0.,float(np.mean(covered,dtype=np.float64))]
        best=ranks[:,ids.index(anchor)].copy()
        query_steps=0
        query_active=True
        steps=[]
        # 步骤2：四描述等权、同分同值；正边际增益优先，视觉分仅用于同增益排序。
        while len(chosen)<min(target,len(ids)):
            proposals=[]
            for j,i in enumerate(ids):
                if i in chosen:continue
                gain=int(np.sum(np.maximum(best,ranks[:,j])-best))
                visual=float(support[j])*math.sqrt(1-float(D[j]))
                proposals.append(dict(candidate_index=i,position=j,query_gain=gain,support=float(support[j]),redundancy=float(D[j]),local_score=visual))
            if query_active and query_steps<4 and max(p['query_gain'] for p in proposals)>0:
                winner=min(proposals,key=lambda p:(-p['query_gain'],-p['local_score'],*frame_key(candidates[p['candidate_index']])))
                query_steps+=1
                mode='description_coverage'
            else:
                query_active=False
                winner=min(proposals,key=lambda p:(-p['local_score'],*frame_key(candidates[p['candidate_index']])))
                mode='original_visual_rule'
            i=winner['candidate_index']
            chosen.append(i)
            best=np.maximum(best,ranks[:,winner['position']])
            D=np.maximum(D,np.clip(f[ids]@f[i],0,1))
            covered=np.maximum(covered,np.clip(f[refs]@f[i],0,1))
            curve.append(float(np.mean(covered,dtype=np.float64)))
            steps.append(dict(winner,mode=mode,query_coverage_count=int(best.sum())))
        records.append(dict(region_id=r['region_id'],members=ids,reference_coarse_indices=refs,
            original_window_indices=local_ids,prefix=chosen,coverage_curve=curve,density_weight=r['weight'],
            local_steps=steps,query_steps=query_steps))
    # 步骤3：原经验粗帧覆盖DP决定配额；未把描述代理与视觉目标混成一个加权公式。
    if all_members!=set(range(len(candidates))):raise ProtocolError('Candidate eligibility changed')
    allocation,value=allocate_prefix_budget([r['coverage_curve'] for r in records],[r['density_weight'] for r in records],target)
    selected=sorted([i for r,k in zip(records,allocation) for i in r['prefix'][:k]],key=lambda i:frame_key(candidates[i]))
    baseline=sum(r['density_weight']*r['coverage_curve'][reference_quotas[str(r['region_id'])]] for r in records)
    if len(selected)!=len(set(selected)) or len(selected)!=target or value+1e-12<baseline:
        raise ProtocolError('Budget or DP invariant failed')
    return selected,dict(local_query_coverage_enabled=True,regions=records,
        region_budgets={str(r['region_id']):k for r,k in zip(records,allocation)},
        protected_indices=[r['anchor_index'] for r in plan['regions']],eligible_indices=sorted(all_members),
        objective=value,reference_objective=baseline,
        limitation='Ordinal candidate-description coverage is not answer confidence; no final QA performed by selector')
