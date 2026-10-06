"""单因素视觉时间联合重复度：预算覆盖和原种子支持不变。"""
import math
import numpy as np
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget
from videoqa_methods.density_selection import frame_key,validate_candidates


def joint_redundancy(features,candidate_times,selected_times,selected_features,scale_seconds=4.0):
    """接口：每个相同参照帧的cos+×时间核取max，不能错配两个独立max。

    scale=None为旧纯视觉控制；正式只使用继承的4秒，不扫描尺度或融合系数。
    """
    f=np.asarray(features,dtype=np.float32);s=np.asarray(selected_features,dtype=np.float32)
    t=np.asarray(candidate_times,dtype=np.float64);u=np.asarray(selected_times,dtype=np.float64)
    if f.ndim!=2 or s.ndim!=2 or f.shape[1]!=s.shape[1] or len(f)!=len(t) or len(s)!=len(u) or not len(s):
        raise ValueError('Invalid joint-reference population')
    if not all(np.isfinite(a).all() for a in (f,s,t,u)):
        raise ValueError('Non-finite visual/time references')
    # 步骤1：相似与时间必须来自同一参照s；不同时间可能是不同发生位置，非语义真值。
    # 与冻结实现相同，逐参照GEMV而非批GEMM，避免近重复帧末位误差经sqrt放大。
    visual=np.stack([np.clip(f@row,0,1) for row in s],axis=1)
    if scale_seconds is None:return visual.max(axis=1)
    if not math.isfinite(scale_seconds) or scale_seconds<=0:raise ValueError('Invalid frozen temporal scale')
    # 步骤2：固定Laplace时间核；界限0–1，时间不同不再仅因外观相同被置零。
    temporal=np.exp(-np.abs(t[:,None]-u[None,:])/scale_seconds)
    return (visual*temporal).max(axis=1)


def select_visual_temporal_redundancy(pool,features,parent_trace,budget=32,scale_seconds=4.0):
    """接口：原L×sqrt(1−D)只换D为联合相似，原视觉覆盖/密度/DP完全不变。"""
    c=pool['candidates'];validate_candidates(c,pool['video']);f=np.asarray(features,dtype=np.float32)
    if f.ndim!=2 or len(f)!=len(c) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5):
        raise ValueError('Original normalized visual representation changed')
    target=min(budget,len(c))
    if target!=sum(parent_trace['region_budgets'].values()):raise ValueError('Actual source budget differs')
    records=[];seen=set()
    # 步骤1：固定原池、资格单元、锚点、原题种子分数与粗参考。
    for original,region in zip(parent_trace['regions'],pool['plan']['regions']):
        members=original['members'];refs=original['reference_coarse_indices'];anchor=region['anchor_index']
        if original['region_id']!=region['region_id'] or seen&set(members) or anchor not in members:
            raise ValueError('Region qualification changed')
        seen.update(members);weights=np.asarray(original['reference_weights'],dtype=np.float64)
        support,_,_=weighted_support(f,members,region['seed_indices'],pool['initial_scores'])
        positions={i:j for j,i in enumerate(members)};times=np.array([c[i]['timestamp_seconds'] for i in members])
        chosen=[anchor];covered=np.clip(f[refs]@f[anchor],0,1)
        raw=np.clip(f[members]@f[anchor],0,1)
        D=raw.copy() if scale_seconds is None else raw*np.exp(-np.abs(times-c[anchor]['timestamp_seconds'])/scale_seconds)
        curve=[0.,min(1.,float(weights@covered))];steps=[]
        # 步骤2：增加的是时间参照，不额外计算ITM、提取特征或改变重复惩罚指数。
        while len(chosen)<min(target,len(members)):
            winner=min((i for i in members if i not in chosen),key=lambda i:(-float(support[positions[i]])*math.sqrt(1-float(D[positions[i]])),*frame_key(c[i])))
            steps.append(dict(candidate_index=winner,support=float(support[positions[winner]]),
                joint_redundancy=float(D[positions[winner]]),local_score=float(support[positions[winner]])*math.sqrt(1-float(D[positions[winner]]))))
            chosen.append(winner);covered=np.maximum(covered,np.clip(f[refs]@f[winner],0,1))
            new_raw=np.clip(f[members]@f[winner],0,1)
            contribution=new_raw if scale_seconds is None else new_raw*np.exp(-np.abs(times-c[winner]['timestamp_seconds'])/scale_seconds)
            D=np.maximum(D,contribution)
            curve.append(min(1.,max(curve[-1],float(weights@covered))))
        if chosen==original['prefix']:curve=original['coverage_curve']
        records.append(dict(region_id=region['region_id'],members=members,reference_coarse_indices=refs,
            reference_weights=weights.tolist(),prefix=chosen,coverage_curve=curve,density_weight=original['density_weight'],local_steps=steps))
    if seen!=set(range(len(c))):raise ValueError('Global candidate eligibility changed')
    # 步骤3：用原视觉目标评价新实际前缀，避免纯时间目标对静态空场重复分配预算。
    quotas,value=allocate_prefix_budget([r['coverage_curve'] for r in records],[r['density_weight'] for r in records],target)
    ids=sorted([i for r,k in zip(records,quotas) for i in r['prefix'][:k]],key=lambda i:frame_key(c[i]))
    fixed=sorted([i for r in records for i in r['prefix'][:parent_trace['region_budgets'][str(r['region_id'])]]],key=lambda i:frame_key(c[i]))
    protected=[r['anchor_index'] for r in pool['plan']['regions']]
    if len(ids)!=len(set(ids)) or len(ids)!=target or not set(protected)<=set(ids):raise ValueError('Budget/anchor changed')
    return ids,dict(regions=records,region_budgets={str(r['region_id']):k for r,k in zip(records,quotas)},
        protected_indices=protected,fixed_parent_quota_indices=fixed,objective=value,temporal_scale_seconds=scale_seconds,
        coverage_kind='original ITM-weighted pure visual coverage',support_kind='original weighted visual seed support',
        rule='D=max_s cos+(f_i,f_s)*exp(-abs(t_i-t_s)/4); original L*sqrt(1-D), visual coverage and positive-quota DP',
        limitation='Different time can be redundant; this is an occurrence-aware proxy, not verified semantic state novelty')
