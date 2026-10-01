"""预算目标单因素：固定exp07局部前缀，以原问题ITM相对权重评价粗帧覆盖。"""
import numpy as np
from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget
from videoqa_methods.density_selection import frame_key,validate_candidates


def select_relevance_coverage(pool,features,reference_trace):
    """接口：原池/前缀完全固定，仅重算加权覆盖曲线与DP配额；不增加评分或模型。"""
    c=pool['candidates'];validate_candidates(c,pool['video']);f=np.asarray(features,dtype=np.float32)
    scores=np.asarray(pool['initial_scores'],dtype=np.float64);n=pool['initial_count']
    if len(scores)!=n or not np.isfinite(scores).all() or np.any(scores<0) or np.any(scores>1):raise ValueError('Invalid coarse scores')
    if f.ndim!=2 or len(f)!=len(c) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5,rtol=0):raise ValueError('Invalid features')
    target=sum(reference_trace['region_budgets'].values());records=[];old_value=0.
    # 步骤1：固定原粗参考和原局部前缀；仅把均匀参考权重改为区内原问题分数归一化。
    for region in reference_trace['regions']:
        refs=region['reference_coarse_indices'];prefix=region['prefix'];raw=scores[refs]
        if not refs or not prefix:raise ValueError('Empty reference or prefix')
        uniform=bool(np.all(raw==raw[0]));mass=float(raw.sum())
        weights=np.full(len(refs),1/len(refs),dtype=np.float64) if mass==0 else raw/mass
        covered=np.zeros(len(refs),dtype=np.float32);curve=[0.]
        # 步骤2：同时独立重算原均匀曲线，防止把错位特征误判为预算改进。
        for pos,i in enumerate(prefix,1):
            covered=np.maximum(covered,np.clip(f[refs]@f[i],0,1))
            original=float(covered.mean(dtype=np.float64))
            if abs(original-region['coverage_curve'][pos])>1e-6:raise ValueError('Frozen coverage mismatch')
            value=region['coverage_curve'][pos] if uniform else min(1.,max(curve[-1],float(weights@covered)))
            curve.append(value)
        rid=str(region['region_id']);old_value+=region['density_weight']*curve[reference_trace['region_budgets'][rid]]
        records.append(dict(region_id=region['region_id'],prefix=prefix,reference_coarse_indices=refs,
            reference_weights=weights.tolist(),uniform_fallback=mass==0,exact_uniform_equivalence=uniform,
            coverage_curve=curve,density_weight=region['density_weight']))
    # 步骤3：同一前缀族与总预算下精确DP；不把相对ITM权重称为证据可信概率。
    quotas,value=allocate_prefix_budget([r['coverage_curve'] for r in records],[r['density_weight'] for r in records],target)
    selected=sorted([i for r,k in zip(records,quotas) for i in r['prefix'][:k]],key=lambda i:frame_key(c[i]))
    if len(selected)!=len(set(selected)) or len(selected)!=target:raise ValueError('Budget identity mismatch')
    if value+1e-12<old_value:raise ValueError('DP worse than feasible old allocation')
    return selected,dict(regions=records,region_budgets={str(r['region_id']):k for r,k in zip(records,quotas)},
        objective=value,old_allocation_weighted_objective=old_value,local_prefixes_unchanged=True,
        rule='original-question ITM normalized within each coarse reference cell; same exp07 prefixes and density weights',
        limitation='Relative retrieval preference, not calibrated evidence probability; low-score answer clues may be undervalued')
