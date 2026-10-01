"""只替换种子支持表征；外观去重与原问题加权覆盖预算保持原定义。"""
import math
import numpy as np
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.density_selection import frame_key


def select_projected_support(pool,features,projected_features,reference_trace):
    """接口：L用BLIP官方ITC投影，D和覆盖用原CLS；返回唯一帧、前缀及预算轨迹。"""
    parent_selected,parent=select_relevance_coverage(pool,features,reference_trace)
    f=np.asarray(features,dtype=np.float32);g=np.asarray(projected_features,dtype=np.float32);c=pool['candidates'];target=len(parent_selected)
    if g.ndim!=2 or len(g)!=len(f) or not np.isfinite(g).all() or not np.allclose(np.linalg.norm(g,axis=1),1,atol=1e-5,rtol=0):raise ValueError('Invalid projected features')
    records=[]
    # 步骤1：原最终区域、全局粗资格、锚点及原问题ITM种子权重不变。
    for original,weighted,region in zip(reference_trace['regions'],parent['regions'],pool['plan']['regions']):
        assert original['region_id']==weighted['region_id']==region['region_id']
        ids=original['members'];refs=weighted['reference_coarse_indices'];weights=np.asarray(weighted['reference_weights'],dtype=np.float64)
        support,_,_=weighted_support(g,ids,region['seed_indices'],pool['initial_scores']);position={i:j for j,i in enumerate(ids)}
        chosen=[region['anchor_index']];D=np.clip(f[ids]@f[chosen[0]],0,1);covered=np.clip(f[refs]@f[chosen[0]],0,1);curve=[0.,min(1.,float(weights@covered))];steps=[]
        # 步骤2：只更换L的帧间表征，最大重复度仍对原CLS计算；不新增混合系数或支持下限。
        while len(chosen)<min(target,len(ids)):
            winner=min((i for i in ids if i not in chosen),key=lambda i:(-float(support[position[i]])*math.sqrt(1-float(D[position[i]])),*frame_key(c[i])))
            steps.append(dict(candidate_index=winner,projected_support=float(support[position[winner]]),raw_redundancy=float(D[position[winner]]),
                local_score=float(support[position[winner]])*math.sqrt(1-float(D[position[winner]]))))
            chosen.append(winner);D=np.maximum(D,np.clip(f[ids]@f[winner],0,1));covered=np.maximum(covered,np.clip(f[refs]@f[winner],0,1))
            curve.append(min(1.,max(curve[-1],float(weights@covered))))
        if chosen==weighted['prefix']:curve=weighted['coverage_curve']
        records.append(dict(region_id=region['region_id'],members=ids,reference_coarse_indices=refs,reference_weights=weights.tolist(),
            prefix=chosen,coverage_curve=curve,density_weight=weighted['density_weight'],local_steps=steps))
    # 步骤3：同一个预算目标评价新的实际前缀；配额可变是下游结果，不是改变预算公式。
    quotas,value=allocate_prefix_budget([r['coverage_curve'] for r in records],[r['density_weight'] for r in records],target)
    selected=sorted([i for r,k in zip(records,quotas) for i in r['prefix'][:k]],key=lambda i:frame_key(c[i]))
    anchors={r['anchor_index'] for r in pool['plan']['regions']}
    if len(selected)!=target or len(selected)!=len(set(selected)) or not anchors<=set(selected):raise ValueError('Protected budget mismatch')
    return selected,dict(regions=records,region_budgets={str(r['region_id']):k for r,k in zip(records,quotas)},objective=value,
        parent_selected_indices=parent_selected,parent_region_budgets=parent['region_budgets'],
        support_kind='recovered official BLIP image-text contrastive vision projection',redundancy_kind='original pure visual CLS',coverage_kind='original pure visual CLS, original-question weighted coarse references',
        rule='Only support representation changed; same raw-CLS novelty, exp12 reference weights, density and exact prefix DP',
        limitation='Caption-aligned visual association is not a temporal relation or answer-value estimator')
