"""局部序列诊断：保护原锚点后，以原问题加权覆盖边际增益构造前缀。"""
import numpy as np
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget
from videoqa_methods.density_selection import frame_key


def select_marginal_coverage_prefix(pool,features,reference_trace):
    """接口：固定exp12参考、密度与DP，仅替换局部前缀；无模型和标准答案输入。"""
    parent_selected,parent=select_relevance_coverage(pool,features,reference_trace)
    f=np.asarray(features,dtype=np.float32);c=pool['candidates'];target=len(parent_selected);records=[]
    # 步骤1：原exp07轨迹给出不相交资格单元，粗参考和保护锚点均保持原样。
    for original,weighted,region in zip(reference_trace['regions'],parent['regions'],pool['plan']['regions']):
        assert original['region_id']==weighted['region_id']==region['region_id']
        members=original['members'];refs=weighted['reference_coarse_indices'];weights=np.asarray(weighted['reference_weights'],dtype=np.float64)
        if len(refs)*len(members)*4>512*1024**2:raise ValueError('Similarity workspace exceeds 512 MiB; pause without changing objective')
        similarity=np.clip(f[refs]@f[members].T,0,1)
        position={i:j for j,i in enumerate(members)};chosen=[region['anchor_index']]
        covered=similarity[:,position[chosen[0]]].copy();curve=[0.,float(weights@covered)];steps=[]
        # 步骤2：只奖励能提高既有粗参考最大相似度的部分，避免仅以自身差异奖励离群帧。
        while len(chosen)<min(target,len(members)):
            gains=np.empty(len(members),dtype=np.float64)
            for start in range(0,len(members),256):
                gains[start:start+256]=weights@np.maximum(similarity[:,start:start+256]-covered[:,None],0)
            proposals=[i for i in members if i not in chosen]
            winner=min(proposals,key=lambda i:(-float(gains[position[i]]),*frame_key(c[i])))
            gain=float(gains[position[winner]]);chosen.append(winner)
            covered=np.maximum(covered,similarity[:,position[winner]])
            value=min(1.,max(curve[-1],float(weights@covered)));curve.append(value)
            steps.append(dict(candidate_index=winner,weighted_coverage_gain=gain,coverage_after=value))
        records.append(dict(region_id=original['region_id'],members=members,reference_coarse_indices=refs,
            reference_weights=weights.tolist(),prefix=chosen,coverage_curve=curve,density_weight=weighted['density_weight'],local_steps=steps))
    # 步骤3：复用精确DP；最优性仅限此次构造的前缀族，视觉覆盖不等于答题证据。
    quotas,value=allocate_prefix_budget([r['coverage_curve'] for r in records],[r['density_weight'] for r in records],target)
    selected=sorted([i for r,k in zip(records,quotas) for i in r['prefix'][:k]],key=lambda i:frame_key(c[i]))
    if len(selected)!=target or len(set(selected))!=target:raise ValueError('Budget identity mismatch')
    return selected,dict(regions=records,region_budgets={str(r['region_id']):k for r,k in zip(records,quotas)},
        parent_selected_indices=parent_selected,parent_region_budgets=parent['region_budgets'],objective=value,
        rule='Original protected anchor then original-question-weighted empirical coverage marginal; same density and exact DP',
        limitation='Visual facility coverage is only a proxy; rare states and low-score relations may still be undervalued')
