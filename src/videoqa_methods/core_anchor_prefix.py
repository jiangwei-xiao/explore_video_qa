"""原检索核心锚点保护诊断：合并窗口不自动消除各原核心的定位代表。"""
import math
import numpy as np
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.density_selection import frame_key


def allocate_with_minima(curves,weights,minima,budget):
    """接口：以变量平移复用原DP，精确满足每区域核心锚点数下界。"""
    if len(curves)!=len(minima) or any(not 1<=m<len(c) for m,c in zip(minima,curves)):
        raise ValueError('Invalid minimum prefix lengths')
    # 步骤1：n_r=m_r−1+x_r，x_r至少1；原目标不减去常数，保留原DP同分次序。
    shifted=[[0.]+list(c[m:]) for c,m in zip(curves,minima)]
    quotas,value=allocate_prefix_budget(shifted,weights,budget-sum(m-1 for m in minima))
    return [n+m-1 for n,m in zip(quotas,minima)],value


def select_core_anchor_prefix(pool,features,reference_trace):
    """接口：只提升保护粒度到原核心，仍用原局部支持×差异和exp12预算目标。"""
    parent_selected,parent=select_relevance_coverage(pool,features,reference_trace)
    f=np.asarray(features,dtype=np.float32);c=pool['candidates'];scores=pool['initial_scores'];target=len(parent_selected)
    cores={x['core_id']:x for x in pool['plan']['cores']};records=[];minima=[];all_protected=[]
    # 步骤1：每个原核心取已有最高ITM种子，不把几何核心解释为语义事件。
    for original,weighted,region in zip(reference_trace['regions'],parent['regions'],pool['plan']['regions']):
        assert original['region_id']==weighted['region_id']==region['region_id']
        anchors=[min(cores[k]['seed_indices'],key=lambda i:(-scores[i],*frame_key(c[i]))) for k in region['core_ids']]
        anchors=sorted(set(anchors),key=lambda i:(-scores[i],*frame_key(c[i])))
        if not anchors or anchors[0]!=region['anchor_index']:raise ValueError('Original region anchor lost')
        all_protected.extend(anchors);minima.append(len(anchors));members=original['members'];refs=weighted['reference_coarse_indices']
        weights=np.asarray(weighted['reference_weights'],dtype=np.float64);support,_,_=weighted_support(f,members,region['seed_indices'],scores)
        position={i:j for j,i in enumerate(members)};chosen=[];D=np.zeros(len(members),dtype=np.float32);covered=np.zeros(len(refs),dtype=np.float32);curve=[0.];steps=[]
        # 步骤2：先固定所有核心锚点，后续候选对全部固定帧计算重复度，不能直接拼接旧尾部。
        while len(chosen)<min(target,len(members)):
            if len(chosen)<len(anchors):i=anchors[len(chosen)];protected=True
            else:
                i=min((j for j in members if j not in chosen),key=lambda j:(-float(support[position[j]])*math.sqrt(1-float(D[position[j]])),*frame_key(c[j])))
                protected=False
            steps.append(dict(candidate_index=i,protected_core_anchor=protected,support=float(support[position[i]]),redundancy=float(D[position[i]])))
            chosen.append(i);D=np.maximum(D,np.clip(f[members]@f[i],0,1));covered=np.maximum(covered,np.clip(f[refs]@f[i],0,1))
            curve.append(min(1.,max(curve[-1],float(weights@covered))))
        # 单核心应严格退化到父版本局部序列；复用其等分数数值边界以保证逐位一致。
        if len(anchors)==1:
            if chosen!=weighted['prefix']:raise ValueError('Single-core local equivalence lost')
            curve=weighted['coverage_curve']
        records.append(dict(region_id=region['region_id'],members=members,reference_coarse_indices=refs,reference_weights=weights.tolist(),
            prefix=chosen,coverage_curve=curve,density_weight=weighted['density_weight'],protected_core_anchors=anchors,local_steps=steps))
    if len(set(all_protected))!=len(all_protected) or len(all_protected)>target:raise ValueError('Core anchors exceed budget')
    # 步骤3：只给保护帧设必要下界，剩余预算由同一加权覆盖DP竞争，不均分区域名额。
    quotas,value=allocate_with_minima([r['coverage_curve'] for r in records],[r['density_weight'] for r in records],minima,target)
    selected=sorted([i for r,k in zip(records,quotas) for i in r['prefix'][:k]],key=lambda i:frame_key(c[i]))
    if len(selected)!=target or len(set(selected))!=target or not set(all_protected)<=set(selected):raise ValueError('Protected budget mismatch')
    return selected,dict(regions=records,region_budgets={str(r['region_id']):k for r,k in zip(records,quotas)},protected_indices=all_protected,
        parent_selected_indices=parent_selected,parent_region_budgets=parent['region_budgets'],objective=value,minimum_quotas=minima,
        rule='One highest original-question ITM seed per original geometric core; same local support*sqrt(novelty) and exp12 weighted coverage DP',
        limitation='Core peaks can be redundant or irrelevant; mandatory anchors may displace useful context; not semantic event coverage')
