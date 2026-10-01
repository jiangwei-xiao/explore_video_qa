"""固定exp15核心锚点，隔离不同原核心的重复与覆盖参照；总参考权重守恒。"""
import math
import numpy as np
from videoqa_methods.core_anchor_prefix import select_core_anchor_prefix
from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.density_selection import frame_key


def nearest_core(source_pts,intervals):
    """接口：以真实PTS到未扩窗核心种子区间的距离归属，同距取更早核心。"""
    if not intervals:raise ValueError('No original core intervals')
    # 同视频time_base为正且相同，直接比较整数PTS距离与比较有理秒距离严格等价。
    return min(intervals,key=lambda k:(max(intervals[k][0]-source_pts,0,source_pts-intervals[k][1]),intervals[k][0],k))


def select_core_partition(pool,features,reference_trace):
    """接口：同池、同exp15锚点和参考质量，仅限定核心内去重与覆盖竞争。"""
    parent_selected,parent=select_core_anchor_prefix(pool,features,reference_trace)
    f=np.asarray(features,dtype=np.float32);c=pool['candidates'];scores=pool['initial_scores'];target=len(parent_selected)
    cores={x['core_id']:x for x in pool['plan']['cores']};records=[];reference_value=0.;all_members=set()
    # 步骤1：只拆选择单元，不改最终解码窗口、补查池、种子集合或原问题分数。
    for before,region in zip(parent['regions'],pool['plan']['regions']):
        intervals={k:(min(c[i]['source_pts'] for i in cores[k]['seed_indices']),max(c[i]['source_pts'] for i in cores[k]['seed_indices'])) for k in region['core_ids']}
        groups={k:[] for k in intervals}
        for i in before['members']:groups[nearest_core(c[i]['source_pts'],intervals)].append(i)
        reference_weights=dict(zip(before['reference_coarse_indices'],before['reference_weights']))
        for core_id,ids in groups.items():
            refs=[i for i in ids if i in reference_weights];weights=np.array([reference_weights[i] for i in refs],dtype=np.float64)
            if not refs or all_members&set(ids):raise ValueError('Core partition lost or duplicated references')
            all_members.update(ids);seeds=cores[core_id]['seed_indices'];anchor=min(seeds,key=lambda i:(-scores[i],*frame_key(c[i])))
            if anchor not in ids or not set(seeds)<=set(ids):raise ValueError('Core seed ownership lost')
            # 步骤2：L仍用全部原父区域种子及原归一化，避免同时改变定位支持信号。
            support,_,_=weighted_support(f,ids,region['seed_indices'],scores);position={i:j for j,i in enumerate(ids)}
            chosen=[anchor];D=np.clip(f[ids]@f[anchor],0,1);covered=np.clip(f[refs]@f[anchor],0,1);curve=[0.,min(1.,float(weights@covered))];steps=[]
            while len(chosen)<min(target,len(ids)):
                winner=min((i for i in ids if i not in chosen),key=lambda i:(-float(support[position[i]])*math.sqrt(1-float(D[position[i]])),*frame_key(c[i])))
                steps.append(dict(candidate_index=winner,support=float(support[position[winner]]),redundancy=float(D[position[winner]])))
                chosen.append(winner);D=np.maximum(D,np.clip(f[ids]@f[winner],0,1));covered=np.maximum(covered,np.clip(f[refs]@f[winner],0,1))
                curve.append(min(1.,max(curve[-1],float(weights@covered))))
            if len(groups)==1:
                if chosen!=before['prefix']:raise ValueError('Single-core equivalence lost')
                curve=before['coverage_curve']
            old_local=[i for i in parent_selected if i in ids];assert anchor in old_local
            reference_value+=before['density_weight']*float(weights@np.max(np.clip(f[refs]@f[old_local].T,0,1),axis=1))
            records.append(dict(region_id=core_id,parent_region_id=region['region_id'],unit_type='original_core_selection_cell',
                members=ids,reference_coarse_indices=refs,reference_weights=weights.tolist(),reference_mass=float(weights.sum()),
                prefix=chosen,coverage_curve=curve,density_weight=before['density_weight'],local_steps=steps,anchor_index=anchor,
                support_seed_indices=region['seed_indices']))
        mass=sum(x['reference_mass'] for x in records if x['parent_region_id']==region['region_id'])
        if abs(mass-1)>1e-12:raise ValueError('Parent reference mass not conserved')
    # 步骤3：子核心权重不重新归一化，避免仅因拆分就增加区域奖励；曲线直接携带原参考质量。
    if all_members!=set(range(len(c))):raise ValueError('Global candidate eligibility changed')
    quotas,value=allocate_prefix_budget([r['coverage_curve'] for r in records],[r['density_weight'] for r in records],target)
    selected=sorted([i for r,k in zip(records,quotas) for i in r['prefix'][:k]],key=lambda i:frame_key(c[i]))
    if len(selected)!=len(set(selected)) or len(selected)!=target or not set(parent['protected_indices'])<=set(selected):raise ValueError('Protected budget mismatch')
    aggregated={str(r['region_id']):sum(n for unit,n in zip(records,quotas) if unit['parent_region_id']==r['region_id']) for r in pool['plan']['regions']}
    return selected,dict(regions=records,region_budgets={str(r['region_id']):n for r,n in zip(records,quotas)},aggregated_parent_budgets=aggregated,
        parent_selected_indices=parent_selected,parent_region_budgets=parent['region_budgets'],protected_indices=parent['protected_indices'],
        objective=value,parent_selection_under_restricted_objective=reference_value,
        rule='Nearest original-core interval ownership; same parent support and unrenormalized reference mass; core-local redundancy/coverage; exact prefix DP',
        limitation='Temporal ownership is not semantic relatedness; cross-core equivalents no longer share coverage; raw objective not comparable to parent objective')
