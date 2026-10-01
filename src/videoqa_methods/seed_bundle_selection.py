"""多种子候选帧组：视觉覆盖与成对状态完成联合贪心，独立探索，不改exp07。"""
import math
from fractions import Fraction
import numpy as np
from videoqa_methods.density_selection import frame_key,source_time,validate_candidates


def propose_seed_bundles(pool,features,radius=Fraction(4)):
    """接口：每个原种子在±4秒内各侧提出一个相关且不同的状态，输出去重组和边权。"""
    c=pool['candidates'];v=pool['video'];f=np.asarray(features,dtype=np.float32)
    seeds=pool['plan']['seed_indices'];scores=pool['initial_scores']
    if not seeds or len(scores)!=pool['initial_count'] or any(not math.isfinite(s) or not 0<=s<=1 for s in scores) or any(i>=len(scores) or i<0 for i in seeds):
        raise ValueError('Invalid seed scores')
    peak=max(scores[i] for i in seeds)
    times=[source_time(row,v) for row in c];edges={};groups=set();details=[]
    # 步骤1：全部原种子都有提议资格，不以区域最高分锚点代替其他局部入口。
    for seed in seeds:
        partners=[];entries=[]
        for side in ('before','after'):
            ids=[i for i,t in enumerate(times) if (0<times[seed]-t<=radius if side=='before' else 0<t-times[seed]<=radius)]
            if not ids:continue
            similarities=np.clip(f[ids]@f[seed],0,1)
            quality=similarities*np.sqrt(np.maximum(0,1-similarities))
            # 单种子支持乘sqrt差异，与现有直觉一致；不是事件/答案价值概率。
            j=min(range(len(ids)),key=lambda j:(-float(quality[j]),*frame_key(c[ids[j]])))
            i=ids[j];weight=(scores[seed]/peak if peak>0 else 1.)*float(quality[j])
            if weight<=0:continue
            pair=tuple(sorted((seed,i),key=lambda x:frame_key(c[x])))
            edges[pair]=edges.get(pair,0.)+weight;groups.add(pair);partners.append(i)
            entries.append(dict(side=side,index=i,similarity=float(similarities[j]),quality=float(quality[j]),weight=weight))
        if len(partners)==2:groups.add(tuple(sorted((seed,*partners),key=lambda x:frame_key(c[x]))))
        details.append(dict(seed_index=seed,partners=entries))
    # 步骤2：重叠帧不重复计成本；互向提议同一对时合并权重，不重复建立状态。
    total=sum(edges.values())
    normalized=[dict(indices=list(pair),weight=w/total) for pair,w in sorted(edges.items())] if total else []
    return sorted(groups),normalized,details


def select_seed_bundles(pool,features,reference_trace,budget=32,with_pairs=True):
    """接口：固定旧池与保护锚点，单帧和最多三帧组共同竞争；输出实际配额和完整增益轨迹。"""
    c=pool['candidates'];v=pool['video'];validate_candidates(c,v);f=np.asarray(features,dtype=np.float32)
    if f.ndim!=2 or len(f)!=len(c) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5,rtol=0):
        raise ValueError('Invalid normalized features')
    if not isinstance(budget,int) or budget<1:raise ValueError('Invalid budget')
    target=min(budget,len(c));protected=[r['anchor_index'] for r in pool['plan']['regions']]
    if len(set(protected))!=len(protected) or len(protected)>target:raise ValueError('Invalid protected capacity')
    groups,edges,proposal_trace=propose_seed_bundles(pool,f);owners={};matrices=[]
    total_weight=sum(r['density_weight'] for r in reference_trace['regions'])
    # 步骤1：逐区构建原粗帧覆盖矩阵；与exp07相同的参照和密度权重，不固定旧配额/前缀。
    for r in reference_trace['regions']:
        ids=r['members'];refs=r['reference_coarse_indices']
        if not refs or not ids:raise ValueError('Empty region')
        for j,i in enumerate(ids):
            if i in owners:raise ValueError('Overlapping candidate cells')
            owners[i]=(len(matrices),j)
        sim=np.clip(f[refs]@f[ids].T,0,1)
        matrices.append(dict(region_id=r['region_id'],ids=ids,sim=sim,covered=np.zeros(len(refs),dtype=np.float32),weight=r['density_weight']/total_weight))
    if set(owners)!=set(range(len(c))):raise ValueError('Incomplete ownership')
    selected=set(protected)
    for i in protected:
        k,j=owners[i];m=matrices[k];m['covered']=np.maximum(m['covered'],m['sim'][:,j])
    initial_coverage=sum(m['weight']*float(m['covered'].mean(dtype=np.float64)) for m in matrices)
    def pair_value(s):
        """累计两端均已选的状态对权重；重叠源帧不重复计成本。"""
        return sum(e['weight'] for e in edges if set(e['indices'])<=s)
    initial_pair=pair_value(selected);coverage_scale=max(0.,1-initial_coverage);pair_scale=max(0.,1-initial_pair)
    steps=[]
    # 步骤2：两个代理按相对保护集合剩余可提升范围归一化，固定等权，不搜索系数。
    # 该选择是实验假设，不保证答题最优；成对完成有互补性，不套用子模贪心最优保证。
    while len(selected)<target:
        remaining=target-len(selected);pair_before=pair_value(selected)
        singles=np.zeros(len(c),dtype=np.float64)
        for m in matrices:
            delta=np.maximum(m['sim'],m['covered'][:,None])-m['covered'][:,None]
            singles[m['ids']]=m['weight']*delta.mean(axis=0,dtype=np.float64)
        pair_singles=np.zeros(len(c),dtype=np.float64)
        if with_pairs and pair_scale>1e-12:
            for e in edges:
                missing=set(e['indices'])-selected
                if len(missing)==1:pair_singles[next(iter(missing))]+=e['weight']
        candidates=[]
        for i in range(len(c)):
            if i in selected:continue
            gain=(singles[i]/coverage_scale if coverage_scale>1e-12 else 0.)
            gain+=(pair_singles[i]/pair_scale if with_pairs and pair_scale>1e-12 else 0.)
            candidates.append(dict(added=[i],coverage_gain=float(singles[i]),pair_gain=float(pair_singles[i]),gain=float(gain),cost=1))
        # 纯覆盖控制中帧组按单位成本被某个单帧弱支配，直接省略，避免浮点假破平局。
        for group in groups if with_pairs else []:
            added=[i for i in group if i not in selected]
            if not 1<len(added)<=remaining:continue
            per_region={}
            for i in added:
                k,j=owners[i];per_region.setdefault(k,[]).append(j)
            cg=0.
            for k,js in per_region.items():
                m=matrices[k];cg+=m['weight']*float((np.maximum(m['covered'],m['sim'][:,js].max(axis=1))-m['covered']).mean(dtype=np.float64))
            pg=pair_value(selected|set(added))-pair_before if with_pairs else 0.
            gain=(cg/coverage_scale if coverage_scale>1e-12 else 0.)+(pg/pair_scale if with_pairs and pair_scale>1e-12 else 0.)
            candidates.append(dict(added=added,coverage_gain=cg,pair_gain=pg,gain=gain,cost=len(added)))
        # 同收益密度优先更少帧；原PTS/编号为最后稳定排序，禁止暗中偏爱帧组。
        winner=min(candidates,key=lambda x:(-x['gain']/x['cost'],x['cost'],tuple(frame_key(c[i]) for i in x['added'])))
        before=len(selected);selected.update(winner['added'])
        for i in winner['added']:
            k,j=owners[i];m=matrices[k];m['covered']=np.maximum(m['covered'],m['sim'][:,j])
        steps.append(dict(step=len(steps),selected_count_before=before,**winner))
    # 步骤3：恢复源时间顺序并核对唯一预算；新旧候选来源不赋予额外优先级。
    ids=sorted(selected,key=lambda i:frame_key(c[i]));quotas={str(m['region_id']):sum(i in selected for i in m['ids']) for m in matrices}
    assert len(ids)==target and set(protected)<=selected
    final_coverage=sum(m['weight']*float(m['covered'].mean(dtype=np.float64)) for m in matrices)
    return ids,dict(with_pairs=with_pairs,region_budgets=quotas,protected_indices=protected,steps=steps,
        proposal_groups=[list(g) for g in groups],edges=edges,proposal_trace=proposal_trace,
        initial_coverage=initial_coverage,final_coverage=final_coverage,initial_pair=initial_pair,final_pair=pair_value(selected),
        multi_frame_steps=sum(s['cost']>1 for s in steps),rule='normalized coverage plus pair completion, marginal gain per new unique frame',
        limitation='Visual/time proxy; greedy approximate search, not semantic evidence guarantee or exact optimizer')
