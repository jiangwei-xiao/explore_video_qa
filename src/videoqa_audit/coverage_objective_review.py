"""等权与ITM加权覆盖的只读对照；结构可达性不等于语义充分性。"""
from fractions import Fraction
import math


def frame_identity(frame):
    """接口：同一已核验视频内以源帧号和真实PTS对应，不用候选编号跨池匹配。"""
    return int(frame['source_frame_index']),int(frame['source_pts'])


def selected_indices(trace):
    """接口：从固定前缀与正配额恢复选择，检查名额和源候选唯一性。"""
    result=[]
    for r in trace['regions']:
        k=trace['region_budgets'][str(r['region_id'])]
        if not isinstance(k,int) or not 1<=k<=len(r['prefix']):raise ValueError('Invalid positive quota')
        result.extend(r['prefix'][:k])
    if len(result)!=len(set(result)):raise ValueError('Duplicate candidate across regions')
    return result


def compare_allocations(pool,left,right):
    """接口：同池、同前缀两目标输出配额/帧变化；上游不一致时拒绝归因。"""
    a,b=left['regions'],right['regions']
    if len(a)!=len(b) or not a:raise ValueError('Region count changed')
    changes=[]
    # 步骤1：逐区核验原参考、局部次序和密度权重，只有配额可不同。
    for x,y in zip(a,b):
        for key in ['region_id','prefix','reference_coarse_indices','density_weight']:
            if x[key]!=y[key]:raise ValueError('Upstream changed: '+key)
        rid=str(x['region_id']);ka,kb=left['region_budgets'][rid],right['region_budgets'][rid]
        changes.append(dict(region_id=x['region_id'],left_quota=ka,right_quota=kb,delta=kb-ka,
            members=x.get('members',[]),reference_coarse_indices=x['reference_coarse_indices'],
            prefix=x['prefix'],density_weight=x['density_weight']))
    sa,sb=set(selected_indices(left)),set(selected_indices(right));c=pool['candidates']
    if len(sa)!=len(sb):raise ValueError('Actual frame budget changed')
    seeds=set(pool['plan']['seed_indices']);n=pool['initial_count']
    # 步骤2：保留源身份与新旧来源；变化数量不能自动解释为证据损失。
    return dict(budget=len(sa),regions=changes,region_count=len(a),
        transferred_slots=sum(abs(r['delta']) for r in changes)//2,
        selected_overlap=len(sa&sb),selected_jaccard=len(sa&sb)/len(sa|sb),
        left_only=[dict(candidate_index=i,frame=c[i]) for i in sorted(sa-sb)],
        right_only=[dict(candidate_index=i,frame=c[i]) for i in sorted(sb-sa)],
        retained_seeds=[len(sa&seeds),len(sb&seeds)],selected_fine=[sum(i>=n for i in sa),sum(i>=n for i in sb)],
        selected_left=sorted(sa),selected_right=sorted(sb))


def trace_witness_path(pool,trace,witnesses):
    """接口：已列源见证→合法池/唯一归属/前缀位置/可行配额；支持粗细替代见证。"""
    lookup={frame_identity(c):i for i,c in enumerate(pool['candidates'])};selected=set(selected_indices(trace))
    budget=len(selected);rnum=len(trace['regions']);out=[]
    # 步骤1：跨池只映射源身份，重复见证去重；缺池与缺前缀分别记录。
    for key in sorted(set(witnesses)):
        if key not in lookup:
            out.append(dict(source_frame_index=key[0],source_pts=key[1],state='outside_legal_pool',selected=False));continue
        i=lookup[key]
        owners=[r for r in trace['regions'] if i in r.get('members',r['reference_coarse_indices'])]
        if len(owners)!=1:raise ValueError('Witness owner is not unique')
        r=owners[0];position=r['prefix'].index(i)+1 if i in r['prefix'] else None
        quota=trace['region_budgets'][str(r['region_id'])]
        capacity=min(len(r['prefix']),budget-rnum+1)
        chosen=i in selected
        if chosen!=(position is not None and position<=quota):raise ValueError('Prefix and selected witness disagree')
        state='selected' if chosen else 'outside_prefix' if position is None else 'beyond_feasible_quota' if position>capacity else 'quota_reachable'
        out.append(dict(candidate_index=i,source_frame_index=key[0],source_pts=key[1],region_id=r['region_id'],
            prefix_position=position,quota=quota,max_feasible_quota=capacity,selected=chosen,state=state,
            additional_slots=None if position is None else max(0,position-quota)))
    return dict(paths=out,exact_hit=any(r['selected'] for r in out),
        any_quota_reachable=any(r['state']=='quota_reachable' for r in out),
        semantic_sufficiency='not_inferred; unlisted alternatives may exist')


def region_correspondence(pool_a,trace_a,pool_b,trace_b):
    """接口：不同预算按真实范围及源身份建对应，允许多对多，不沿用region_id。"""
    if pool_a['video']['path']!=pool_b['video']['path'] or pool_a['video']['time_base']!=pool_b['video']['time_base']:
        raise ValueError('Video clock identity changed')
    tb=Fraction(pool_a['video']['time_base']);start=pool_a['video']['start_pts']
    def regions(pool,trace):
        values=[]
        for r in trace['regions']:
            keys={frame_identity(pool['candidates'][i]) for i in r['reference_coarse_indices']}
            times=[(p-start)*tb for _,p in keys]
            values.append(dict(region_id=r['region_id'],keys=keys,left=min(times),right=max(times)))
        return values
    a,b=regions(pool_a,trace_a),regions(pool_b,trace_b);edges=[]
    # 步骤1：范围重叠与共享源候选分别保存，离散采样范围不冒充语义事件边界。
    for x in a:
        for y in b:
            overlap=max(Fraction(0),min(x['right'],y['right'])-max(x['left'],y['left']))
            shared=len(x['keys']&y['keys'])
            if shared or overlap>0:
                edges.append(dict(left_region=x['region_id'],right_region=y['region_id'],
                    overlap_seconds=float(overlap),shared_coarse_sources=shared))
    for edge in edges:
        edge['left_degree']=sum(e['left_region']==edge['left_region'] for e in edges)
        edge['right_degree']=sum(e['right_region']==edge['right_region'] for e in edges)
        edge['relation']='many_to_many' if edge['left_degree']>1 and edge['right_degree']>1 else 'split' if edge['left_degree']>1 else 'merge' if edge['right_degree']>1 else 'one_to_one'
    return edges


def component_at(pool,features,region,index,before):
    """接口：只用缓存视觉特征重算一个候选的支持/差异分量，不调用编码器。"""
    import numpy as np
    seeds=next(r['seed_indices'] for r in pool['plan']['regions'] if r['region_id']==region['region_id'])
    scores=pool['initial_scores'];peak=max(scores[i] for i in seeds)
    support=max((scores[j]/peak if peak>0 else 1.)*float(np.clip(features[index]@features[j],0,1)) for j in seeds)
    redundant=max((float(np.clip(features[index]@features[j],0,1)) for j in before),default=0.)
    return dict(candidate_index=index,itm=scores[index] if index<pool['initial_count'] else None,
        support=support,redundancy=redundant,diversity=1-redundant,local_score=support*math.sqrt(1-redundant))
