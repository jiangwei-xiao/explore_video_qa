"""离线分解原方法每步增益；只读历史分数/特征，不生成答案或新的模型特征。"""
from bisect import bisect_right
from collections import Counter
from fractions import Fraction
from pathlib import Path
import numpy as np
from videoqa_runtime.common import read_json,write_json,sha256

LANDMARKS={
 '264-2':[0,18,19,56,57], '004-3':[82,84,89,95], '012-3':[10,15],
 '383-2':[944,945,946], '538-1':[10,23,63], '260-2':[10,22,59,60,61],
 '170-2':[0,9,10,17,18], '158-3':[3,66,68,69], '395-2':[129,270,276,414,424],
 '306-2':[83,88,89], '103-1':[2,5,6,7,27,59], '705-1':[555,2379,2383],
 '544-1':[36,41,44,52,60]}


def groups_for(candidates,segments):
    """按冻结PTS边界归段；本函数不改变分段。"""
    boundaries=[s['start_pts'] for s in segments[1:]]
    return np.array([bisect_right(boundaries,r['source_pts']) for r in candidates])


def parts(scores,representatives,groups,counts,duplicate,lam,penalty):
    """返回原公式的三个加法项及总增益；保留符号，方便验证归因之和。"""
    relevance=(1-lam)*scores
    coverage=lam*representatives[groups]/(1+counts[groups])
    redundancy=-penalty*duplicate
    return relevance,coverage,redundancy,relevance+coverage+redundancy


def describe(i,pool,terms):
    """单帧记录明确是代理增益，不把相关性分数解释为答对概率。"""
    row=pool['candidates'][i]
    return dict(candidate_index=int(i),source_pts=row['source_pts'],seconds=row['timestamp_seconds'],
                score=float(pool['scores'][i]),relevance_term=float(terms[0][i]),
                coverage_term=float(terms[1][i]),redundancy_term=float(terms[2][i]),gain=float(terms[3][i]))


def margin(winner,target,pool,terms):
    """计算同一历史状态下胜者减目标的分项差；这是选择规则解释，不是问答因果结论。"""
    delta={name:float(term[winner]-term[target]) for name,term in zip(('relevance','coverage','redundancy'),terms[:3])}
    gap=float(terms[3][winner]-terms[3][target])
    if abs(sum(delta.values())-gap)>1e-8:raise ValueError('增益分项不闭合')
    return dict(winner=describe(winner,pool,terms),target=describe(target,pool,terms),
                winner_minus_target=delta,gap=gap,
                largest_positive_term=max(delta,key=delta.get) if max(delta.values())>1e-10 else 'tie_order',
                interpretation='固定该步已选前缀的局部数值比较，不证明替换后整轮选择或回答会怎样')


def coarse_audit(record,features,targets,penalty):
    """重放粗选状态，逐步核对原胜者增益，并记录目标被本段提名/全球竞争淘汰的位置。"""
    pool=record['pool'];n=pool['initial_count'];scores=np.asarray(pool['scores'][:n],dtype=np.float64)
    groups=groups_for(pool['candidates'][:n],pool['segments']);reps=np.array([s['representative_score'] for s in pool['segments']])
    counts=np.zeros(len(reps),dtype=np.int64);duplicate=np.zeros(n,dtype=np.float32);chosen=set()
    target_ids={p:i for i,r in enumerate(pool['candidates'][:n]) if (p:=r['source_pts']) in targets}
    details={str(p):dict(candidate_index=i,segment_id=int(groups[i]),events=[]) for p,i in target_ids.items()}
    wins=[];maximum_error=0.
    # 步骤1：每一步只使用此前真实选择，避免从最终集合倒推重复惩罚。
    for step,saved in enumerate(pool['competition_trace']):
        terms=parts(scores,reps,groups,counts,duplicate,pool['lambda_value'],penalty)
        winner=saved['candidate_index'];error=abs(float(terms[3][winner])-saved['gain']);maximum_error=max(maximum_error,error)
        if error>1e-6:raise ValueError('粗选重放增益与原日志不一致')
        remaining=[i for i in range(n) if i not in chosen]
        expected=min(remaining,key=lambda i:(-terms[3][i],pool['candidates'][i]['source_pts'],i))
        if expected!=winner:raise ValueError('粗选重放胜者不一致')
        nominees={p['segment_id']:p['candidate_index'] for p in saved['proposals']}
        wins.append(dict(step=step+1,segment_id=int(groups[winner]),segment_count_before=int(counts[groups[winner]]),**describe(winner,pool,terms)))
        # 步骤2：记录目标在本段先输掉，还是已获提名但输给其他片段。
        for pts,i in target_ids.items():
            if i in chosen:continue
            nominee=nominees[int(groups[i])]
            event=dict(step=step+1,selected=i==winner,was_segment_nominee=nominee==i,
                       same_segment_as_winner=bool(groups[i]==groups[winner]),segment_count_before=int(counts[groups[i]]),
                       global_margin=margin(winner,i,pool,terms))
            if nominee!=i:event['local_margin']=margin(nominee,i,pool,terms)
            details[str(pts)]['events'].append(event)
        chosen.add(winner);sid=int(groups[winner]);counts[sid]+=1
        local=np.flatnonzero(groups==sid)
        duplicate[local]=np.maximum(duplicate[local],np.clip(features[local]@features[winner],0,1))
    # 步骤3：配额与最终状态分别报告，不把零配额和段内失选混为一谈。
    for pts,i in target_ids.items():
        d=details[str(pts)];d['coarse_selected']=i in chosen;d['final_quota']=int(counts[groups[i]])
        if i in chosen:d['stage']='coarse_selected'
        elif counts[groups[i]]==0:d['stage']='segment_received_zero_quota'
        elif any(e['was_segment_nominee'] for e in d['events']):d['stage']='nominated_but_not_taken_within_budget'
        else:d['stage']='never_best_in_own_segment'
        local_winners=[e for e in d['events'] if e['same_segment_as_winner'] and not e['selected']]
        d['last_same_segment_loss']=local_winners[-1]['global_margin'] if local_winners else None
    return dict(winners=wins,targets=details,maximum_gain_error=maximum_error)


def refinement_audit(record,features,targets,penalty):
    """重放C已冻结配额的整段重选，跟踪窗口外目标如何因新前缀失选。"""
    pool=record['pool'];scores=np.asarray(pool['scores'],dtype=np.float64)
    groups=groups_for(pool['candidates'],pool['segments']);chosen_by_segment={};details={};maximum_error=0.
    positions={r['source_pts']:i for i,r in enumerate(pool['candidates'])}
    # 步骤1：每个收到新候选的片段按原实现从空集合开始，不沿用粗选重复度。
    for saved in pool['reselection_trace']:
        sid=saved['segment_id'];winner=saved['candidate_index'];prefix=chosen_by_segment.setdefault(sid,[])
        dup=np.zeros(len(scores)) if not prefix else np.clip(features@features[prefix].T,0,1).max(axis=1)
        coverage=np.full(len(scores),pool['lambda_value']*pool['segments'][sid]['representative_score']/(1+len(prefix)))
        rel=(1-pool['lambda_value'])*scores;red=-penalty*dup;terms=rel,coverage,red,rel+coverage+red
        error=abs(float(terms[3][winner])-saved['gain']);maximum_error=max(maximum_error,error)
        if error>1e-6:raise ValueError('重选重放增益不一致')
        available=[i for i in range(len(scores)) if groups[i]==sid and i not in prefix]
        expected=min(available,key=lambda i:(-terms[3][i],pool['candidates'][i]['source_pts'],i))
        if expected!=winner:raise ValueError('重选重放胜者不一致')
        for pts in targets:
            if pts not in positions:continue
            i=positions[pts]
            if groups[i]!=sid or i in prefix:continue
            d=details.setdefault(str(pts),dict(segment_id=sid,events=[]))
            d['events'].append(dict(segment_pick=len(prefix)+1,selected=i==winner,margin=margin(winner,i,pool,terms)))
        prefix.append(winner)
    # 步骤2：最终配额不变仍可损失窗口外帧；具体时间距离使用精确有理数PTS。
    windows=[(Fraction(h['left_fraction']),Fraction(h['right_fraction'])) for h in pool['hotspot_trace'] if h['selected']]
    tb=Fraction(pool['video']['time_base']);origin=pool['video']['start_pts']
    for pts,d in details.items():
        i=positions[int(pts)];t=(int(pts)-origin)*tb
        d['final_selected']=i in pool['selected_indices'];d['coarse_selected']=i in pool['coarse_selected']
        d['distance_to_hotspot_seconds']=min((float(max(l-t,t-r,Fraction(0))) for l,r in windows),default=None)
    return dict(targets=details,maximum_gain_error=maximum_error)


def divergence(a,b,features,penalty):
    """在A/B首次分歧前的共同状态比较两候选，隔离λ变化的即时数值作用。"""
    pa,pb=a['pool'],b['pool'];ta,tb=pa['competition_trace'],pb['competition_trace']
    step=next((s for s,(x,y) in enumerate(zip(ta,tb)) if x['candidate_index']!=y['candidate_index']),None)
    if step is None:return None
    groups=groups_for(pa['candidates'],pa['segments']);scores=np.asarray(pa['scores'],dtype=np.float64)
    reps=np.array([s['representative_score'] for s in pa['segments']]);counts=np.zeros(len(reps),dtype=np.int64);dup=np.zeros(len(scores),dtype=np.float32)
    # 步骤1：共同前缀逐步恢复，两种λ使用完全相同的配额与重复度状态。
    for saved in ta[:step]:
        i=saved['candidate_index'];sid=groups[i];counts[sid]+=1;local=np.flatnonzero(groups==sid)
        dup[local]=np.maximum(dup[local],np.clip(features[local]@features[i],0,1))
    x,y=ta[step]['candidate_index'],tb[step]['candidate_index']
    result=dict(step=step+1,A_candidate=x,B_candidate=y,same_segment=bool(groups[x]==groups[y]),
                final_quotas_changed=pa['quotas']!=pb['quotas'],A_lambda=pa['lambda_value'],B_lambda=pb['lambda_value'])
    # 步骤2：同段时覆盖差恒为0，不能把该首次分歧直接归因于配额竞争。
    for label,lam in [('under_A',pa['lambda_value']),('under_B',pb['lambda_value'])]:
        terms=parts(scores,reps,groups,counts,dup,lam,penalty)
        result[label]=margin(y,x,pa,terms)
    return result


def analyze(source,audit):
    """覆盖50题全部A/B首次分歧、B/C变化目标及明确检查点，保存可追溯分项日志。"""
    source=Path(source);audit=Path(audit);out=audit/'full_review'/'stage_attribution';out.mkdir(parents=True,exist_ok=True)
    cfg=read_json(source/'method_config.json');penalty=cfg['redundancy_coefficient'];all_summaries=[]
    questions=read_json(audit/'questions.json');maximum=0.
    # 步骤1：固定原始记录与特征哈希；这些文件不是本轮输出，禁止改写。
    for q in questions:
        qid=q['question_id'];records={m:read_json(source/'results'/m/f'{qid}.json') for m in 'ABCD'}
        sets={m:{r['source_pts'] for r in records[m]['pool']['selected_frames']} for m in records}
        targets=(sets['A']^sets['B'])|(sets['B']^sets['C'])
        landmarks={records['B']['pool']['candidates'][i]['source_pts']:i for i in LANDMARKS.get(qid,[])}
        targets.update(landmarks)
        frames={};result=dict(question_id=qid,question=q['question'],landmarks={str(p):i for p,i in landmarks.items()},target_pts=sorted(targets),methods={})
        for m in 'ABC':
            rec=records[m];path=source/rec['feature_file']
            if sha256(path)!=rec['feature_sha256']:raise ValueError('特征文件已改变')
            features=np.load(path,mmap_mode='r',allow_pickle=False);frames[m]=features
            coarse=coarse_audit(rec,features,targets,penalty);maximum=max(maximum,coarse['maximum_gain_error'])
            result['methods'][m]=dict(coarse=coarse)
        # 步骤2：对C整段重选独立追踪；对所有目标同时记录原/扩充池排名。
        refined=refinement_audit(records['C'],frames['C'],targets,penalty);maximum=max(maximum,refined['maximum_gain_error'])
        result['methods']['C']['refinement']=refined
        if records['A']['pool']['segments']!=records['B']['pool']['segments'] or records['A']['pool']['scores']!=records['B']['pool']['scores']:
            raise ValueError('A/B输入不同，不能只归因于λ')
        if not np.array_equal(frames['A'],frames['B']):raise ValueError('A/B视觉特征不同，不能做共同状态比较')
        result['first_AB_divergence']=divergence(records['A'],records['B'],frames['A'],penalty)
        result['rankings']={}
        for m in ('B','D'):
            pool=records[m]['pool'];ranked=sorted(range(len(pool['candidates'])),key=lambda i:(-pool['scores'][i],pool['candidates'][i]['source_pts'],i))
            positions={pool['candidates'][i]['source_pts']:rank+1 for rank,i in enumerate(ranked)}
            result['rankings'][m]={str(p):positions.get(p) for p in sorted(targets)}
        result['AB_changed']=sets['A']!=sets['B'];result['BC_removed']=sorted(sets['B']-sets['C']);result['BC_added']=sorted(sets['C']-sets['B'])
        write_json(out/f'{qid}.json',result)
        all_summaries.append(dict(question_id=qid,AB_changed=result['AB_changed'],BC_removed_count=len(result['BC_removed']),BC_added_count=len(result['BC_added']),first_AB_divergence=result['first_AB_divergence']))
        print('Stage attribution',qid,flush=True)
    # 步骤3：统计不把逐步增益验证当作回答因果证明。
    changed=[r for r in all_summaries if r['AB_changed']]
    summary=dict(questions=50,AB_changed=len(changed),BC_removed=sum(r['BC_removed_count'] for r in all_summaries),BC_added=sum(r['BC_added_count'] for r in all_summaries),
                 first_divergence_same_segment=sum(r['first_AB_divergence']['same_segment'] for r in changed),
                 first_divergence_different_segment=sum(not r['first_AB_divergence']['same_segment'] for r in changed),
                 maximum_replay_gain_error=maximum,model_calls=0,scoring_calls=0,details=all_summaries,
                 warning='解释原算法的数值决策；不把局部增益分解当成答题正确率的因果分解。')
    write_json(out/'summary.json',summary)
    return summary
