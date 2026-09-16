"""首轮改进统计：固定问答配对、r6代理事实与离线查询，不新增模型调用。"""
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from videoqa_runtime.common import ROOT, read_json, write_json, sha256
from videoqa_runtime.baseline_report import wilson, distribution, paired_interval
from .followups import HISTORY, AUDIT, validate_e


def ranks(pool):
    """接口：按分数降序/PTS编号升序产生1起始排名；仅同题内比较，不跨题比较ITM概率。"""
    rows=pool['candidates']
    order=sorted(range(len(rows)),key=lambda i:(-pool['scores'][i],rows[i]['source_pts'],rows[i]['candidate_index']))
    return {rows[i]['source_pts']:j+1 for j,i in enumerate(order)}


def fact_metrics(pool, witnesses):
    """接口：已列见证的最佳排名及是否至少一张进入16帧；未标注候选不判为无关。"""
    rank=ranks(pool)
    present=set(witnesses)&set(rank)
    selected={r['source_pts'] for r in pool['selected_frames']}
    if not present: return dict(best_rank=None,normalized_rank=None,retained=False)
    best=min(rank[p] for p in present)
    return dict(best_rank=best,normalized_rank=(best-1)/max(1,len(rank)-1),retained=bool(present&selected))


def write_csv(path, rows):
    """接口：写小型统计表；逐题原始JSON继续作为可复算的原始证据。"""
    if not rows: return
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def summarize(run):
    """核心接口：逐项核验全部结果后输出准确率、代理事实及成本，不根据新结果改变评价分母。"""
    run=Path(run); rows=read_json(ROOT/'data/manifests/video_mme_development.json')['rows']
    fingerprint=sha256(run/'protocol.json')
    if len(list((run/'results/E').glob('*.json')))!=50 or len(list((run/'results/Q').glob('*.json')))!=50:
        raise ValueError('Need all 50 E and 50 Q results')
    # 步骤1：加载六组冻结参照和E，核对输入、上游、调用预算及所有历史哈希。
    records={m:{} for m in ('uniform','topk','A','B','C','D','E')}
    for row in rows:
        qid=row['question_id']
        for m in records:
            base=run/'results/E' if m=='E' else HISTORY/'baseline_reference/results'/m if m in ('uniform','topk') else HISTORY/'results'/m
            records[m][qid]=read_json(base/f'{qid}.json')
        validate_e(records['E'][qid],row,fingerprint)
    for name,digest in read_json(run/'protocol.json')['historical_sha256'].items():
        if sha256(ROOT/name)!=digest: raise ValueError(f'Historical artifact changed: {name}')
    calls={k:[read_json(p) for p in (run/'calls'/k).glob('*.json')] for k in ('answer','scope','query')}
    for k,items in calls.items():
        if len(items)!=50 or any(c['status']!='completed' for c in items): raise ValueError(f'Call budget/integrity: {k}')
    stats={}
    for m,items in records.items():
        groups={}
        for layer in ('all','short','medium','long'):
            group=[items[r['question_id']] for r in rows if layer=='all' or r['stratum']==layer]
            n=len(group); correct=sum(x['correct'] for x in group)
            primary=[x for x in group if x.get('timing_kind')!='resumed_partial_not_primary']
            groups[layer]=dict(n=n,correct=correct,accuracy=correct/n,wilson_95=wilson(correct,n),
                timings={k:distribution([x['timings'][k] for x in primary]) for k in primary[0]['timings']} if primary else {},
                memory={k:distribution([x['memory'][k] for x in primary]) for k in primary[0]['memory']} if primary else {},
                primary_timing_n=len(primary))
        stats[m]=groups
    comparisons={}
    for m in ('C','B','uniform','topk'):
        cells={name:[] for name in ('improved','regressed','both_correct','both_wrong','prediction_changed')}
        for row in rows:
            qid=row['question_id']; e=records['E'][qid]; old=records[m][qid]
            label='both_correct' if e['correct'] and old['correct'] else 'improved' if e['correct'] else 'regressed' if old['correct'] else 'both_wrong'
            cells[label].append(qid)
            if e['answer']['parsed_answer']!=old['answer']['parsed_answer']: cells['prediction_changed'].append(qid)
        comparisons[f'E-{m}']=dict(delta_pp=100*(stats['E']['all']['accuracy']-stats[m]['all']['accuracy']),
            bootstrap_95_pp=paired_interval(rows,{'topk':records['E'],'uniform':records[m]}),**cells)
    # 步骤2：对事前75事实评估Q，既有B正确题必要事实损失单独列出。
    manifest=read_json(run/'proxy_manifest.json'); facts=[]; question_proxy=[]; sentinel=[]
    qrecords={r['question_id']:read_json(run/'results/Q'/f'{r["question_id"]}.json') for r in rows}
    queries={r['question_id']:read_json(run/'queries'/f'{r["question_id"]}.json') for r in rows}
    for fact in manifest['entries']:
        qid=fact['question_id']; b=fact_metrics(records['B'][qid]['pool'],fact['witness_pts']); q=fact_metrics(qrecords[qid]['pool'],fact['witness_pts'])
        facts.append(dict(question_id=qid,fact_id=fact['fact_id'],role=fact['role'],description=fact['description'],
                          **{'B_'+k:v for k,v in b.items()},**{'Q_'+k:v for k,v in q.items()},B_correct=records['B'][qid]['correct']))
    for row in rows:
        qid=row['question_id']; group=[f for f in facts if f['question_id']==qid]
        if not group: continue
        question_proxy.append(dict(question_id=qid,n_facts=len(group),
            B_retention=float(np.mean([f['B_retained'] for f in group])), Q_retention=float(np.mean([f['Q_retained'] for f in group])),
            B_normalized_rank=float(np.mean([f['B_normalized_rank'] for f in group])),
            Q_normalized_rank=float(np.mean([f['Q_normalized_rank'] for f in group]))))
    for item in manifest['sentinels']:
        qid=item['question_id']; b=fact_metrics(records['B'][qid]['pool'],[item['source_pts']]); q=fact_metrics(qrecords[qid]['pool'],[item['source_pts']])
        sentinel.append(dict(**item,**{'B_'+k:v for k,v in b.items()},**{'Q_'+k:v for k,v in q.items()}))
    improved=[x['question_id'] for x in question_proxy if x['Q_retention']>x['B_retention']]
    degraded=[x['question_id'] for x in question_proxy if x['Q_retention']<x['B_retention']]
    losses=[f for f in facts if f['B_correct'] and f['role']=='required' and f['B_retained'] and not f['Q_retained']]
    query_summary=dict(denominator_questions=42,denominator_facts=75,
        fallback_count=sum(q['fallback'] for q in queries.values()),
        effective_blip_changes=sum(not q['application_cache_hits'] for q in qrecords.values()),
        B_macro_retention=float(np.mean([x['B_retention'] for x in question_proxy])),
        Q_macro_retention=float(np.mean([x['Q_retention'] for x in question_proxy])),
        B_macro_normalized_rank=float(np.mean([x['B_normalized_rank'] for x in question_proxy])),
        Q_macro_normalized_rank=float(np.mean([x['Q_normalized_rank'] for x in question_proxy])),
        improved_questions=improved,degraded_questions=degraded,B_correct_required_fact_losses=losses,
        fidelity_review='pending', query_qa_calls=0)
    query_summary['fallback_reasons']=dict(Counter(reason for q in queries.values() for reason in q['reasons']))
    if (run/'review/query_fidelity.json').exists():
        reviewed=read_json(run/'review/query_fidelity.json')
        query_summary['fidelity_review']=reviewed['status']
        query_summary['effective_faithful_changed_queries']=reviewed['effective_faithful_changed_queries']
        query_summary['semantic_drift_questions']=reviewed['changed_with_semantic_drift']
        query_summary['recommend_formal_query_qa']=bool(
            not reviewed['changed_with_semantic_drift'] and reviewed['effective_faithful_changed_queries']>0 and
            query_summary['Q_macro_retention']>query_summary['B_macro_retention'] and
            len(improved)>len(degraded) and not losses)
    query_summary['timings_all50']={k:distribution([r['timings'][k] for r in qrecords.values()]) for k in next(iter(qrecords.values()))['timings']}
    query_changes=[]
    for row in rows:
        qid=row['question_id']; before=records['B'][qid]['pool']; after=qrecords[qid]['pool']
        old_pts={r['source_pts'] for r in before['selected_frames']}; new_pts={r['source_pts'] for r in after['selected_frames']}
        query_changes.append(dict(question_id=qid,B_removed_pts=sorted(old_pts-new_pts),Q_added_pts=sorted(new_pts-old_pts),
            overlap=len(old_pts&new_pts),B_segments=before['segments'],Q_segments=after['segments'],
            B_quotas=before['quotas'],Q_quotas=after['quotas'],lambda_value=after['lambda_value'],
            blip_changed=not bool(qrecords[qid]['application_cache_hits'])))
    # 步骤3：E事实保留与输入变化独立于正确率报告，不将窗口外保留自动解释为必要证据收益。
    e_facts=[]; e_changes=[]
    for row in rows:
        qid=row['question_id']; e=records['E'][qid]['pool']; c=records['C'][qid]['pool']; b=records['B'][qid]['pool']
        getpts=lambda p:{r['source_pts'] for r in p['selected_frames']}
        e_changes.append(dict(question_id=qid,C_removed_pts=sorted(getpts(c)-getpts(e)),E_added_pts=sorted(getpts(e)-getpts(c)),
            B_overlap=len(getpts(e)&getpts(b)),C_overlap=len(getpts(e)&getpts(c)),
            E_new_selected=sum(i>=e['initial_count'] for i in e['selected_indices']),
            protected_indices=sorted({i for t in e['local_reselection_trace'] for i in t['fixed_indices']})))
        for fact in read_json(AUDIT/'full_review/witnesses_r6'/f'{qid}.json')['facts']:
            if fact['role'] not in ('required','partial'): continue
            pts={w['source_pts'] for w in fact['witnesses']}
            e_facts.append(dict(question_id=qid,fact_id=fact['id'],role=fact['role'],description=fact['description'],
                                B_retained=bool(pts&getpts(b)),C_retained=bool(pts&getpts(c)),E_retained=bool(pts&getpts(e))))
    result=dict(run_id=run.name,methods=stats,comparisons=comparisons,query=query_summary,
                calls={k:len(v) for k,v in calls.items()},failures=len(list((run/'failures').glob('*.json'))),
                historical_hashes_unchanged=True,bootstrap_repetitions=10000,seed=2027,
                E_restored_facts=[x for x in e_facts if x['E_retained'] and not x['C_retained']],
                E_lost_facts=[x for x in e_facts if x['C_retained'] and not x['E_retained']],
                executions=[read_json(p) for p in sorted(run.glob('execution_*.json'))],
                blip_calls=[read_json(p) for p in sorted((run/'blip_calls').glob('*.json'))])
    write_json(run/'summary.json',result); write_json(run/'E_selection_changes.json',e_changes)
    write_json(run/'Q_selection_changes.json',query_changes)
    write_csv(run/'Q_facts.csv',facts); write_csv(run/'Q_questions.csv',question_proxy); write_csv(run/'Q_sentinels.csv',sentinel)
    write_csv(run/'E_fact_retention.csv',e_facts)
    write_csv(run/'Q_queries.csv',[dict(question_id=r['question_id'],question=r['question'],raw_output=queries[r['question_id']]['raw_output'],
        query=queries[r['question_id']]['query'],fallback=queries[r['question_id']]['fallback'],
        reasons=';'.join(queries[r['question_id']]['reasons']),blip_changed=not bool(qrecords[r['question_id']]['application_cache_hits'])) for r in rows])
    write_csv(run/'answers.csv',[dict(question_id=r['question_id'],stratum=r['stratum'],method=m,
        correct=records[m][r['question_id']]['correct'],prediction=records[m][r['question_id']]['answer']['parsed_answer'],
        raw_output=records[m][r['question_id']]['answer']['raw_output'],reference='ABCD'[r['answer_index']]) for r in rows for m in records])
    # 步骤4：输出简明统计草稿；语义与变化图板review完成后另存最终决策，不覆盖原始结果。
    lines=['# C-local与受限查询：统计结果','',f'运行：`{run.name}`。E为新50题；其他六组为历史。Q最终问答0次。','',
           '| 方法 | 正确数/50 | 准确率 | 95% Wilson | 直接E2E或D诊断归因均值(s) |','|---|---:|---:|---|---:|']
    for m in records:
        s=stats[m]['all']; interval=s['wilson_95']; t=s['timings']
        cost=t['diagnostic_attributed_e2e_seconds']['mean'] if m=='D' else t['end_to_end_seconds']['mean']
        lines.append(f'| {m} | {s["correct"]} | {s["accuracy"]:.1%} | {interval[0]:.1%}–{interval[1]:.1%} | {cost:.3f} |')
    lines+=['','历史批次时延仅描述性比较；D不是独立算法E2E。','', '## E配对结果','']
    for key,val in comparisons.items():
        lines.append(f'- {key}：{val["delta_pp"]:+.1f}百分点，95%分层配对bootstrap {val["bootstrap_95_pp"]}；新增正确 {val["improved"]}；退化 {val["regressed"]}。')
    lines+=['','## Q代理结果','',f'固定42题75事实；回退{query_summary["fallback_count"]}/50；BLIP有效变化{query_summary["effective_blip_changes"]}/50。',
        f'按题宏平均已列事实保留：B {query_summary["B_macro_retention"]:.4f} → Q {query_summary["Q_macro_retention"]:.4f}。',
        f'按题宏平均归一化最佳排名（越低越好）：B {query_summary["B_macro_normalized_rank"]:.4f} → Q {query_summary["Q_macro_normalized_rank"]:.4f}。',
        f'保留改善题 {improved}；退化题 {degraded}。',
        f'B正确题必要事实全部已列见证损失 {len(losses)} 项。查询忠实性状态：{query_summary["fidelity_review"]}；具体语义判断见review/，代理改善不是正确率提升。','',
        '原始结果、分时长和全部耗时分位数在summary.json；逐事实/逐题CSV可复算。']
    (run/'statistics.md').write_text('\n'.join(lines)+'\n')
    return result
