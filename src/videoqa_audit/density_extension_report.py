"""200题四版本统计；旧50、新150和合计分别报告，来源不冒充新测。"""
import csv
from itertools import combinations
from pathlib import Path
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_methods.followups import durable
from videoqa_methods.density_report import accuracy,distribution
from .density_extension import MANIFEST,LINKS,ORIGINALS,VARIANTS,protocol,check_recovery


def source_inventory(run):
    """接口：准备阶段冻结500条可复用结果，尚未生成的新结果不记作完成。"""
    manifest=read_json(MANIFEST);links=read_json(LINKS);items=[]
    for row in manifest['rows']:
        q=row['question_id'];cohort='old50' if q in manifest['old_question_ids'] else 'new150'
        for method,ref in links['references'][q].items():
            assert sha256(ROOT/ref['path'])==ref['sha256']
            items.append(dict(question_id=q,method=method,cohort=cohort,origin='historical_full_baseline',**ref))
        if cohort=='old50':
            for method,base in ORIGINALS.items():
                p=base/'results'/f'{q}.json';items.append(dict(question_id=q,method=method,cohort=cohort,
                    origin='historical_development50',path=str(p.relative_to(ROOT)),sha256=sha256(p)))
    assert len(items)==500
    durable(Path(run)/'source_inventory.json',dict(reused_results=500,pending_new_answers=300,items=items))


def summarize(run):
    """接口：必须300次新增调用闭合才输出四版本各200题成绩与对照。"""
    run=Path(run);manifest=read_json(MANIFEST);rows=manifest['rows'];spec=read_json(run/'protocol.json')
    assert spec==protocol() and read_json(run/'execution.json')['status']=='completed'
    fingerprint=sha256(run/'protocol.json');new=set(manifest['new_question_ids'])
    assert len(check_recovery(run,[r for r in rows if r['question_id'] in new],fingerprint))==150
    assert len(list((run/'calls/answer').glob('*.json')))==300
    assert all(not list((run/'calls'/kind).glob('*.json')) for kind in ('scope','query'))
    links=read_json(LINKS);methods={m:{} for m in ('uniform','topk','v1','soft')};origin={}
    # 步骤1：按显式来源汇合，绝不将同题旧新答案择优拼接。
    for row in rows:
        q=row['question_id']
        for method in methods:
            if method in ('uniform','topk'):p=ROOT/links['references'][q][method]['path']
            elif q in new:p=run/'results'/method/f'{q}.json'
            else:p=ORIGINALS[method]/'results'/f'{q}.json'
            methods[method][q]=read_json(p);origin[q,method]=str(p.relative_to(ROOT))
    cohorts={'old50':manifest['old_question_ids'],'new150':manifest['new_question_ids'],'all200':[r['question_id'] for r in rows]}
    byid={r['question_id']:r for r in rows};stats={};comparisons={}
    # 步骤2：分时长、分队列报告；每视频仅一道题，分层配对bootstrap按视频等价抽样。
    for cohort,ids in cohorts.items():
        groups={'all':ids,**{s:[q for q in ids if byid[q]['stratum']==s] for s in ('short','medium','long')}}
        stats[cohort]={m:{s:accuracy([records[q] for q in qids]) for s,qids in groups.items()} for m,records in methods.items()}
        comparisons[cohort]={}
        for left,right in combinations(methods,2):
            rng=np.random.default_rng(2027);samples=np.zeros(10000)
            for s in ('short','medium','long'):
                diff=np.array([int(methods[right][q]['correct'])-int(methods[left][q]['correct']) for q in groups[s]])
                samples+=rng.choice(diff,size=(10000,len(diff)),replace=True).sum(axis=1)/len(ids)*100
            improved=[q for q in ids if methods[right][q]['correct'] and not methods[left][q]['correct']]
            regressed=[q for q in ids if not methods[right][q]['correct'] and methods[left][q]['correct']]
            comparisons[cohort][right+'-'+left]=dict(delta_pp=100*(len(improved)-len(regressed))/len(ids),
                improved=improved,regressed=regressed,bootstrap95_pp=np.percentile(samples,[2.5,97.5]).tolist(),
                both_correct=[q for q in ids if methods[left][q]['correct'] and methods[right][q]['correct']],
                both_wrong=[q for q in ids if not methods[left][q]['correct'] and not methods[right][q]['correct']])
    performance={};mechanisms={}
    for variant in VARIANTS:
        fresh=[methods[variant][q] for q in manifest['new_question_ids'] if methods[variant][q]['timing_kind']=='direct_no_application_cache']
        performance[variant]={key:distribution([r['timings'][key] for r in fresh]) for key in ('selection_core_seconds','selection_total_seconds','qa_total_seconds','end_to_end_seconds')}
        mechanisms[variant]=dict(new_candidates=sum(methods[variant][q]['selection']['new_candidate_count'] for q in new),
            selected_new=sum(sum(i>=methods[variant][q]['selection']['initial_count'] for i in methods[variant][q]['selection']['selected_indices']) for q in new))
    result=dict(status='completed',accuracy=stats,comparisons=comparisons,new150_performance=performance,new150_mechanisms=mechanisms,
        calls=dict(new_answers=300,reused_answers=500,scope=0,query=0),execution=read_json(run/'execution.json'),
        limitation='Expanded observed development assessment, not an untouched independent test set; historical timings across batches descriptive only')
    durable(run/'summary.json',result)
    with (run/'per_question.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['question_id','cohort','stratum','task_type','method','prediction','correct','source'])
        for row in rows:
            q=row['question_id']
            for method in methods:
                r=methods[method][q];writer.writerow([q,'new150' if q in new else 'old50',row['stratum'],row['task_type'],method,r['answer']['parsed_answer'],r['correct'],origin[q,method]])
    lines=['# 开发200题四版本结果','','| 方法 | 原50题 | 新增150题 | 全部200题 |','|---|---:|---:|---:|']
    for method in methods:
        lines.append('| '+method+' | '+' | '.join(f'{stats[c][method]["all"]["correct"]}/{len(cohorts[c])} ({stats[c][method]["all"]["accuracy"]:.1%})' for c in cohorts)+' |')
    lines+=['','新增150题及合计200题是不同口径，主要用新增样本检查小样本现象是否延续。',
        '完整分时长、Wilson区间、六组配对bootstrap/得失、性能及来源见summary.json和per_question.csv。',
        '不按本轮结果重新抽题或自动调参；旧结果未覆盖。']
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    return result
