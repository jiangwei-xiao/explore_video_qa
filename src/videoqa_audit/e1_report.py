"""E1两遍正式200题统计：固定历史对照，禁止择优复用答案。"""
import csv
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_methods.followups import durable
from videoqa_methods.density_report import accuracy,distribution
from .e1_run import MANIFEST,check_recovery,protocol

def summarize(run):
    """完整200调用闭合后，生成三种样本口径、分层配对区间和性能统计。"""
    manifest=read_json(MANIFEST);rows=manifest['rows'];ids=[r['question_id'] for r in rows];old=set(manifest['old_question_ids'])
    if read_json(run/'protocol.json')!=protocol():raise ValueError('Protocol changed')
    assert read_json(run/'execution.json')['status']=='completed'
    assert len(check_recovery(run,rows,sha256(run/'protocol.json')))==200
    assert len(list((run/'calls/answer').glob('*.json')))==200
    assert not list((run/'failures').glob('*.json'))
    # 步骤1：本次只计新E1回答，四组旧结果按冻结题号直接对齐。
    base=ROOT/'outputs/methods/density_extension200_20260920_r1/combined_results'
    methods={m:{q:read_json(base/m/f'{q}.json') for q in ids} for m in ('uniform','topk','v1','soft')}
    methods['e1']={q:read_json(run/'results/e1'/f'{q}.json') for q in ids}
    cohorts={'old50':manifest['old_question_ids'],'new150':manifest['new_question_ids'],'all200':ids};byid={r['question_id']:r for r in rows}
    stats={};comparisons={}
    for cohort,qids in cohorts.items():
        groups={'all':qids,**{s:[q for q in qids if byid[q]['stratum']==s] for s in ('short','medium','long')}}
        stats[cohort]={m:{s:accuracy([rec[q] for q in qs]) for s,qs in groups.items()} for m,rec in methods.items()};comparisons[cohort]={}
        # 步骤2：固定10000次、2027种子，全部对照如实报告，不挑最有利一项。
        for m in ('uniform','topk','v1','soft'):
            rng=np.random.default_rng(2027);samples=np.zeros(10000)
            for s in ('short','medium','long'):
                diff=np.array([int(methods['e1'][q]['correct'])-int(methods[m][q]['correct']) for q in groups[s]])
                samples+=rng.choice(diff,(10000,len(diff)),replace=True).sum(axis=1)/len(qids)*100
            gains=[q for q in qids if methods['e1'][q]['correct'] and not methods[m][q]['correct']]
            losses=[q for q in qids if not methods['e1'][q]['correct'] and methods[m][q]['correct']]
            comparisons[cohort]['e1-'+m]=dict(delta_pp=100*(len(gains)-len(losses))/len(qids),improved=gains,regressed=losses,bootstrap95_pp=np.percentile(samples,[2.5,97.5]).tolist(),
                both_correct=[q for q in qids if methods['e1'][q]['correct'] and methods[m][q]['correct']],both_wrong=[q for q in qids if not methods['e1'][q]['correct'] and not methods[m][q]['correct']])
    fresh=[r for r in methods['e1'].values() if r['timing_kind']=='direct_no_application_cache']
    performance={s:{k:distribution([r['timings'][k] for r in fresh if s=='all' or r['stratum']==s]) for k in ('selection_core_seconds','selection_total_seconds','qa_total_seconds','end_to_end_seconds')} for s in ('all','short','medium','long')}
    result=dict(status='completed',accuracy=stats,comparisons=comparisons,performance=performance,
        calls=dict(new_answers=200,scope=0,query=0),execution=read_json(run/'execution.json'),
        memory={k:distribution([r['memory'][k] for r in fresh]) for k in fresh[0]['memory']},
        limitation='Observed development set, not an independent confirmation set; timing across historical batches descriptive only')
    durable(run/'summary.json',result)
    with (run/'per_question.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['question_id','cohort','stratum','method','prediction','correct'])
        for row in rows:
            q=row['question_id']
            for m,rec in methods.items():writer.writerow([q,'old50' if q in old else 'new150',row['stratum'],m,rec[q]['answer']['parsed_answer'],rec[q]['correct']])
    return result
