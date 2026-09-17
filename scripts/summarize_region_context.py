#!/usr/bin/env python3
"""R实验只读核对与统计：不调用模型，不改变执行版本、旧实验或逐题结果。"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, write_json, sha256
from videoqa_methods.region_run import execution_protocol, validate_formal, check_gate


def distribution(values):
    """接口：以秒为单位报告均值和百分位，不将阶段相加冒充直接E2E。"""
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                p90=float(np.percentile(a, 90)), p95=float(np.percentile(a, 95)))


def accuracy(records):
    """正确率与95% Wilson区间；保持原始计分，解析失败不剔除。"""
    n = len(records); k = sum(r['correct'] for r in records); p = k / n; z = 1.959963984540054
    center = (p + z*z/(2*n)) / (1+z*z/n)
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(correct=k, n=n, accuracy=p, wilson95=[center-radius, center+radius])


def summarize(run):
    """步骤1核对50题及调用；步骤2复算配对统计；步骤3保存派生产物及输入指纹。"""
    rows = read_json(ROOT/'data/manifests/video_mme_development.json')['rows']
    ids = [r['question_id'] for r in rows]; fingerprint = sha256(run/'protocol.json')
    assert read_json(run/'protocol.json') == execution_protocol(), 'Execution/old asset fingerprint changed'
    check_gate(run, fingerprint)
    assert read_json(run/'execution.json')['status'] == 'formal_completed'
    assert {p.stem for p in (run/'results').glob('*.json')} == set(ids)
    sources = {'R':run/'results', 'TopK':ROOT/'outputs/baselines/videomme50_uniform_topk_20260914_r1/results/topk',
               'Uniform':ROOT/'outputs/baselines/videomme50_uniform_topk_20260914_r1/results/uniform',
               'B':ROOT/'outputs/methods/method_v1_videomme50_20260914_r1/results/B',
               'C-local':ROOT/'outputs/methods/followups_clocal_query_videomme50_20260915_r1/results/E'}
    records = {m:{q:read_json(p/f'{q}.json') for q in ids} for m,p in sources.items()}
    for row in rows: validate_formal(run, records['R'][row['question_id']], row, fingerprint)
    calls = [read_json(p) for p in (run/'calls/answer').glob('*.json')]
    assert len(calls) == 50 and {c['question_id'] for c in calls} == set(ids)
    assert all(c['status']=='completed' for c in calls)
    assert all(r['timing_kind']=='direct_no_application_cache' for r in records['R'].values())
    summary = dict(accuracy={}, comparisons={}, performance={}, calls=dict(answer_started=len(calls),
                   answer_completed=len(calls), scope=0, query=0), execution=read_json(run/'execution.json'))
    groups = {'all':ids, **{s:[r['question_id'] for r in rows if r['stratum']==s] for s in ('short','medium','long')}}
    for method, results in records.items():
        summary['accuracy'][method] = {s:accuracy([results[q] for q in qs]) for s,qs in groups.items()}
    # 配对且按时长分层抽样；每个比较均使用同一预定随机种子和10000次重复。
    for method in ('TopK','B','C-local','Uniform'):
        rng=np.random.default_rng(2027); samples=np.zeros(10000)
        for s in ('short','medium','long'):
            d=np.array([int(records['R'][q]['correct'])-int(records[method][q]['correct']) for q in groups[s]])
            samples += rng.choice(d, size=(10000,len(d)), replace=True).sum(axis=1)/50*100
        improved=[q for q in ids if records['R'][q]['correct'] and not records[method][q]['correct']]
        regressed=[q for q in ids if records[method][q]['correct'] and not records['R'][q]['correct']]
        summary['comparisons']['R-'+method]=dict(delta_pp=(len(improved)-len(regressed))*2,
            improved=improved,regressed=regressed,bootstrap95_pp=np.percentile(samples,[2.5,97.5]).tolist(),
            both_correct=[q for q in ids if records['R'][q]['correct'] and records[method][q]['correct']],
            both_wrong=[q for q in ids if not records['R'][q]['correct'] and not records[method][q]['correct']])
    metrics=list(records['R'][ids[0]]['timings'])
    for s,qs in groups.items():
        summary['performance'][s]={k:distribution([records['R'][q]['timings'][k] for q in qs]) for k in metrics}
    summary['memory']={k:distribution([r['memory'][k] for r in records['R'].values()]) for k in records['R'][ids[0]]['memory']}
    summary['memory']['maximum_worker_peak_gib']=max(r['memory']['worker_peak_allocated_gib'] for r in records['R'].values())
    summary['preflight_seconds']=read_json(run/'formal_preflight.json')['seconds']
    summary['prediction_changes_vs_topk']=[q for q in ids if records['R'][q]['answer']['parsed_answer']!=records['TopK'][q]['answer']['parsed_answer']]
    summary['historical_performance_descriptive']={m:{k:distribution([r['timings'][k] for r in results.values()])
        for k in ('selection_core_seconds','selection_total_seconds','end_to_end_seconds')} for m,results in records.items() if m!='R'}
    off=read_json(run/'offline_summary.json'); selection=[records['R'][q]['selection'] for q in ids]
    summary['selection']=dict(changed_questions=sum(bool(s['replacements']) for s in selection),
        total_replacements=sum(len(s['replacements']) for s in selection),new_candidates=sum(s['new_candidate_count'] for s in selection),
        max_new_candidates=max(s['new_candidate_count'] for s in selection),listed_fact_losses=len(off['issues']),
        scored_initial_frames=sum(s['initial_count'] for s in selection),
        blip_real_forwards=sum(math.ceil(s['initial_count']/16) for s in selection),fine_blip_forwards=0,
        offline_wall_seconds=off['wall_seconds'])
    summary['workers']=[read_json(p) for p in (run/'workers').glob('*.json')]
    summary['failures']=[read_json(p) for p in (run/'failures').glob('*.json')]
    write_json(run/'summary.json',summary)
    with (run/'answers.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['question_id','stratum','reference','R_raw','R_prediction','R_correct',
            'TopK_prediction','TopK_correct','B_correct','C-local_correct','Uniform_correct','replacements','selection_seconds','e2e_seconds'])
        for row in rows:
            q=row['question_id'];r=records['R'][q];t=records['TopK'][q]
            writer.writerow([q,row['stratum'],r['reference_answer'],r['answer']['raw_output'],r['answer']['parsed_answer'],r['correct'],
                t['answer']['parsed_answer'],t['correct'],records['B'][q]['correct'],records['C-local'][q]['correct'],records['Uniform'][q]['correct'],
                len(r['selection']['replacements']),r['timings']['selection_total_seconds'],r['timings']['end_to_end_seconds']])
    files=[p/f'{q}.json' for p in sources.values() for q in ids]+[run/'summary.json',run/'answers.csv',Path(__file__)]
    write_json(run/'analysis_manifest.json',{str(p.relative_to(ROOT)):sha256(p) for p in files})
    print(json.dumps({k:summary[k] for k in ('accuracy','comparisons','selection','calls')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    summarize(ROOT/'outputs/methods'/parser.parse_args().run_id)
