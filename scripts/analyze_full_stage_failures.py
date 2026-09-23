#!/usr/bin/env python3
"""全2700题零模型调用定位：统计配对得失，提取窗口/补查/预算轨迹。"""
import argparse
import csv
import sys
from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import rows,durable,utc
from videoqa_full.report import compare
from videoqa_lmms_baseline.methods import UNIFORM,SOURCE,METHODS


def pair_cell(u,d):
    """接口：仅按真实计分划分对照单元，不把答对当作证据充分。"""
    return 'both_correct' if u and d else 'rd_gain' if d else 'rd_loss' if u else 'both_wrong'


def summarize_group(items):
    """接口：分母、三方法正确数及配对四格同时报告，机制指标仅是代理。"""
    n=len(items);out=dict(n=n,counts={m:sum(r[m] for r in items) for m in ('uniform','topk','rd')},
        cells=dict(Counter(r['cell'] for r in items)))
    out['accuracy']={m:v/n for m,v in out['counts'].items()}
    for k in ('region_count','window_fraction','new_count','new_selected','seed_retained','uniform_outside_windows','max_region_quota'):
        values=[r[k] for r in items];out[k]=dict(mean=float(np.mean(values)),median=float(np.median(values)))
    out['refined_questions']=sum(r['new_count']>0 for r in items)
    return out


def analyze(run,experiment):
    """接口：绑定同协议结果与原RD候选池，生成可重建统计和固定复核面板。"""
    run.mkdir(parents=True,exist_ok=True);data=rows();spec=read_json(experiment/'protocol.json')
    assert read_json(experiment/'completion_audit.json')['status']=='passed'
    hashes={};records=[];byid={}
    # 步骤1：全部题按qid对齐，不重新抽样，不新增BLIP或问答。
    for row in data:
        q=row['question_id']; paths={'uniform':UNIFORM/'results'/(q+'.json'),
            'topk':experiment/'results'/METHODS[0]/(q+'.json'),'rd':experiment/'results'/METHODS[1]/(q+'.json')}
        answers={m:read_json(p) for m,p in paths.items()}
        for m,p in paths.items():hashes[str(p)]=sha256(p)
        ck=read_json(SOURCE/'llava/checkpoints'/(q+'.json'));path=SOURCE/ck['pool_file']
        assert sha256(path)==ck['pool_sha256'];pool=read_json(path);hashes[str(path)]=ck['pool_sha256']
        assert pool['selected_frames']==answers['rd']['selection']['selected_frames']
        regions=pool['plan']['regions'];intervals=[(float(Fraction(g['left_fraction'])),float(Fraction(g['right_fraction']))) for g in regions]
        inside=lambda t:any(l<=t<=r for l,r in intervals)
        seeds=set(pool['plan']['seed_indices']);chosen=set(pool['selected_indices']);initial=pool['initial_count']
        quotas=pool['selection_trace']['region_budgets'];u,d,t=[answers[m]['correct'] for m in ('uniform','rd','topk')]
        rec=dict(question_id=q,video_id=row['video_id'],stratum=row['stratum'],task_type=row['task_type'],
            question=row['question'],options=row['options'],answer='ABCD'[row['answer_index']],uniform=u,topk=t,rd=d,cell=pair_cell(u,d),
            predictions={m:a['result']['answer']['parsed_answer'] for m,a in answers.items()},
            initial_count=initial,region_count=len(regions),window_fraction=sum(r-l for l,r in intervals)/pool['video']['duration_seconds'],
            new_count=pool['new_candidate_count'],new_selected=sum(i>=initial for i in chosen),seed_retained=len(seeds&chosen),
            max_region_quota=max(quotas.values(),default=0),uniform_outside_windows=sum(not inside(f['timestamp_seconds']) for f in answers['uniform']['selection']['selected_frames']),
            regions=regions,region_budgets=quotas,selected_indices=pool['selected_indices'],timings=ck['original_timings'])
        records.append(rec);byid[q]=rec
    # 步骤2：官方题型不替代人工局部/全局证据布局；分时长分题型都保留分母。
    tables={}
    for dim in ('stratum','task_type','cell'):
        groups=defaultdict(list)
        for r in records:groups[r[dim]].append(r)
        tables[dim]={key:summarize_group(v) for key,v in groups.items()}
    tables['stratum_task']={s:{task:summarize_group([r for r in records if r['stratum']==s and r['task_type']==task])
        for task in sorted({r['task_type'] for r in records if r['stratum']==s})} for s in ('short','medium','long')}
    paired={s:compare([r for r in data if r['stratum']==s],
        {r['question_id']:dict(correct=r['rd']) for r in records},{r['question_id']:dict(correct=r['uniform']) for r in records}) for s in ('short','medium','long')}
    # 步骤3：固定15题有界复核，既含损失、收益，也含共同失败与共同正确控制。
    bad_priority=['Temporal Perception','Temporal Reasoning','Counting Problem','Action Reasoning','Action Recognition']
    good_priority=['OCR Problems','Object Recognition','Attribute Perception','Action Recognition']
    panel=[]
    for s in ('short','medium','long'):
        for cell,top in [('rd_loss',True),('rd_loss',False),('rd_gain',None),('both_wrong',None),('both_correct',None)]:
            choices=[r for r in records if r['stratum']==s and r['cell']==cell and (top is None or r['topk']==top)]
            priority=good_priority if cell in ('rd_gain','both_correct') else bad_priority
            choices.sort(key=lambda r:(priority.index(r['task_type']) if r['task_type'] in priority else 99,r['question_id']))
            if choices:
                r=choices[0];panel.append(dict(question_id=r['question_id'],stratum=s,cell=cell,topk_condition=top,
                    task_type=r['task_type'],selection_reason='frozen category priority then ascending qid; not visual success filtering'))
    identity=dict(utc=utc(),experiment=str(experiment),experiment_protocol_sha256=sha256(experiment/'protocol.json'),
        code_sha256=sha256(__file__),source_hashes=hashes,new_model_calls=0,
        limitation='Descriptive proxies and official task labels; no semantic evidence sufficiency inferred from correctness')
    durable(run/'protocol.json',identity);durable(run/'per_question.json',records)
    durable(run/'summary.json',dict(tables=tables,paired_by_duration=paired,questions=2700,new_model_calls=0))
    durable(run/'review_panel.json',panel)
    fields=['question_id','video_id','stratum','task_type','uniform','topk','rd','cell','region_count','window_fraction','new_count','new_selected','seed_retained','max_region_quota','uniform_outside_windows','question']
    with (run/'per_question.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in records)
    print(tables['stratum']);print(tables['task_type']);print('PANEL',panel)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    assert Path(a.run_id).name==a.run_id and a.run_id not in ('.','..')
    analyze(ROOT/'outputs/analysis'/a.run_id,ROOT/'outputs/full_videoqa/topk_rd12__llava__lmms_ref_v1__videomme2700__20260923__r01')
