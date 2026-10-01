#!/usr/bin/env python3
"""200题固定配额窗内重选；只检验窗外竞争，不调用模型或复用预测。"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.window_competition import select_inside_windows
from videoqa_audit.option_entry_attribution import nearest_similarities
from videoqa_runtime.baseline_report import distribution
SOURCE=ROOT/'outputs/methods/rd12_exp08__qwen32__dev200__20261001_r1'
PROXY=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'


def prepare(run):
    """接口：固定200题与原配额，来源绑定完整资产，不读答案挑题。"""
    run.mkdir(parents=True,exist_ok=True);parent=read_json(SOURCE/'protocol.json');sources={}
    for q in parent['question_ids']:
        for p in (SOURCE/'pools'/(q+'.json'),SOURCE/'features'/(q+'.npy'),SOURCE/'traces'/(q+'.json'),SOURCE/'selections'/(q+'.json')):sources[str(p)]=sha256(p)
    for p in (PROXY,ROOT/'src/videoqa_audit/window_competition.py',ROOT/'scripts/analyze_window_competition.py',ROOT/'src/videoqa_methods/weighted_seed_support.py'):
        sources[str(p)]=sha256(p)
    spec=dict(question_ids=parent['question_ids'],sources=sources,configuration=parent['configuration'],new_model_calls=0,
        scope='Fixed exp08 quotas; original support; only remove outside-window candidate eligibility. No QA accuracy.')
    if (run/'protocol.json').exists():assert read_json(run/'protocol.json')==spec
    else:durable(run/'protocol.json',spec)
    return spec


def worker(job):
    """接口：每题保持32帧和全部原区域名额；不具可行性则暂停。"""
    run,q=Path(job[0]),job[1];spec=read_json(run/'protocol.json');dest=run/'questions'/(q+'.json')
    if dest.exists():
        value=read_json(dest);assert value['protocol_sha256']==sha256(run/'protocol.json');return value
    paths=[SOURCE/'pools'/(q+'.json'),SOURCE/'features'/(q+'.npy'),SOURCE/'traces'/(q+'.json'),SOURCE/'selections'/(q+'.json')]
    for p in paths:assert sha256(p)==spec['sources'][str(p)]
    pool=read_json(paths[0]);features=np.load(paths[1],allow_pickle=False);quotas=read_json(paths[2])['region_budgets'];old=read_json(paths[3])
    selected,trace=select_inside_windows(pool,features,quotas,spec['configuration'])
    assert len(selected)==len(old['selected_indices']) and set(trace['protected_indices'])<=set(selected)
    # 步骤1：原选择若本就全在窗内，删除未选窗外候选不应改变已选结果。
    if set(old['selected_indices'])<=set(trace['eligible_indices']):assert selected==old['selected_indices'],('Unexpected in-window drift',q)
    # 步骤2：只保存选帧及轨迹，绝不附用原exp08的回答或正确率。
    result=dict(question_id=q,protocol_sha256=sha256(run/'protocol.json'),selected_indices=selected,
        selected_frames=[pool['candidates'][i] for i in selected],trace=trace,
        original_selected_frames=old['selection']['selected_frames'],source_overlap=len(set(selected)&set(old['selected_indices'])),
        nearest_cosine=nearest_similarities(features,selected),new_model_calls=0,qa_prediction=None)
    durable(dest,result);return result


def summarize(run,spec):
    """接口：同目录同分母记录得失，未列精确见证不是语义真值。"""
    records={q:read_json(run/'questions'/(q+'.json')) for q in spec['question_ids']};facts=read_json(PROXY)['entries'];evaluation=[]
    for f in facts:
        r=records[f['question_id']];pts=set(f['witness_pts'])
        evaluation.append(dict(question_id=f['question_id'],fact_id=f['fact_id'],role=f['role'],
            original=bool(pts&{x['source_pts'] for x in r['original_selected_frames']}),
            window=bool(pts&{x['source_pts'] for x in r['selected_frames']})))
    qs={r['question_id'] for r in evaluation};assert len(evaluation)==75 and len(qs)==42;metrics={}
    for name in ('original','window'):
        metrics[name]=dict(retained=sum(r[name] for r in evaluation),macro=sum(sum(r[name] for r in evaluation if r['question_id']==q)/sum(r['question_id']==q for r in evaluation) for q in qs)/42)
    result=dict(status='completed',metrics=metrics,facts=evaluation,
        gained=[(r['question_id'],r['fact_id']) for r in evaluation if r['window'] and not r['original']],
        lost=[(r['question_id'],r['fact_id']) for r in evaluation if r['original'] and not r['window']],
        changed_questions=[q for q,r in records.items() if r['selected_frames']!=r['original_selected_frames']],
        overlap=distribution([r['source_overlap'] for r in records.values()]),
        nearest_cosine_question_means=distribution([float(np.mean(r['nearest_cosine'])) for r in records.values()]),
        new_model_calls=0,qa_accuracy_not_defined=True,utc=utc())
    durable(run/'summary.json',result);print({k:v for k,v in result.items() if k not in ('facts','changed_questions')},flush=True)


def execute(run):
    """接口：后台CPU重放，完成题按指纹跳过；模型调用始终为0。"""
    with exclusive(run/'coordinator.lock'):
        spec=prepare(run)
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as executor:
            futures=[executor.submit(worker,(str(run),q)) for q in spec['question_ids']]
            for i,future in enumerate(as_completed(futures),1):
                future.result();durable(run/'status.json',dict(status='running',completed=i,total=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        summarize(run,spec);durable(run/'status.json',dict(status='completed',completed=200,total=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    import re
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id)
    execute(ROOT/'outputs/analysis'/args.run_id)
