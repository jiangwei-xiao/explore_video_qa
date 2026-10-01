#!/usr/bin/env python3
"""多种子帧组固定双条件离线准备：纯覆盖控制与覆盖加成对完成，零模型调用。"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import re
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.query_input_probe import BASE,FULL
from videoqa_methods.seed_bundle_selection import select_seed_bundles
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
FACTS=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'
CONDITIONS=('coverage_only','coverage_with_pairs')


def prepare(run):
    """接口：冻结原池、预算参照、代码与两个明确对照；不根据输出选择参数。"""
    ids=[r['question_id'] for r in read_json(DATA)['rows']];sources={str(DATA):sha256(DATA),str(FACTS):sha256(FACTS)};inputs={}
    for name in ('src/videoqa_methods/seed_bundle_selection.py','scripts/analyze_seed_bundles.py'):
        sources[str(ROOT/name)]=sha256(ROOT/name)
        dest=run/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha256(dest)==sources[str(ROOT/name)]
        else:shutil.copy2(ROOT/name,dest)
    for q in ids:
        paths=dict(pool=BASE/'pools'/(q+'.json'),features=BASE/'features'/(q+'.npy'),trace=FULL/'traces'/(q+'.json'),
                   selection=FULL/'selections/RD-1.2-exp07'/(q+'.json'))
        inputs[q]={k:str(v) for k,v in paths.items()}
        for p in paths.values():sources[str(p)]=sha256(p)
    protocol=dict(question_ids=ids,inputs=inputs,sources=sources,conditions=list(CONDITIONS),radius_seconds=4,budget=32,
        pair_weight='seed relative original ITM times cos*sqrt(1-cos); duplicate pairs aggregate then normalize',
        objective='coverage and pair completion equally weighted after remaining-range normalization from protected anchors',
        single_control='same greedy coverage, pair term disabled; no bundle can strictly beat all constituent singletons per cost',
        new_model_calls=0,qa_not_run=True)
    if (run/'protocol.json').exists():assert read_json(run/'protocol.json')==protocol
    else:durable(run/'protocol.json',protocol)
    return protocol


def worker(args):
    """接口：每题两个冻结条件只读同一资产，人工事实只供最终汇总。"""
    directory,q=args;run=Path(directory);spec=read_json(run/'protocol.json');dest=run/'questions'/(q+'.json')
    if dest.exists():
        saved=read_json(dest);assert saved['protocol_sha256']==sha256(run/'protocol.json');return q
    paths=spec['inputs'][q]
    for p in paths.values():assert sha256(p)==spec['sources'][p]
    pool=read_json(paths['pool']);f=np.load(paths['features'],allow_pickle=False);trace=read_json(paths['trace']);conditions={}
    for name in CONDITIONS:
        selected,detail=select_seed_bundles(pool,f,trace,budget=32,with_pairs=name=='coverage_with_pairs')
        conditions[name]=dict(selected_indices=selected,selected_frames=[pool['candidates'][i] for i in selected],trace=detail)
    durable(dest,dict(question_id=q,protocol_sha256=sha256(run/'protocol.json'),conditions=conditions,
        exp07_indices=read_json(paths['selection'])['selected_indices'],new_model_calls=0))
    return q


def summarize(run,spec):
    """接口：报告成组是否真实发生、完整见证增减，不附用旧答案。"""
    records={q:read_json(run/'questions'/(q+'.json')) for q in spec['question_ids']};facts=[]
    for f in read_json(FACTS)['entries']:
        q=f['question_id'];pool=read_json(spec['inputs'][q]['pool']);w=set(f['witness_pts']);r=records[q]
        ids={name:value['selected_indices'] for name,value in r['conditions'].items()};ids['exp07']=r['exp07_indices']
        facts.append(dict(question_id=q,fact_id=f['fact_id'],hits={name:bool(w&{pool['candidates'][i]['source_pts'] for i in values}) for name,values in ids.items()}))
    qids={f['question_id'] for f in facts};assert len(facts)==75 and len(qids)==42
    metrics={name:dict(retained=sum(f['hits'][name] for f in facts),macro=sum(sum(f['hits'][name] for f in facts if f['question_id']==q)/sum(f['question_id']==q for f in facts) for q in qids)/42) for name in (*CONDITIONS,'exp07')}
    changes={ref:{kind:[(f['question_id'],f['fact_id']) for f in facts if (f['hits'][ref],f['hits']['coverage_with_pairs'])==pattern]
                 for kind,pattern in [('gained',(False,True)),('lost',(True,False))]} for ref in ('coverage_only','exp07')}
    summary=dict(status='completed',question_count=200,metrics=metrics,changes=changes,facts=facts,new_model_calls=0,
        multi_frame_questions=sum(r['conditions']['coverage_with_pairs']['trace']['multi_frame_steps']>0 for r in records.values()),
        multi_frame_steps=sum(r['conditions']['coverage_with_pairs']['trace']['multi_frame_steps'] for r in records.values()),
        scope='Nonexhaustive fact proxies only; no new answers, no semantic gain inferred from group count')
    durable(run/'summary.json',summary);print({k:v for k,v in summary.items() if k!='facts'},flush=True)


def execute(run):
    """接口：四个双线程CPU进程后台完成固定对照，逐题落盘，恢复跳过完整结果。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'run.lock'):
        spec=prepare(run)
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
            jobs=[pool.submit(worker,(str(run),q)) for q in spec['question_ids']]
            for n,future in enumerate(as_completed(jobs),1):
                future.result();durable(run/'status.json',dict(status='running',completed=n,total=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        summarize(run,spec);durable(run/'status.json',dict(status='completed',completed=200,total=200,pid=os.getpid(),utc=utc()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id)
    execute(ROOT/'outputs/analysis'/a.run_id)
