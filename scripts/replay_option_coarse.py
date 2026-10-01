#!/usr/bin/env python3
"""200题零模型配对重放：只变种子集合，不把离线选择附用旧答案。"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.query_input_probe import BASE,ranking
from videoqa_audit.option_coarse_replay import plan_from_seeds,replay_coarse
from videoqa_audit.option_entry_attribution import nearest_similarities
from videoqa_budget32.selection import config
from videoqa_runtime.baseline_report import distribution
SOURCE=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1'
CONDITIONS=('question_seeds','option_seeds')


def prepare(run):
    """接口：固定200题和同粗池两条件，评价目录另存且不传给选择函数。"""
    run.mkdir(parents=True,exist_ok=True)
    ids=read_json(SOURCE/'protocol.json')['question_ids']
    assert ids==[r['question_id'] for r in read_json(ROOT/'data/manifests/video_mme_development_200_20260920.json')['rows']]
    sources={};inputs={}
    for q in ids:
        files=dict(pool=BASE/'pools'/(q+'.json'),features=BASE/'features'/(q+'.npy'),check=BASE/'selection_checkpoints'/(q+'.json'),
                   options=SOURCE/'results'/(q+'.json'))
        inputs[q]={k:str(p) for k,p in files.items()}
        for p in files.values():sources[str(p)]=sha256(p)
    names=['src/videoqa_audit/option_coarse_replay.py','scripts/replay_option_coarse.py',
           'src/videoqa_audit/option_entry_attribution.py','src/videoqa_methods/density_selection.py',
           'src/videoqa_methods/coarse_eligibility.py','src/videoqa_audit/density_reward_decay.py',
           'src/videoqa_methods/prefix_coverage_budget.py','src/videoqa_methods/weighted_seed_support.py',
           'configs/retrieval_density_v1.json']
    for name in names:sources[str(ROOT/name)]=sha256(ROOT/name)
    sources[str(SOURCE/'witness_evaluation_protocol.json')]=sha256(SOURCE/'witness_evaluation_protocol.json')
    value=dict(question_ids=ids,inputs=inputs,sources=sources,configuration=config(32,32),conditions=CONDITIONS,
               source_method='RD-1.2-exp07',new_model_calls=0,fine_candidates=0,
               scope='Paired coarse-only replay. Seed membership changes; original question scores, local prefix and DP remain.')
    # 步骤1：先固定所有来源，恢复不得静默接纳新字节。
    if (run/'protocol.json').exists():
        old=read_json(run/'protocol.json');value['conditions']=list(value['conditions']);assert old==value
    else:durable(run/'protocol.json',value)
    return value


def worker(job):
    """接口：单题两条件共享同一粗池和CLS，原种子重建必须与旧区域逐字段一致。"""
    run,q=Path(job[0]),job[1];spec=read_json(run/'protocol.json');dest=run/'questions'/(q+'.json')
    if dest.exists():
        result=read_json(dest);assert result['protocol_sha256']==sha256(run/'protocol.json');return result
    files={k:Path(p) for k,p in spec['inputs'][q].items()}
    for p in files.values():assert sha256(p)==spec['sources'][str(p)]
    pool=read_json(files['pool']);check=read_json(files['check']);n=pool['initial_count']
    for p in (files['pool'],files['features']):assert sha256(p)==check['files'][str(p.relative_to(BASE))]
    candidates=pool['candidates'][:n];features=np.load(files['features'],allow_pickle=False)[:n].copy();scores=pool['initial_scores']
    old_seeds=ranking(candidates,scores)[:min(32,n)];new_seeds=read_json(files['options'])['selected_indices']
    assert plan_from_seeds(candidates,scores,old_seeds,pool['video'],spec['configuration'])==pool['plan']
    # 步骤1：新旧条件均不读取旧细帧前缀以外的内容，不借用旧补查池。
    outcomes={};start=time.perf_counter()
    for name,seeds in zip(CONDITIONS,(old_seeds,new_seeds)):
        result=replay_coarse(candidates,features,scores,seeds,pool['video'],spec['configuration'])
        result['selected_frames']=[candidates[i] for i in result['selected_indices']]
        result['nearest_cosine']=nearest_similarities(features,result['selected_indices'])
        outcomes[name]=result
    record=dict(question_id=q,protocol_sha256=sha256(run/'protocol.json'),old_plan_exact=True,initial_count=n,
                conditions=outcomes,seconds=time.perf_counter()-start,new_model_calls=0)
    durable(dest,record);return record


def summarize(run,spec):
    """接口：只评价选择和冻结见证；无正确率字段，也不复制历史预测。"""
    records=[read_json(run/'questions'/(q+'.json')) for q in spec['question_ids']]
    assert len(records)==200 and all(r['old_plan_exact'] and r['new_model_calls']==0 for r in records)
    facts=read_json(SOURCE/'witness_evaluation_protocol.json')['entries'];assert len(facts)==75
    byid={r['question_id']:r for r in records};evaluation=[];metrics={}
    for fact in facts:
        r=byid[fact['question_id']];hits={}
        for name in CONDITIONS:
            hits[name]=bool(set(fact['witness_pts'])&{f['source_pts'] for f in r['conditions'][name]['selected_frames']})
        evaluation.append(dict(question_id=fact['question_id'],fact_id=fact['fact_id'],role=fact['role'],retained=hits))
    qids={f['question_id'] for f in facts};assert len(qids)==42
    # 步骤2：同时报告命中数量、宏平均、区域/预算变化及外观代理，不做择优问答。
    for name in CONDITIONS:
        macro=sum(sum(v['retained'][name] for v in evaluation if v['question_id']==q)/sum(v['question_id']==q for v in evaluation) for q in qids)/42
        metrics[name]=dict(facts_retained=sum(v['retained'][name] for v in evaluation),facts_total=75,macro_retention=macro,
            nearest_cosine_question_means=distribution([float(np.mean(r['conditions'][name]['nearest_cosine'])) for r in records]),
            regions=distribution([len(r['conditions'][name]['plan']['regions']) for r in records]),
            actual_frames=dict(Counter(len(r['conditions'][name]['selected_indices']) for r in records)))
    change={}
    for key,wanted in [('gained',(False,True)),('lost',(True,False))]:
        change[key]=[(v['question_id'],v['fact_id']) for v in evaluation if tuple(v['retained'][n] for n in CONDITIONS)==wanted]
    durable(run/'summary.json',dict(status='completed',metrics=metrics,facts=evaluation,changes=change,
        changed_questions=[r['question_id'] for r in records if r['conditions'][CONDITIONS[0]]['selected_indices']!=r['conditions'][CONDITIONS[1]]['selected_indices']],
        new_model_calls=0,qa_accuracy_not_defined=True,utc=utc()))
    print(metrics,change,flush=True)


def execute(run):
    """接口：CPU后台配对重放，完成题跳过；所有原资产只读，失败保留已完成结果。"""
    with exclusive(run/'coordinator.lock'):
        spec=prepare(run);start=time.perf_counter();done=0
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as executor:
            futures=[executor.submit(worker,(str(run),q)) for q in spec['question_ids']]
            for future in as_completed(futures):
                future.result();done+=1
                durable(run/'status.json',dict(status='running',completed=done,total=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        summarize(run,spec)
        durable(run/'status.json',dict(status='completed',completed=200,total=200,pid=os.getpid(),sid=os.getsid(0),seconds=time.perf_counter()-start,utc=utc()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    import re
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id)
    execute(ROOT/'outputs/analysis'/args.run_id)
