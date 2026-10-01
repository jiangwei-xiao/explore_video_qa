#!/usr/bin/env python3
"""同粗池四路/五路入口对照：200题主统计，固定两例单列，模型调用0。"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction
import multiprocessing as mp
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.query_input_probe import BASE,WITNESSES
from videoqa_audit.option_rank_selection import select_option_ranks
from videoqa_audit.option_coarse_replay import replay_coarse
from videoqa_audit.joint_query_views import select_joint_views
from videoqa_methods.density_selection import source_time
from videoqa_budget32.selection import config
from videoqa_runtime.baseline_report import distribution
FOUR=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1'
REPLAY=ROOT/'outputs/analysis/option_coarse_replay200_20261001_r1'
TWO=ROOT/'outputs/analysis/option_routes_pilot2_20261001_r1'
EXTRA=['602-2','607-2']
CONDITIONS=('four_views','five_views')


def prepare(run):
    """接口：先冻结主清单和既有两例，不根据本轮命中或旧对错扩充案例。"""
    run.mkdir(parents=True,exist_ok=True);ids=read_json(FOUR/'protocol.json')['question_ids'];sources={};inputs={}
    for q in ids+EXTRA:
        paths=dict(pool=BASE/'pools'/(q+'.json'),features=BASE/'features'/(q+'.npy'),
                   scores=(TWO if q in EXTRA else FOUR)/'results'/(q+'.json'))
        if q not in EXTRA:paths['replay']=REPLAY/'questions'/(q+'.json')
        inputs[q]={k:str(v) for k,v in paths.items()}
        for p in paths.values():sources[str(p)]=sha256(p)
    proxy=FOUR/'witness_evaluation_protocol.json';sources[str(proxy)]=sha256(proxy)
    for name in ('src/videoqa_audit/joint_query_views.py','scripts/analyze_joint_query_views.py','src/videoqa_audit/option_coarse_replay.py',
                 'src/videoqa_methods/prefix_coverage_budget.py','src/videoqa_audit/option_rank_selection.py','src/videoqa_audit/query_input_probe.py'):
        sources[str(ROOT/name)]=sha256(ROOT/name)
    spec=dict(question_ids=ids,supplemental_ids=EXTRA,inputs=inputs,sources=sources,configuration=config(32,32),new_model_calls=0,
        scope='Same full coarse pool; preserve original-question anchors/support and prefix/DP; no refinement or QA. Supplemental cases excluded from main denominator.')
    if (run/'protocol.json').exists():assert read_json(run/'protocol.json')==spec
    else:durable(run/'protocol.json',spec)
    return spec


def worker(job):
    """接口：原题/四候选都用已有分数，只有种子集合可能变化。"""
    run,q=Path(job[0]),job[1];spec=read_json(run/'protocol.json');dest=run/'questions'/(q+'.json')
    if dest.exists():
        saved=read_json(dest);assert saved['protocol_sha256']==sha256(run/'protocol.json');return saved
    paths=spec['inputs'][q]
    for p in paths.values():assert sha256(p)==spec['sources'][p]
    pool=read_json(paths['pool']);n=pool['initial_count'];coarse=pool['candidates'][:n]
    features=np.load(paths['features'],allow_pickle=False)[:n].copy();scores=read_json(paths['scores'])['scores']
    four,_=select_option_ranks(coarse,scores);five,trace=select_joint_views(coarse,pool['initial_scores'],scores)
    # 步骤1：主集四路控制直接复用已完成的同粗池结果，验证种子身份一致。
    if 'replay' in paths:
        before=read_json(paths['replay'])['conditions']['option_seeds'];assert set(four)==set(before['plan']['seed_indices'])
    else:before=replay_coarse(coarse,features,pool['initial_scores'],four,pool['video'],spec['configuration'])
    after=replay_coarse(coarse,features,pool['initial_scores'],five,pool['video'],spec['configuration'])
    if set(four)==set(five):assert before['selected_indices']==after['selected_indices']
    # 步骤2：输出两阶段可达性，但不携带任何历史预测或正确率。
    output={}
    for name,value in zip(CONDITIONS,(before,after)):
        output[name]=dict(plan=value['plan'],selected_indices=value['selected_indices'],selected_frames=[coarse[i] for i in value['selected_indices']],
                          region_budgets=value['trace']['region_budgets'])
    result=dict(question_id=q,protocol_sha256=sha256(run/'protocol.json'),conditions=output,five_view_ranking=trace,
        seed_overlap=len(set(four)&set(five)),source_overlap=len(set(before['selected_indices'])&set(after['selected_indices'])),new_model_calls=0,qa_prediction=None)
    durable(dest,result);return result


def summarize(run,spec):
    """接口：固定75组主代理，两道已知定位例单列；不将未列画面判成无证据。"""
    records={q:read_json(run/'questions'/(q+'.json')) for q in spec['question_ids']+EXTRA}
    entries=read_json(FOUR/'witness_evaluation_protocol.json')['entries'];facts=[]
    for f in entries:
        r=records[f['question_id']];pts=set(f['witness_pts']);hits={}
        pool=read_json(spec['inputs'][f['question_id']]['pool'])
        for name,value in r['conditions'].items():
            hits[name]=dict(seed=bool(pts&{pool['candidates'][i]['source_pts'] for i in value['plan']['seed_indices']}),
                           selected=bool(pts&{x['source_pts'] for x in value['selected_frames']}))
        facts.append(dict(question_id=f['question_id'],fact_id=f['fact_id'],hits=hits))
    qids={f['question_id'] for f in facts};assert len(facts)==75 and len(qids)==42;metrics={}
    for name in CONDITIONS:
        metrics[name]={}
        for stage in ('seed','selected'):
            metrics[name][stage]=dict(retained=sum(f['hits'][name][stage] for f in facts),
                macro=sum(sum(f['hits'][name][stage] for f in facts if f['question_id']==q)/sum(f['question_id']==q for f in facts) for q in qids)/42)
    extra=[]
    # 步骤3：补充例检查原已定位精确候选、窗口及最终保留；均为部分见证而非完整答案。
    for q in EXTRA:
        pool=read_json(spec['inputs'][q]['pool']);r=records[q]
        for fact,ids,meaning in WITNESSES[q]:
            item=dict(question_id=q,fact_id=fact,meaning=meaning)
            for name,value in r['conditions'].items():
                windows=value['plan']['regions']
                item[name]=dict(seed=bool(set(ids)&set(value['plan']['seed_indices'])),selected=bool(set(ids)&set(value['selected_indices'])),
                    window=any(Fraction(w['left_fraction'])<=source_time(pool['candidates'][i],pool['video'])<=Fraction(w['right_fraction']) for w in windows for i in ids))
            extra.append(item)
    changes={stage:{kind:[(f['question_id'],f['fact_id']) for f in facts if (f['hits']['four_views'][stage],f['hits']['five_views'][stage])==pattern]
                    for kind,pattern in [('gained',(False,True)),('lost',(True,False))]} for stage in ('seed','selected')}
    summary=dict(status='completed',metrics=metrics,changes=changes,facts=facts,supplemental=extra,
        changed_seed_questions=[q for q in spec['question_ids'] if set(records[q]['conditions']['four_views']['plan']['seed_indices'])!=set(records[q]['conditions']['five_views']['plan']['seed_indices'])],
        changed_selection_questions=[q for q in spec['question_ids'] if records[q]['conditions']['four_views']['selected_indices']!=records[q]['conditions']['five_views']['selected_indices']],
        source_overlap=distribution([records[q]['source_overlap'] for q in spec['question_ids']]),new_model_calls=0,qa_accuracy_not_defined=True,utc=utc())
    durable(run/'summary.json',summary);print(metrics,changes,extra,flush=True)


def execute(run):
    """接口：独立后台CPU重放，不启动任何模型或修改已测版本。"""
    with exclusive(run/'coordinator.lock'):
        spec=prepare(run)
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as ex:
            futures=[ex.submit(worker,(str(run),q)) for q in spec['question_ids']+EXTRA]
            for i,f in enumerate(as_completed(futures),1):
                f.result();durable(run/'status.json',dict(status='running',completed=i,total=202,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        summarize(run,spec);durable(run/'status.json',dict(status='completed',completed=202,total=202,pid=os.getpid(),sid=os.getsid(0),utc=utc()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    import re
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id)
    execute(ROOT/'outputs/analysis'/args.run_id)
