#!/usr/bin/env python3
"""两项独立零调用诊断：严格选项结构、固定配额等权种子；禁止合并成新成绩。"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc,rows
from videoqa_audit.option_structure_support import option_structure,fixed_quota_support
from videoqa_audit.option_entry_attribution import nearest_similarities
from videoqa_audit.query_input_probe import FULL
from videoqa_runtime.baseline_report import distribution
NAME='rd12_exp08__qwen32__dev200__20261001_r1';SOURCE=ROOT/'outputs/methods'/NAME;QA=ROOT/'outputs/diagnostics'/NAME
PROXY=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'


def prepare(run):
    """接口：先只用原题/选项固定结构路由，再绑定旧预测用于事后组合统计。"""
    run.mkdir(parents=True,exist_ok=True);source_spec=read_json(SOURCE/'protocol.json')
    ids=source_spec['question_ids'];data={r['question_id']:r for r in rows()};routing={};sources={}
    # 步骤1：此处不加载任何方法答案；规则全200题同用，不排除不利条目。
    for q in ids:
        row=data[q];kind=option_structure(row['question'],row['options'])
        routing[q]=dict(question=row['question'],options=row['options'],structure=kind,
                        retrospective_source='RD-1.2-exp07' if kind!='other' else 'RD-1.2-exp08')
    rp=run/'structure_routes.json'
    if rp.exists():assert read_json(rp)==routing
    else:durable(rp,routing)
    # 步骤2：两个诊断共用冻结资产，但不能把支持消融的输入附用旧答案。
    for q in ids:
        for p in (SOURCE/'pools'/(q+'.json'),SOURCE/'features'/(q+'.npy'),SOURCE/'traces'/(q+'.json'),
                  SOURCE/'selections'/(q+'.json'),QA/'results/RD-1.2-exp08'/(q+'.json'),FULL/'imported/results/RD-1.2-exp07'/(q+'.json')):
            sources[str(p)]=sha256(p)
    for p in (PROXY,ROOT/'src/videoqa_audit/option_structure_support.py',ROOT/'scripts/analyze_option_structure_support.py',
              ROOT/'src/videoqa_methods/weighted_seed_support.py',ROOT/'src/videoqa_methods/coarse_eligibility.py'):
        sources[str(p)]=sha256(p)
    spec=dict(question_ids=ids,sources=sources,routing_sha256=sha256(rp),configuration=source_spec['configuration'],
        new_model_calls=0,independent_interventions=['strict_text_structure_retrospective_routing','equal_seed_support_fixed_exp08_quota'],
        interpretation='Post-hoc development diagnostics. Cached routing is not fresh inference. Equal support has no QA predictions.')
    pp=run/'protocol.json'
    if pp.exists():assert read_json(pp)==spec
    else:durable(pp,spec)
    return spec


def worker(job):
    """接口：先还原exp08，再固定配额取消分数权重，不改变区域/补查/种子/锚点。"""
    run,q=Path(job[0]),job[1];spec=read_json(run/'protocol.json');dest=run/'support'/(q+'.json')
    if dest.exists():
        saved=read_json(dest);assert saved['protocol_sha256']==sha256(run/'protocol.json');return saved
    pp=SOURCE/'pools'/(q+'.json');fp=SOURCE/'features'/(q+'.npy');tp=SOURCE/'traces'/(q+'.json');sp=SOURCE/'selections'/(q+'.json')
    for p in (pp,fp,tp,sp):assert sha256(p)==spec['sources'][str(p)]
    pool=read_json(pp);features=np.load(fp,allow_pickle=False);old=read_json(sp);quotas=read_json(tp)['region_budgets']
    original,_=fixed_quota_support(pool,features,quotas,spec['configuration'],equal=False)
    assert original==old['selected_indices'],('Exp08 local replay differs',q)
    # 步骤1：仅将支持权重变为1；不重算DP配额，不新评分或补查。
    selected,trace=fixed_quota_support(pool,features,quotas,spec['configuration'],equal=True)
    assert trace['region_budgets']==quotas and len(selected)==len(original)
    record=dict(question_id=q,protocol_sha256=sha256(run/'protocol.json'),original_replay=True,selected_indices=selected,
                selected_frames=[pool['candidates'][i] for i in selected],trace=trace,
                source_overlap=len(set(original)&set(selected)),nearest_cosine=nearest_similarities(features,selected),
                new_model_calls=0,qa_prediction=None)
    durable(dest,record);return record


def summarize(run,spec):
    """接口：路由估算和无问答的支持消融分开输出，不把两者收益相加。"""
    routing=read_json(run/'structure_routes.json');old={};new={};support={}
    for q in spec['question_ids']:
        old[q]=read_json(FULL/'imported/results/RD-1.2-exp07'/(q+'.json'))
        new[q]=read_json(QA/'results/RD-1.2-exp08'/(q+'.json'))
        assert old[q]['model']==new[q]['model'] and old[q]['question']==new[q]['question'] and old[q]['options']==new[q]['options']
        for key in ('source_frames','processed_size','visual_tokens','prefill_tokens','prompt','generation_kwargs'):
            assert old[q]['answer']['observed'][key]==new[q]['answer']['observed'][key],(q,key)
        support[q]=read_json(run/'support'/(q+'.json'))
    # 步骤1：结构规则只决定复用哪个完整旧输入/答案；结果明确为回顾性组合。
    buckets={}
    for kind in sorted({r['structure'] for r in routing.values()}):
        ids=[q for q,r in routing.items() if r['structure']==kind]
        buckets[kind]=dict(question_ids=ids,n=len(ids),exp07_correct=sum(old[q]['correct'] for q in ids),
            exp08_correct=sum(new[q]['correct'] for q in ids),exp07_only=[q for q in ids if old[q]['correct'] and not new[q]['correct']],
            exp08_only=[q for q in ids if new[q]['correct'] and not old[q]['correct']])
    combined=sum((old if r['retrospective_source']=='RD-1.2-exp07' else new)[q]['correct'] for q,r in routing.items())
    route_report=dict(status='completed',categories=buckets,retrospective_composed_correct=combined,n=200,new_model_calls=0,
        warning='Retrospective composition of exact frozen inputs/answers, NOT a newly measured algorithm score; rule derived after seeing examples.')
    durable(run/'structure_summary.json',route_report)
    # 步骤2：等权支持只评估固定目录和输入变化，绝不复用exp08答案算正确率。
    facts=read_json(PROXY)['entries'];evaluation=[]
    for f in facts:
        q=f['question_id'];pts=set(f['witness_pts'])
        evaluation.append(dict(question_id=q,fact_id=f['fact_id'],role=f['role'],
            original=bool(pts&{r['source_pts'] for r in new[q]['selection']['selected_frames']}),
            equal=bool(pts&{r['source_pts'] for r in support[q]['selected_frames']})))
    qids={x['question_id'] for x in evaluation};assert len(evaluation)==75 and len(qids)==42;metrics={}
    for name in ('original','equal'):
        metrics[name]=dict(retained=sum(f[name] for f in evaluation),macro=sum(sum(f[name] for f in evaluation if f['question_id']==q)/sum(f['question_id']==q for f in evaluation) for q in qids)/42)
    result=dict(status='completed',metrics=metrics,facts=evaluation,
        gained=[(f['question_id'],f['fact_id']) for f in evaluation if f['equal'] and not f['original']],
        lost=[(f['question_id'],f['fact_id']) for f in evaluation if f['original'] and not f['equal']],
        changed_questions=[q for q in support if support[q]['selected_frames']!=new[q]['selection']['selected_frames']],
        nearest_cosine_question_means=distribution([float(np.mean(x['nearest_cosine'])) for x in support.values()]),
        new_model_calls=0,qa_accuracy_not_defined=True,utc=utc())
    durable(run/'support_summary.json',result);print(route_report,flush=True);print({k:v for k,v in result.items() if k not in ('facts','changed_questions')},flush=True)


def execute(run):
    """接口：CPU后台执行两个分离诊断，已有结果按指纹恢复，不启动GPU模型。"""
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
