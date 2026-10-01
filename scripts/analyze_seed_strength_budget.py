#!/usr/bin/env python3
"""区域种子检索强度的单一预算对照，零模型调用且不改变exp12。"""
import argparse
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import re
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.query_input_probe import BASE,FULL
from videoqa_methods.seed_strength_budget import select_seed_strength_budget
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
FACTS=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'
PARENT=ROOT/'outputs/analysis/relevance_coverage_dev200_20261002_r1'
PARENT_QA=ROOT/'outputs/diagnostics/rd12_exp12__qwen32__dev200__20261002_r1'


def execute(run):
    """接口：冻结跨区域强度缩放，不搜索平均/最大值或指数；全部200题统一执行。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'run.lock'):
        ids=[r['question_id'] for r in read_json(DATA)['rows']];sources={str(DATA):sha256(DATA),str(FACTS):sha256(FACTS)};inputs={}
        for name in ('scripts/analyze_seed_strength_budget.py','src/videoqa_methods/seed_strength_budget.py','src/videoqa_methods/relevance_coverage_budget.py','src/videoqa_methods/prefix_coverage_budget.py'):
            p=ROOT/name;sources[str(p)]=sha256(p);dest=run/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha256(dest)==sources[str(p)]
            else:shutil.copy2(p,dest)
        # 步骤1：parent仅用于离散重放验收和后续配对，答案不传给选择函数。
        for q in ids:
            paths=dict(pool=BASE/'pools'/(q+'.json'),features=BASE/'features'/(q+'.npy'),trace=FULL/'traces'/(q+'.json'),
                base=FULL/'selections/RD-1.2-exp07'/(q+'.json'),parent=PARENT/'questions'/(q+'.json'),parent_answer=PARENT_QA/'results/RD-1.2-exp12'/(q+'.json'))
            inputs[q]={k:str(v) for k,v in paths.items()}
            for p in paths.values():sources[str(p)]=sha256(p)
        protocol=dict(question_ids=ids,inputs=inputs,sources=sources,new_model_calls=0,parent_method='RD-1.2-exp12',
            hypothesis='Only exp12 regional utility multiplied by mean seed original ITM / question maximum regional mean; all original prefixes and protected anchors unchanged; no threshold or exponent.')
        pp=run/'protocol.json'
        if pp.exists():assert read_json(pp)==protocol
        else:durable(pp,protocol)
        records={}
        # 步骤2：同资产计算父版本必须精确复现，再接受新配额；不覆盖父版本文件。
        for n,q in enumerate(ids,1):
            dest=run/'questions'/(q+'.json')
            if dest.exists():
                r=read_json(dest);assert r['protocol_sha256']==sha256(pp);records[q]=r;continue
            p=inputs[q];pool=read_json(p['pool']);f=np.load(p['features'],allow_pickle=False);reference=read_json(p['trace'])
            selected,trace=select_seed_strength_budget(pool,f,reference);parent=read_json(p['parent'])
            assert trace['parent_selected_indices']==parent['selected_indices'] and trace['parent_region_budgets']==parent['trace']['region_budgets']
            r=dict(question_id=q,protocol_sha256=sha256(pp),old_indices=parent['selected_indices'],base_indices=read_json(p['base'])['selected_indices'],
                selected_indices=selected,selected_frames=[pool['candidates'][i] for i in selected],trace=trace,new_model_calls=0,qa_prediction=None)
            durable(dest,r);records[q]=r;durable(run/'status.json',dict(status='running',completed=n,total=200,pid=os.getpid(),utc=utc()))
        # 步骤3：固定事实分母，父版本和exp07两种参照均报告，不附用旧预测。
        facts=[]
        for fact in read_json(FACTS)['entries']:
            q=fact['question_id'];pool=read_json(inputs[q]['pool']);r=records[q];w=set(fact['witness_pts'])
            hits={key:bool(w&{pool['candidates'][i]['source_pts'] for i in r[field]}) for key,field in [('parent','old_indices'),('exp07','base_indices'),('new','selected_indices')]}
            facts.append(dict(question_id=q,fact_id=fact['fact_id'],**hits))
        qids=sorted({f['question_id'] for f in facts});metrics={key:dict(retained=sum(f[key] for f in facts),macro=sum(sum(f[key] for f in facts if f['question_id']==q)/sum(f['question_id']==q for f in facts) for q in qids)/42) for key in ('parent','exp07','new')}
        summary=dict(status='completed',question_count=200,metrics=metrics,facts=facts,new_model_calls=0,
            changed_questions=[q for q,r in records.items() if r['old_indices']!=r['selected_indices']],
            replacements=sum(len(set(r['old_indices'])-set(r['selected_indices'])) for r in records.values()),
            gained=[(f['question_id'],f['fact_id']) for f in facts if not f['parent'] and f['new']],
            lost=[(f['question_id'],f['fact_id']) for f in facts if f['parent'] and not f['new']])
        assert len(facts)==75 and len(qids)==42
        durable(run/'summary.json',summary);durable(run/'status.json',dict(status='completed',completed=200,pid=os.getpid(),utc=utc()))
        print({k:v for k,v in summary.items() if k not in ('facts','changed_questions')},'changed',len(summary['changed_questions']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id)
    execute(ROOT/'outputs/analysis'/a.run_id)
