#!/usr/bin/env python3
"""局部覆盖前缀的零问答诊断；先固定代表题，再扩展同一规则，不搜索系数。"""
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
from videoqa_methods.marginal_coverage_prefix import select_marginal_coverage_prefix
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
FACTS=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'
PILOT=['103-1','395-2','306-2','383-2','170-2','544-1','844-1','872-2','893-1','264-2','602-2','607-2']


def execute(run,pilot):
    """接口：固定同一粗细池和参考目标，仅改变局部序列；答案不参与选择。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'run.lock'):
        ids=PILOT if pilot else [r['question_id'] for r in read_json(DATA)['rows']]
        sources={str(DATA):sha256(DATA),str(FACTS):sha256(FACTS)};inputs={}
        for name in ('scripts/analyze_marginal_prefix.py','src/videoqa_methods/marginal_coverage_prefix.py','src/videoqa_methods/relevance_coverage_budget.py','src/videoqa_methods/prefix_coverage_budget.py'):
            p=ROOT/name;sources[str(p)]=sha256(p);dest=run/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha256(dest)==sources[str(p)]
            else:shutil.copy2(p,dest)
        # 步骤1：输入身份与固定名单先落盘，人工事实仅用于后续评价。
        for q in ids:
            paths=dict(pool=BASE/'pools'/(q+'.json'),features=BASE/'features'/(q+'.npy'),trace=FULL/'traces'/(q+'.json'),base=FULL/'selections/RD-1.2-exp07'/(q+'.json'))
            inputs[q]={k:str(v) for k,v in paths.items()}
            for p in paths.values():sources[str(p)]=sha256(p)
        spec=dict(question_ids=ids,inputs=inputs,sources=sources,new_model_calls=0,pilot=pilot,
            hypothesis='Only exp12 local prefix changed to original-question-weighted empirical coverage marginal; same references, density, protected anchors, exact DP and frozen pool.')
        pp=run/'protocol.json'
        if pp.exists():assert read_json(pp)==spec
        else:durable(pp,spec)
        records={}
        # 步骤2：完整前缀、边际贡献和最终预算均保存；成功恢复只读原条目。
        for n,q in enumerate(ids,1):
            dest=run/'questions'/(q+'.json')
            if dest.exists():
                r=read_json(dest);assert r['protocol_sha256']==sha256(pp);records[q]=r;continue
            p=inputs[q];pool=read_json(p['pool']);f=np.load(p['features'],allow_pickle=False)
            selected,trace=select_marginal_coverage_prefix(pool,f,read_json(p['trace']))
            r=dict(question_id=q,protocol_sha256=sha256(pp),old_indices=trace['parent_selected_indices'],base_indices=read_json(p['base'])['selected_indices'],
                selected_indices=selected,selected_frames=[pool['candidates'][i] for i in selected],trace=trace,new_model_calls=0,qa_prediction=None)
            durable(dest,r);records[q]=r;durable(run/'status.json',dict(status='running',completed=n,total=len(ids),pid=os.getpid(),utc=utc()))
        # 步骤3：全部已列粗见证逐项比较；未标注新帧不能自动判无关。
        facts=[]
        for fact in read_json(FACTS)['entries']:
            q=fact['question_id']
            if q not in records:continue
            pool=read_json(inputs[q]['pool']);r=records[q];w=set(fact['witness_pts'])
            hits={key:bool(w&{pool['candidates'][i]['source_pts'] for i in r[field]}) for key,field in [('parent','old_indices'),('exp07','base_indices'),('new','selected_indices')]}
            facts.append(dict(question_id=q,fact_id=fact['fact_id'],**hits))
        qids=sorted({f['question_id'] for f in facts});metrics={key:dict(retained=sum(f[key] for f in facts),macro=sum(sum(f[key] for f in facts if f['question_id']==q)/sum(f['question_id']==q for f in facts) for q in qids)/len(qids)) for key in ('parent','exp07','new')}
        if not pilot:assert len(facts)==75 and len(qids)==42
        summary=dict(status='completed',question_count=len(ids),metrics=metrics,facts=facts,new_model_calls=0,
            changed_questions=[q for q,r in records.items() if r['old_indices']!=r['selected_indices']],
            replacements=sum(len(set(r['old_indices'])-set(r['selected_indices'])) for r in records.values()),
            gained=[(f['question_id'],f['fact_id']) for f in facts if not f['parent'] and f['new']],lost=[(f['question_id'],f['fact_id']) for f in facts if f['parent'] and not f['new']])
        durable(run/'summary.json',summary);durable(run/'status.json',dict(status='completed',completed=len(ids),pid=os.getpid(),utc=utc()))
        print({k:v for k,v in summary.items() if k not in ('facts','changed_questions')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id)
    execute(ROOT/'outputs/analysis'/a.run_id,a.pilot)
