#!/usr/bin/env python3
"""固定exp07前缀的原问题加权覆盖预算离线对照，零模型调用。"""
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
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
FACTS=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'


def execute(run):
    """接口：冻结一个无新系数的预算对照，保持窗口、成员、前缀及实际总预算。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'run.lock'):
        ids=[r['question_id'] for r in read_json(DATA)['rows']];sources={str(DATA):sha256(DATA),str(FACTS):sha256(FACTS)};inputs={}
        for name in ('scripts/analyze_relevance_coverage.py','src/videoqa_methods/relevance_coverage_budget.py','src/videoqa_methods/prefix_coverage_budget.py'):
            sources[str(ROOT/name)]=sha256(ROOT/name);dest=run/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha256(dest)==sources[str(ROOT/name)]
            else:shutil.copy2(ROOT/name,dest)
        # 步骤1：源数据身份先冻结；人工事实只供评价，不能参与权重或预算计算。
        for q in ids:
            paths=dict(pool=BASE/'pools'/(q+'.json'),features=BASE/'features'/(q+'.npy'),trace=FULL/'traces'/(q+'.json'),selection=FULL/'selections/RD-1.2-exp07'/(q+'.json'))
            inputs[q]={k:str(v) for k,v in paths.items()}
            for p in paths.values():sources[str(p)]=sha256(p)
        protocol=dict(question_ids=ids,sources=sources,inputs=inputs,new_model_calls=0,
            hypothesis='Only coarse coverage reference weights change from uniform to original-question ITM normalized within each region; all-zero uses uniform; no exponent/threshold search.')
        pp=run/'protocol.json'
        if pp.exists():assert read_json(pp)==protocol
        else:durable(pp,protocol)
        records={}
        # 步骤2：每题输出同一前缀族的旧新配额，不导入旧预测为新条件答案。
        for pos,q in enumerate(ids):
            dest=run/'questions'/(q+'.json')
            if dest.exists():
                r=read_json(dest);assert r['protocol_sha256']==sha256(pp);records[q]=r;continue
            paths=inputs[q];pool=read_json(paths['pool']);f=np.load(paths['features'],allow_pickle=False);old=read_json(paths['trace'])
            selected,trace=select_relevance_coverage(pool,f,old);before=read_json(paths['selection'])['selected_indices']
            assert len(selected)==len(before)
            r=dict(question_id=q,protocol_sha256=sha256(pp),old_indices=before,selected_indices=selected,
                selected_frames=[pool['candidates'][i] for i in selected],trace=trace,new_model_calls=0,qa_prediction=None)
            durable(dest,r);records[q]=r;durable(run/'status.json',dict(status='running',completed=pos+1,total=200,pid=os.getpid(),utc=utc()))
        # 步骤3：完整记录固定目录得失和变化规模；指标不是语义充分性或正确率。
        facts=[]
        for fact in read_json(FACTS)['entries']:
            q=fact['question_id'];r=records[q];pool=read_json(inputs[q]['pool']);w=set(fact['witness_pts'])
            hits={label:bool(w&{pool['candidates'][i]['source_pts'] for i in r[field]}) for label,field in [('old','old_indices'),('new','selected_indices')]}
            facts.append(dict(question_id=q,fact_id=fact['fact_id'],**hits))
        qids=sorted({f['question_id'] for f in facts});assert len(facts)==75 and len(qids)==42
        metrics={label:dict(retained=sum(f[label] for f in facts),macro=sum(sum(f[label] for f in facts if f['question_id']==q)/sum(f['question_id']==q for f in facts) for q in qids)/42) for label in ('old','new')}
        summary=dict(status='completed',question_count=200,metrics=metrics,facts=facts,
            changed_questions=[q for q,r in records.items() if r['old_indices']!=r['selected_indices']],
            replaced_frames=sum(len(set(r['old_indices'])-set(r['selected_indices'])) for r in records.values()),
            gained=[(f['question_id'],f['fact_id']) for f in facts if not f['old'] and f['new']],
            lost=[(f['question_id'],f['fact_id']) for f in facts if f['old'] and not f['new']],new_model_calls=0)
        durable(run/'summary.json',summary);durable(run/'status.json',dict(status='completed',completed=200,total=200,pid=os.getpid(),utc=utc()))
        print({k:v for k,v in summary.items() if k not in ('facts','changed_questions')},'changed',len(summary['changed_questions']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id)
    execute(ROOT/'outputs/analysis'/a.run_id)
