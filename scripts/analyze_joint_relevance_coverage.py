#!/usr/bin/env python3
"""定位与预算交互检查：在exp09冻结池上复用exp12公式，无新评分或补查。"""
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
PARENT=ROOT/'outputs/methods/rd12_exp09__qwen32__dev200__20261001_r1'
PARENT_QA=ROOT/'outputs/diagnostics/rd12_exp09__qwen32__dev200__20261001_r1'
EXP12=ROOT/'outputs/diagnostics/rd12_exp12__qwen32__dev200__20261002_r1'


def execute(run):
    """接口：保持exp09入口及局部顺序，仅切换为已冻结exp12预算参考权重。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'run.lock'):
        ids=[r['question_id'] for r in read_json(DATA)['rows']];sources={str(DATA):sha256(DATA),str(FACTS):sha256(FACTS)};inputs={}
        for name in ('scripts/analyze_joint_relevance_coverage.py','src/videoqa_methods/relevance_coverage_budget.py','src/videoqa_methods/prefix_coverage_budget.py'):
            p=ROOT/name;sources[str(p)]=sha256(p);dest=run/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha256(dest)==sources[str(p)]
            else:shutil.copy2(p,dest)
        # 步骤1：先冻结全量开发名单和两项因素的旧答案来源，答案不传给选择器。
        for q in ids:
            paths=dict(pool=PARENT/'pools'/(q+'.json'),features=PARENT/'features'/(q+'.npy'),trace=PARENT/'traces'/(q+'.json'),
                base=FULL/'selections/RD-1.2-exp07'/(q+'.json'),parent=PARENT/'selections'/(q+'.json'),original_pool=BASE/'pools'/(q+'.json'),
                parent_answer=PARENT_QA/'results/RD-1.2-exp09'/(q+'.json'),exp12_answer=EXP12/'results/RD-1.2-exp12'/(q+'.json'))
            inputs[q]={k:str(v) for k,v in paths.items()}
            for p in paths.values():sources[str(p)]=sha256(p)
        spec=dict(question_ids=ids,inputs=inputs,sources=sources,new_model_calls=0,parent_method='RD-1.2-exp09',
            hypothesis='Fixed exp09 five-view seeds, windows, fine pool and local prefixes; only original-question-weighted reference coverage from exp12; no exp13 strength multiplier.')
        pp=run/'protocol.json'
        if pp.exists():assert read_json(pp)==spec
        else:durable(pp,spec)
        records={}
        # 步骤2：原问题ITM必须与exp07原资产完全相同，不能误用选项名次作覆盖权重。
        for n,q in enumerate(ids,1):
            dest=run/'questions'/(q+'.json')
            if dest.exists():
                r=read_json(dest);assert r['protocol_sha256']==sha256(pp);records[q]=r;continue
            p=inputs[q];pool=read_json(p['pool']);original=read_json(p['original_pool']);f=np.load(p['features'],allow_pickle=False);t=read_json(p['trace'])
            assert pool['initial_count']==original['initial_count'] and pool['initial_scores']==original['initial_scores']
            assert pool['candidates'][:pool['initial_count']]==original['candidates'][:original['initial_count']]
            before=read_json(p['parent'])['selected_indices']
            rebuilt={i for x in t['regions'] for i in x['prefix'][:t['region_budgets'][str(x['region_id'])]]};assert rebuilt==set(before)
            selected,trace=select_relevance_coverage(pool,f,t)
            r=dict(question_id=q,protocol_sha256=sha256(pp),old_indices=before,selected_indices=selected,
                selected_frames=[pool['candidates'][i] for i in selected],trace=trace,new_model_calls=0,qa_prediction=None)
            durable(dest,r);records[q]=r;durable(run/'status.json',dict(status='running',completed=n,total=200,pid=os.getpid(),utc=utc()))
        # 步骤3：跨池仅以精确源PTS匹配已列事实，不能直接比较新旧候选编号。
        facts=[]
        for fact in read_json(FACTS)['entries']:
            q=fact['question_id'];pool=read_json(inputs[q]['pool']);r=records[q];w=set(fact['witness_pts'])
            hits={key:bool(w&{pool['candidates'][i]['source_pts'] for i in r[field]}) for key,field in [('parent','old_indices'),('new','selected_indices')]}
            facts.append(dict(question_id=q,fact_id=fact['fact_id'],**hits))
        qids=sorted({f['question_id'] for f in facts});assert len(facts)==75 and len(qids)==42
        metrics={key:dict(retained=sum(f[key] for f in facts),macro=sum(sum(f[key] for f in facts if f['question_id']==q)/sum(f['question_id']==q for f in facts) for q in qids)/42) for key in ('parent','new')}
        summary=dict(status='completed',question_count=200,metrics=metrics,facts=facts,new_model_calls=0,
            changed_questions=[q for q,r in records.items() if r['old_indices']!=r['selected_indices']],
            replacements=sum(len(set(r['old_indices'])-set(r['selected_indices'])) for r in records.values()),
            gained=[(f['question_id'],f['fact_id']) for f in facts if not f['parent'] and f['new']],lost=[(f['question_id'],f['fact_id']) for f in facts if f['parent'] and not f['new']])
        durable(run/'summary.json',summary);durable(run/'status.json',dict(status='completed',completed=200,pid=os.getpid(),utc=utc()))
        print({k:v for k,v in summary.items() if k not in ('facts','changed_questions')},'changed',len(summary['changed_questions']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id)
    execute(ROOT/'outputs/analysis'/a.run_id)
