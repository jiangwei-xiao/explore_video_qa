#!/usr/bin/env python3
"""固定200题零模型缺失侧补齐对照；保持exp07窗口、池与配额，输出全量得失。"""
import argparse
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.query_input_probe import BASE,FULL
from videoqa_audit.local_evidence_groups import complete_anchor_sides
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
FACTS=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'


def execute(run):
    """接口：先冻结规则与来源，再执行一次离线版本；完成恢复不能改写问题级结果。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'comparison.lock'):
        ids=[r['question_id'] for r in read_json(DATA)['rows']];assert len(ids)==200
        sources={str(DATA):sha256(DATA),str(FACTS):sha256(FACTS)};inputs={}
        for name in ('src/videoqa_audit/local_evidence_groups.py','scripts/analyze_local_evidence_groups.py',
                     'src/videoqa_methods/weighted_seed_support.py','src/videoqa_methods/density_selection.py'):
            sources[str(ROOT/name)]=sha256(ROOT/name)
        # 步骤1：人工事实只绑定评价来源，不传给选择函数。
        for q in ids:
            paths=dict(pool=BASE/'pools'/(q+'.json'),features=BASE/'features'/(q+'.npy'),trace=FULL/'traces'/(q+'.json'),
                       selection=FULL/'selections/RD-1.2-exp07'/(q+'.json'))
            inputs[q]={k:str(v) for k,v in paths.items()}
            for p in paths.values():sources[str(p)]=sha256(p)
        protocol=dict(base_method='RD-1.2-exp07',question_ids=ids,sources=sources,inputs=inputs,new_model_calls=0,
            rule='fixed quotas; only refine-enabled regions with quota>=3 and both sides available; repair missing side(s) once; keep anchor and sole side witness; remove last eligible old prefix item; original L*sqrt(u) rank within missing side',
            scope='No new frames, no QA, no semantic classifier, no optimization of window size or budget objective')
        pp=run/'comparison_protocol.json'
        if pp.exists():assert read_json(pp)==protocol
        else:durable(pp,protocol)
        records={}
        # 步骤2：逐题核对旧选择可由配额前缀重建，局部补齐后仍保持总预算与源身份。
        for pos,q in enumerate(ids):
            dest=run/'comparison_questions'/(q+'.json')
            if dest.exists():
                record=read_json(dest);assert record['protocol_sha256']==sha256(pp);records[q]=record;continue
            paths=inputs[q];pool=read_json(paths['pool']);features=np.load(paths['features'],allow_pickle=False);trace=read_json(paths['trace'])
            old=read_json(paths['selection'])['selected_indices']
            assert set(old)=={i for r in trace['regions'] for i in r['prefix'][:trace['region_budgets'][str(r['region_id'])]]}
            selected,changes=complete_anchor_sides(pool,features,trace)
            assert len(selected)==len(old) and changes['region_budgets']==trace['region_budgets']
            record=dict(question_id=q,protocol_sha256=sha256(pp),old_indices=old,selected_indices=selected,
                selected_frames=[pool['candidates'][i] for i in selected],trace=changes,new_model_calls=0,qa_prediction=None)
            durable(dest,record);records[q]=record
            durable(run/'comparison_status.json',dict(status='running',completed=pos+1,total=200,utc=utc()))
        # 步骤3：完整列出固定75项见证的增减及分母，不挑选只改善的题目。
        facts=[]
        for fact in read_json(FACTS)['entries']:
            record=records[fact['question_id']];pool=read_json(inputs[fact['question_id']]['pool']);witness=set(fact['witness_pts'])
            hits={key:bool(witness&{pool['candidates'][i]['source_pts'] for i in record[field]})
                  for key,field in [('old','old_indices'),('new','selected_indices')]}
            facts.append(dict(question_id=fact['question_id'],fact_id=fact['fact_id'],**hits))
        qids={f['question_id'] for f in facts};assert len(facts)==75 and len(qids)==42
        metrics={name:dict(retained=sum(f[name] for f in facts),macro=sum(sum(f[name] for f in facts if f['question_id']==q)/sum(f['question_id']==q for f in facts) for q in qids)/len(qids)) for name in ('old','new')}
        summary=dict(status='completed',question_count=200,metrics=metrics,facts=facts,
            changed_questions=[q for q,r in records.items() if r['old_indices']!=r['selected_indices']],
            replacements=sum(r['trace']['replacement_count'] for r in records.values()),
            changed_regions=sum(bool(g['changes']) for r in records.values() for g in r['trace']['regions']),
            gained=[(f['question_id'],f['fact_id']) for f in facts if not f['old'] and f['new']],
            lost=[(f['question_id'],f['fact_id']) for f in facts if f['old'] and not f['new']],
            new_model_calls=0,limitation='Exact nonexhaustive witnesses; side coverage is not semantic completeness or QA accuracy.')
        durable(run/'comparison_summary.json',summary);durable(run/'comparison_status.json',dict(status='completed',completed=200,total=200,utc=utc()))
        print({k:v for k,v in summary.items() if k not in ('facts','changed_questions')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id)
    execute(ROOT/'outputs/analysis'/a.run_id)
