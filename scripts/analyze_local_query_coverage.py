#!/usr/bin/env python3
"""已评分池上的局部描述覆盖验证；只重放选择，禁止复用旧答案作为新成绩。"""
import argparse
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.query_input_probe import FULL
from videoqa_audit.option_structure_support import option_structure
from videoqa_budget32.selection import config
from videoqa_methods.option_seed_entry import select_option_pool
from videoqa_methods.local_query_coverage import select_local_query_coverage
FACTS=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'


def execute(run,source):
    """接口：事前绑定评分池和路由文本，重放、常量回退与选项置换均须精确通过。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'coordinator.lock'):
        prior=read_json(source/'summary.json')
        assert prior['status']=='completed' and prior['qa_calls']==0
        ids=read_json(source/'protocol.json')['question_ids']
        sources=dict(prior['artifacts'])
        sources[str(FACTS)]=sha256(FACTS)
        paths={}
        for q in ids:
            item=read_json(source/'inputs'/(q+'.json'))
            paths[q]=dict(pool=item['pool'],features=item['features'],scores=str(source/'results'/(q+'.json')),
                text_input=str(source/'inputs'/(q+'.json')),reference=str(FULL/'selections/RD-1.2-exp07'/(q+'.json')))
            for p in paths[q].values():sources[p]=sha256(p)
        for name in ['scripts/analyze_local_query_coverage.py','src/videoqa_methods/local_query_coverage.py',
                     'src/videoqa_audit/option_structure_support.py','src/videoqa_methods/prefix_coverage_budget.py']:
            sources[str(ROOT/name)]=sha256(ROOT/name)
        spec=dict(question_ids=ids,inputs=paths,sources=sources,configuration=config(32,32),
            hypothesis='Original question anchors; up to four marginal rank-coverage steps for local option descriptions; original visual completion and coverage DP; strict integer/reference syntax fallback',new_model_calls=0)
        pp=run/'protocol.json'
        if pp.exists():assert read_json(pp)==spec
        else:durable(pp,spec)
        for p,h in sources.items():assert sha256(p)==h,p
        if (run/'summary.json').exists():
            for p,h in read_json(run/'summary.json')['artifacts'].items():assert sha256(p)==h
            print('Completed recovery:0 model calls',flush=True)
            return
        records={}
        for q in ids:
            dest=run/'questions'/(q+'.json')
            if dest.exists():
                records[q]=read_json(dest)
                assert records[q]['protocol_sha256']==sha256(pp)
                continue
            item=read_json(paths[q]['text_input'])
            options=item['queries']['option']
            suffix='\n'+options[0]
            assert item['queries']['joined'][0].endswith(suffix)
            question=item['queries']['joined'][0][:-len(suffix)]
            assert item['queries']['joined']==[question+'\n'+o for o in options]
            route=option_structure(question,options)
            enabled=route=='other'
            pool=read_json(paths[q]['pool'])
            features=np.load(paths[q]['features'],allow_pickle=False)
            scores=np.asarray(read_json(paths[q]['scores'])['scores']['option'])
            # 步骤1：先复现完整exp07，避免将来源/实现差异解释成新规则收益。
            old,reference=select_option_pool(pool,features,spec['configuration'])
            assert old==read_json(paths[q]['reference'])['selected_indices']
            new,trace=select_local_query_coverage(pool,features,spec['configuration'],reference['region_budgets'],scores,enabled=enabled)
            null,null_trace=select_local_query_coverage(pool,features,spec['configuration'],reference['region_budgets'],np.zeros_like(scores))
            assert null==old and null_trace['objective']==reference['objective']
            permuted,_=select_local_query_coverage(pool,features,spec['configuration'],reference['region_budgets'],scores[[3,1,0,2]],enabled=enabled)
            assert new==permuted and set(reference['protected_indices'])<=set(new)
            # 步骤2：保存完整前缀/配额与新旧精确源帧，不携带任何新预测。
            result=dict(question_id=q,protocol_sha256=sha256(pp),route=route,enabled=enabled,
                old_indices=old,selected_indices=new,selected_frames=[pool['candidates'][i] for i in new],trace=trace,
                null_replay_equal=True,permutation_equal=True,new_model_calls=0)
            durable(dest,result)
            records[q]=result
        # 步骤3：固定目录只用于事后审核，未列细帧不判断为无关。
        facts=[]
        for fact in read_json(FACTS)['entries']:
            q=fact['question_id']
            if q not in records:continue
            r=records[q]
            pool=read_json(paths[q]['pool'])
            known=set(fact['witness_pts'])
            facts.append(dict(question_id=q,fact_id=fact['fact_id'],
                before=bool(known & {pool['candidates'][i]['source_pts'] for i in r['old_indices']}),
                after=bool(known & {f['source_pts'] for f in r['selected_frames']})))
        result=dict(status='completed',facts=facts,before=sum(f['before'] for f in facts),after=sum(f['after'] for f in facts),
            gains=[f for f in facts if f['after'] and not f['before']],losses=[f for f in facts if f['before'] and not f['after']],
            changed=[q for q,r in records.items() if r['old_indices']!=r['selected_indices']],
            fallback=[q for q,r in records.items() if not r['enabled']],new_model_calls=0,qa_calls=0,
            artifacts={str(run/'questions'/(q+'.json')):sha256(run/'questions'/(q+'.json')) for q in ids},utc=utc())
        durable(run/'summary.json',result)
        print({k:v for k,v in result.items() if k not in ['artifacts','facts']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',required=True)
    p.add_argument('--source-run',required=True)
    args=p.parse_args()
    import re
    assert all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',v) for v in [args.run_id,args.source_run])
    execute(ROOT/'outputs/analysis'/args.run_id,ROOT/'outputs/analysis'/args.source_run)
