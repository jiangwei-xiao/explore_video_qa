#!/usr/bin/env python3
"""固定exp17前缀的等权预算离线对照；新模型调用为0，不自动问答。"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_methods.equal_region_budget import select_equal_region_budget
SOURCE=ROOT/'outputs/analysis/local_query_coverage200_20261003_r1'
AUDIT=ROOT/'outputs/methods/rd12_exp17__qwen32__dev200__20261003_r1/independent_input_audit.json'
FACTS=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'


def execute(run):
    """接口：先验证父版本完整输入，随后只重分配同一前缀族的预算，最后评价固定事实。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'coordinator.lock'):
        parent=read_json(SOURCE/'protocol.json')
        assert read_json(SOURCE/'summary.json')['status']=='completed' and read_json(AUDIT)['status']=='passed'
        sources=dict(parent['sources']);inputs=parent['inputs']
        for p in [SOURCE/'protocol.json',SOURCE/'summary.json',AUDIT,FACTS,Path(__file__),ROOT/'src/videoqa_methods/equal_region_budget.py']:
            sources[str(p)]=sha256(p)
        for q in parent['question_ids']:
            p=SOURCE/'questions'/(q+'.json');sources[str(p)]=sha256(p)
        spec=dict(question_ids=parent['question_ids'],inputs=inputs,sources=sources,configuration=parent['configuration'],
            hypothesis='Fixed exp17 local prefixes; equal regional coverage utility in DP; original density still used for upstream geometry/refinement; no new scores',new_model_calls=0)
        path=run/'protocol.json'
        if path.exists():assert read_json(path)==spec
        else:durable(path,spec)
        for p,h in sources.items():assert sha256(p)==h,p
        if (run/'summary.json').exists():
            for p,h in read_json(run/'summary.json')['artifacts'].items():assert sha256(p)==h
            print('Completed recovery:0 model calls',flush=True);return
        records={}
        # 步骤1：每题保持同一候选、原区域和前缀；不使用答案挑选预算。
        for q in spec['question_ids']:
            dest=run/'questions'/(q+'.json');old=read_json(SOURCE/'questions'/(q+'.json'));pool=read_json(inputs[q]['pool'])
            if dest.exists():
                records[q]=read_json(dest);assert records[q]['protocol_sha256']==sha256(path);continue
            selected,trace=select_equal_region_budget(pool['candidates'],old['trace'])
            assert all(a['prefix']==b['prefix'] and a['coverage_curve']==b['coverage_curve'] for a,b in zip(old['trace']['regions'],trace['regions']))
            if trace['exact_equal_weight_equivalence']:assert selected==old['selected_indices']
            result=dict(question_id=q,protocol_sha256=sha256(path),selected_indices=selected,
                selected_frames=[pool['candidates'][i] for i in selected],old_indices=old['selected_indices'],trace=trace,
                query_module_enabled=old['enabled'],route=old['route'],new_model_calls=0)
            durable(dest,result);records[q]=result
        # 步骤2：固定75项粗见证，不因细帧未被目录收录就称其无效。
        facts=[]
        for f in read_json(FACTS)['entries']:
            q=f['question_id'];r=records[q];pool=read_json(inputs[q]['pool']);pts=set(f['witness_pts'])
            facts.append(dict(question_id=q,fact_id=f['fact_id'],before=bool(pts & {pool['candidates'][i]['source_pts'] for i in r['old_indices']}),
                after=bool(pts & {c['source_pts'] for c in r['selected_frames']})))
        assert len(facts)==75 and len({f['question_id'] for f in facts})==42
        summary=dict(status='completed',facts=facts,before=sum(f['before'] for f in facts),after=sum(f['after'] for f in facts),
            gains=[f for f in facts if f['after'] and not f['before']],losses=[f for f in facts if f['before'] and not f['after']],
            changed=[q for q,r in records.items() if r['selected_indices']!=r['old_indices']],
            equal_weight_equivalent=[q for q,r in records.items() if r['trace']['exact_equal_weight_equivalence']],
            query_disabled_ids=[q for q,r in records.items() if not r['query_module_enabled']],qa_calls=0,new_model_calls=0,
            artifacts={str(run/'questions'/(q+'.json')):sha256(run/'questions'/(q+'.json')) for q in records},utc=utc())
        durable(run/'summary.json',summary)
        print({k:v for k,v in summary.items() if k not in ['facts','artifacts','changed','equal_weight_equivalent','query_disabled_ids']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);args=p.parse_args()
    import re
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id)
    execute(ROOT/'outputs/analysis'/args.run_id)
