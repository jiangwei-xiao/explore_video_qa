#!/usr/bin/env python3
"""只读分析200题新入口在exp07中的排序与配额位置；新增模型调用为0。"""
import argparse
from collections import Counter
from fractions import Fraction
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,utc,exclusive
from videoqa_audit.query_input_probe import BASE,FULL
from videoqa_audit.option_entry_attribution import prefix_gate,nearest_similarities
from videoqa_methods.density_selection import source_time
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_runtime.baseline_report import distribution


def analyze(run):
    """接口：冻结全部输入哈希后逐题核对，不筛掉退化或无变化问题。"""
    scoring=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1'
    qa=ROOT/'outputs/diagnostics'/scoring.name
    spec=read_json(qa/'protocol.json');run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'analysis.lock'):
        sources={};paths={}
        for q in spec['question_ids']:
            files=dict(pool=BASE/'pools'/(q+'.json'),features=BASE/'features'/(q+'.npy'),
                check=BASE/'selection_checkpoints'/(q+'.json'),trace=FULL/'traces'/(q+'.json'),
                old_export=FULL/'selections/RD-1.2-exp07'/(q+'.json'),new=qa/'results/RD-1.2-diag01'/(q+'.json'),
                old=FULL/'imported/results/RD-1.2-exp07'/(q+'.json'),topk=BASE/'results/BASE-BLIP-TopK'/(q+'.json'))
            paths[q]=files
            for p in files.values():sources[str(p)]=sha256(p)
        for name in ('src/videoqa_audit/option_entry_attribution.py','scripts/analyze_option_entry_attribution.py'):
            sources[str(ROOT/name)]=sha256(ROOT/name)
        protocol=dict(question_ids=spec['question_ids'],sources=sources,new_model_calls=0,
            limitations='New entry frames are not ground-truth evidence. Quota reachability is per-frame, not joint. CLS similarity is not semantic redundancy.')
        if (run/'protocol.json').exists():
            assert read_json(run/'protocol.json')==protocol
        else:durable(run/'protocol.json',protocol)
        records=[]
        # 步骤1：源身份映射回旧池，先核对旧轨迹可重建最终选帧。
        for q,files in paths.items():
            pool=read_json(files['pool']);features=np.load(files['features'],allow_pickle=False)
            check=read_json(files['check']);trace=read_json(files['trace']);old_export=read_json(files['old_export'])
            for p in (files['pool'],files['features']):assert sha256(p)==check['files'][str(p.relative_to(BASE))]
            assert sha256(files['trace'])==old_export['trace_sha256']
            old=read_json(files['old']);new=read_json(files['new']);topk=read_json(files['topk'])
            key=lambda f:(f['source_frame_index'],f['source_pts'])
            lookup={key(f):i for i,f in enumerate(pool['candidates'])}
            original=[lookup[key(f)] for f in old['selection']['selected_frames']]
            selected=[lookup[key(f)] for f in new['selection']['selected_frames']]
            baseline=[lookup[key(f)] for f in topk['selection']['selected_frames']]
            assert original==old_export['selected_indices']
            rebuilt={i for r in trace['regions'] for i in r['prefix'][:trace['region_budgets'][str(r['region_id'])]]}
            assert rebuilt==set(original) and all(i<pool['initial_count'] for i in selected)
            assert set(selected)<=set(trace['eligible_indices'])
            gates=[];budget=len(original);supports={}
            for r in trace['regions']:
                source=next(x for x in pool['plan']['regions'] if x['region_id']==r['region_id'])
                vector,_,_=weighted_support(features,r['members'],source['seed_indices'],pool['initial_scores'])
                supports.update(dict(zip(r['members'],map(float,vector))))
            # 步骤2：只追踪新对照相对exp07多选的帧，避免把已保留的见证再算找回。
            for i in selected:
                if i in rebuilt:continue
                gate=prefix_gate(i,trace,budget);frame=pool['candidates'][i];t=source_time(frame,pool['video'])
                inside=any(Fraction(r['left_fraction'])<=t<=Fraction(r['right_fraction']) for r in pool['plan']['regions'])
                sim=float(np.max(np.clip(features[original]@features[i],0,1)))
                gates.append(dict(candidate_index=i,source_frame=frame,**gate,inside_original_window=inside,
                                  original_seed_support=supports[i],max_cosine_to_old_selected=sim))
            record=dict(question_id=q,stratum=new['stratum'],new_correct=new['correct'],old_correct=old['correct'],
                topk_correct=topk['correct'],old_count=len(original),new_count=len(selected),overlap=len(set(original)&set(selected)),
                added=gates,nearest_cosine=dict(new=nearest_similarities(features,selected),
                    old=nearest_similarities(features,original),topk=nearest_similarities(features,baseline)))
            durable(run/'questions'/(q+'.json'),record);records.append(record)
        # 步骤3：全量和配对得失分组一起保存；不只选择改善题支撑新机制。
        groups={'all':records,'new_only_vs_exp07':[r for r in records if r['new_correct'] and not r['old_correct']],
                'exp07_only':[r for r in records if r['old_correct'] and not r['new_correct']],
                'both_correct':[r for r in records if r['old_correct'] and r['new_correct']],
                'both_wrong':[r for r in records if not r['old_correct'] and not r['new_correct']]}
        summary={}
        for name,items in groups.items():
            added=[f for r in items for f in r['added']]
            summary[name]=dict(question_count=len(items),added_positions=len(added),
                gate_counts=dict(Counter(f['status'] for f in added)),inside_old_windows=sum(f['inside_original_window'] for f in added),
                overlap=distribution([r['overlap'] for r in items]),
                nearest_cosine_question_means={m:distribution([float(np.mean(r['nearest_cosine'][m])) for r in items]) for m in ('new','old','topk')},
                original_support=distribution([f['original_seed_support'] for f in added]),
                similarity_to_old=distribution([f['max_cosine_to_old_selected'] for f in added]))
        durable(run/'summary.json',dict(status='completed',groups=summary,new_model_calls=0,utc=utc(),
                                      protocol_sha256=sha256(run/'protocol.json')))
        print(summary['all'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    import re
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id)
    analyze(ROOT/'outputs/analysis'/args.run_id)
