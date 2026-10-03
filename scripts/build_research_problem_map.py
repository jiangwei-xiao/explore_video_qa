#!/usr/bin/env python3
"""新阶段问题地图：连接既有证据与当前成绩，不重审全片、不调用模型或设计新选择器。"""
import argparse
import csv
from fractions import Fraction
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,rows,utc
from videoqa_audit.query_input_probe import BASE,FULL
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
AUDIT=ROOT/'outputs/analysis/evidence_videomme50_20260915_r1'
PROXY=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'
REFS={m:ROOT/'outputs/diagnostics'/name/'results'/m for m,name in [
    ('RD-1.2-exp12','rd12_exp12__qwen32__dev200__20261002_r1'),
    ('RD-1.2-exp17','rd12_exp17__qwen32__dev200__20261003_r1'),
    ('RD-1.2-exp18','rd12_exp18__qwen32__dev200__20261003_r1')]}


def build(run):
    """接口：冻结200题主范围及既有全量案例补充范围，保留未知，不自动推断错误原因。"""
    run.mkdir(parents=True,exist_ok=True)
    sources={}
    def read(path):
        path=Path(path);sources[str(path)]=sha256(path);return read_json(path)
    ids=[r['question_id'] for r in read(DATA)['rows']]
    assert len(ids)==len(set(ids))==200
    extras=[q for q in read(FULL/'case_manifest.json')['unique_ids'] if q not in ids]
    data={r['question_id']:r for r in rows()}
    entries=read(PROXY)['entries']
    findings={r['question_id']:r for r in read(AUDIT/'full_review/source_findings.json')['findings']}
    old_path=AUDIT/'final_review/diagnostics.csv';sources[str(old_path)]=sha256(old_path)
    with old_path.open(encoding='utf-8-sig',newline='') as stream:
        old_repairs={r['question_id']:r for r in csv.DictReader(stream)}
    video_hashes=read(BASE/'protocol.json')['video_hashes']
    records=[]
    # 步骤1：先连接当前真实答案。历史LLaVA16诊断不当作当前Qwen32因果验证。
    for q in ids+extras:
        info=data[q];result_paths={m:BASE/'results'/m/(q+'.json') for m in ['BASE-Uniform','BASE-BLIP-TopK','RD-1.2']}
        result_paths['RD-1.2-exp07']=FULL/('imported/results' if q in ids else 'results')/'RD-1.2-exp07'/(q+'.json')
        if q in ids:result_paths.update({m:p/(q+'.json') for m,p in REFS.items()})
        results={m:read(p) for m,p in result_paths.items()}
        for r in results.values():
            assert r['question']==info['question'] and r['options']==info['options']
            assert r['correct']==(r['answer']['parsed_answer']=='ABCD'[info['answer_index']])
        old_file=AUDIT/'final_review/questions'/(q+'.json')
        historical=read(old_file) if old_file.exists() else None
        if historical:
            assert historical['question']==info['question'] and historical['options']==info['options']
            assert historical['source_video_sha256']==video_hashes[info['video_id']],q
        repair=old_repairs.get(q)
        evidence={k:historical.get(k) for k in ['operation','operation_subtype','evidence_layout','required_evidence']} if historical else None
        # 步骤2：只计算已列源帧可达性，不将粗帧未命中解释为语义证据丢失。
        pool_path=BASE/'pools'/(q+'.json');trace_path=FULL/'traces'/(q+'.json')
        pool=read(pool_path);trace=read(trace_path);candidates=pool['candidates'];n=pool['initial_count']
        lookup={c['source_pts']:i for i,c in enumerate(candidates)}
        picked={f['source_pts'] for f in results['RD-1.2-exp07']['selection']['selected_frames']}
        tb=Fraction(pool['video']['time_base']);origin=pool['video']['start_pts']
        instant=lambda i:(candidates[i]['source_pts']-origin)*tb
        facts=[]
        for fact in [f for f in entries if f['question_id']==q]:
            witness_ids=[lookup[p] for p in fact['witness_pts'] if p in lookup]
            assert witness_ids and all(i<n for i in witness_ids)
            paths=[]
            for i in witness_ids:
                region=next(r for r in trace['regions'] if i in r['members'])
                position=region['prefix'].index(i)+1 if i in region['prefix'] else None
                cap=min(sum(trace['region_budgets'].values())-len(trace['regions'])+1,len(region['prefix']))
                paths.append(dict(candidate_index=i,source_pts=candidates[i]['source_pts'],
                    source_seconds=float(instant(i)),region_id=region['region_id'],prefix_position=position,
                    quota=trace['region_budgets'][str(region['region_id'])],max_feasible_quota=cap,
                    quota_only_reachable=position is not None and position<=cap))
            nearest=sorted(results['RD-1.2-exp07']['selection']['selected_frames'],
                key=lambda f:min(abs(f['source_pts']-candidates[i]['source_pts']) for i in witness_ids))[:2]
            facts.append(dict(fact_id=fact['fact_id'],role=fact['role'],description=fact['description'],
                listed_coarse_hit=bool(set(fact['witness_pts'])&picked),
                in_original_window=any(Fraction(r['left_fraction'])<=instant(i)<=Fraction(r['right_fraction']) for r in pool['plan']['regions'] for i in witness_ids),
                global_coarse_eligible=True,witness_paths=paths,nearest_selected_frames=nearest,
                semantic_sufficiency='not_inferred_from_exact_frame_match'))
        strong=['RD-1.2-exp07','RD-1.2-exp12','RD-1.2-exp17']
        state='supplemental_full_case' if q not in ids else ('all_three_references_wrong' if all(not results[m]['correct'] for m in strong) else 'reference_outcomes_mixed' if len({results[m]['correct'] for m in strong})>1 else 'all_three_references_correct')
        records.append(dict(question_id=q,scope='development_200' if q in ids else 'existing_full_case_supplement',
            question=info['question'],options=info['options'],stratum=info['stratum'],official_task=info.get('task_type'),
            reference_state=state,results={m:dict(correct=r['correct'],prediction=r['answer']['parsed_answer'],source=str(result_paths[m])) for m,r in results.items()},
            historical_requirements=evidence,historical_source_finding=findings.get(q),historical_llava16_repair=repair,
            repair_transfer_to_qwen32='unverified' if repair else 'not_available',
            known_coarse_witnesses=facts,current_phase_attribution='pending_evidence_review',
            data_issue='duplicate options/event labels; retain original scoring' if q in ['491-2','321-2'] else None))
    # 步骤3：主集、历史审核覆盖和补充案例分别计数；不外推50题原因分布到全量。
    primary=[r for r in records if r['scope']=='development_200']
    summary=dict(development_count=200,supplement_count=len(extras),
        reference_groups={state:[r['question_id'] for r in primary if r['reference_state']==state] for state in ['all_three_references_wrong','reference_outcomes_mixed','all_three_references_correct']},
        exp07_wrong=[r['question_id'] for r in primary if not r['results']['RD-1.2-exp07']['correct']],
        historical_requirements_count=sum(r['historical_requirements'] is not None for r in primary),
        legacy_diagnostic_questions=[r['question_id'] for r in primary if r['historical_llava16_repair']],
        new_model_calls=0,limitations='Historical requirements are not current-input sufficiency labels; exact coarse-frame hits omit alternative fine witnesses; supplements excluded from primary denominators')
    sources[str(Path(__file__))]=sha256(__file__)
    spec=dict(stage='problem_map_inventory',primary_ids=ids,supplemental_ids=extras,sources=sources,
        model_calls=0,new_sampling=0,new_algorithm_versions=0,utc=utc())
    if (run/'records.json').exists():raise RuntimeError('Use a new revision; do not overwrite prior inventory')
    durable(run/'protocol.json',spec);durable(run/'records.json',records);durable(run/'summary.json',summary)
    flat=[]
    for r in records:
        h=r['historical_requirements'] or {}
        row=dict(question_id=r['question_id'],scope=r['scope'],reference_state=r['reference_state'],question=r['question'],
            operation=h.get('operation',''),evidence_layout=h.get('evidence_layout',''),required_evidence=h.get('required_evidence',''),
            old_source_observation=(r['historical_source_finding'] or {}).get('observation',''),
            old_diagnostic_transfer=r['repair_transfer_to_qwen32'],known_coarse_facts=len(r['known_coarse_witnesses']),
            exp07_exact_coarse_hits=sum(f['listed_coarse_hit'] for f in r['known_coarse_witnesses']),
            current_attribution=r['current_phase_attribution'],data_issue=r['data_issue'] or '')
        for m in ['BASE-Uniform','BASE-BLIP-TopK','RD-1.2','RD-1.2-exp07',*REFS]:
            row[m+'_correct']=r['results'].get(m,{}).get('correct','')
        flat.append(row)
    with (run/'questions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print({k:v for k,v in summary.items() if k not in ['reference_groups','exp07_wrong']},flush=True)
    print({k:len(v) for k,v in summary['reference_groups'].items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    import re
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id)
    build(ROOT/'outputs/analysis'/args.run_id)
