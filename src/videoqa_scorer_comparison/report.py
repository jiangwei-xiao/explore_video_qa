"""评分器全量统计：两工作点同协议配对、视频簇区间与真实物理成本。"""
from pathlib import Path
import itertools
from videoqa_runtime.common import ROOT,read_json
from videoqa_full.state import rows,durable,utc
from videoqa_runtime.baseline_report import wilson,distribution
from videoqa_full.report import compare
from videoqa_audit.region_scope_runtime import parent_result
from .preparation import PROFILES,METHODS
from .runtime import audit,validate_result


def summarize(root):
    """接口：六组全部完整才出正式结果，均匀仅引用，无问答重新生成。"""
    root=Path(root);data=rows();dev=read_json(ROOT/'data/manifests/video_mme_development_200_20260920.json')['rows']
    from .analysis import selection_analysis
    mechanisms=selection_analysis(root)
    devq={r['question_id'] for r in dev};devv={r['video_id'] for r in dev};profiles={}
    for profile,k in PROFILES.items():
        run=root/'qa'/profile;check=audit(run);spec=read_json(run/'protocol.json')
        old={m:{r['question_id']:read_json(parent_result(profile,r['question_id'],m)) for r in data}
            for m in ('BASE-Uniform',)+METHODS}
        new={m:{r['question_id']:validate_result(run,r,m,spec) for r in data} for m in METHODS}
        new['BASE-Uniform']=old['BASE-Uniform']
        common=[r for r in data if all(len(bank[r['question_id']]['selection']['selected_frames'])==k
            for banks in (old,new) for bank in banks.values())]
        subsets=dict(all2700=data,development200=[r for r in data if r['question_id'] in devq],
            other2500=[r for r in data if r['question_id'] not in devq],
            outside_development_videos2100=[r for r in data if r['video_id'] not in devv],common_full_budget=common)
        scores={};comparisons={};perf={}
        for sid,banks in (('S1',old),('S2',new)):
            scores[sid]={}
            for method,bank in banks.items():
                scores[sid][method]={}
                for name,subset in subsets.items():
                    layers={}
                    for layer in ('all','short','medium','long'):
                        items=[bank[r['question_id']] for r in subset if layer=='all' or r['stratum']==layer]
                        if not items:continue
                        c=sum(x['correct'] for x in items)
                        layers[layer]=dict(n=len(items),correct=c,accuracy=c/len(items),wilson_95=wilson(c,len(items)))
                    scores[sid][method][name]=layers
        # 步骤1：各方法前端效应与S2内算法效应分别配对，不能把两个差值混称算法收益。
        pairs=[('S2_minus_S1__'+m,new[m],old[m]) for m in METHODS]
        pairs += [('S2__'+m+'_minus_'+b,new[m],new[b]) for m in METHODS[1:] for b in ('BASE-Uniform',METHODS[0])]
        for name,left,right in pairs:
            comparisons[name]={n:compare(rs,left,right) for n,rs in subsets.items() if rs}
        for m in METHODS:
            physical=[x for x in new[m].values() if x['timing_kind']=='physical_primary']
            perf[m]=dict(physical_calls=len(physical),qa_seconds=distribution([x['answer']['observed']['qa_seconds'] for x in physical]) if physical else None,
                actual_frame_counts={str(n):sum(len(x['selection']['selected_frames'])==n for x in new[m].values())
                    for n in sorted({len(x['selection']['selected_frames']) for x in new[m].values()})})
        # 步骤2：官方题型按真实字段统计；没有类型字段的运行明确留空，不猜分类。
        types={}
        for key in ('task_type','sub_category','category'):
            if any(key in r for r in data):
                types[key]={str(t):{m:dict(n=sum(r.get(key)==t for r in data),correct=sum(bank[r['question_id']]['correct'] for r in data if r.get(key)==t))
                    for m,bank in new.items()} for t in sorted({r.get(key) for r in data if r.get(key) is not None})}
        profiles[profile]=dict(scores=scores,paired=comparisons,official_types=types,performance=perf,
            call_audit={key:value for key,value in check.items()},uniform_new_qa=0,
            parse_failures={m:[q for q,x in bank.items() if x['answer']['parsed_answer'] is None] for m,bank in new.items()},
            uncertainty='Observed full and repeatedly used development; exploratory multiple comparisons')
    value=dict(status='completed',profiles=profiles,scorer='S2-BLIP-2-ITM',
        intervention='ITM scores plus pure visual CLS replaced together; mixed diagnostic has zero QA',mechanisms=mechanisms,
        cost_records=dict(stage_wall={p.stem:read_json(p) for p in (root/'stage_completion').glob('*.json')},
            worker_totals=[read_json(p) for p in (root/'worker_totals').glob('*.json')],
            failures=[read_json(p) for p in (root/'failures').glob('*.json')]),utc=utc())
    durable(root/'comparison.json',value)
    return value
