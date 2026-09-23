"""同协议均匀/Top-K/RD三组配对统计，历史答案仅作协议变化对照。"""
import csv
from collections import Counter
from pathlib import Path
from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import rows, durable, utc
from videoqa_full.report import compare
from videoqa_runtime.baseline_report import wilson, distribution
from videoqa_runtime.protocol import parse_answer
from .methods import METHODS, UNIFORM, SOURCE, validate


def summarize(run):
    """接口：两组2700答案齐全才汇总；同协议主比较与历史协议比较分开。"""
    run=Path(run); spec=read_json(run/'protocol.json'); data=rows(); ids={r['question_id'] for r in data}
    assert sha256(UNIFORM/'summary.json')==spec['uniform_summary_sha256']
    groups={'BASE-Uniform': {r['question_id']:read_json(UNIFORM/'results'/(r['question_id']+'.json')) for r in data}}
    # 步骤1：核验2700一一配对、每组固定调用ID与同题同GPU，不掩盖缺失。
    for method in METHODS:
        for folder in ('results','calls'):
            paths=list((run/folder/method).glob('*.json')); assert len(paths)==2700 and {p.stem for p in paths}==ids
        groups[method]={r['question_id']:validate(run,r,method,spec) for r in data}
    assert all(groups[METHODS[0]][q]['gpu']==groups[METHODS[1]][q]['gpu'] for q in ids)
    dev=read_json(ROOT/'data/manifests/video_mme_development_200_20260920.json')['rows']
    devq={r['question_id'] for r in dev}; devv={r['video_id'] for r in dev}
    boundary={q for q in ids if any(len(g[q]['selection']['selected_frames'])!=16 for g in groups.values())}
    subsets={'all2700':data,'development200':[r for r in data if r['question_id'] in devq],
        'other2500':[r for r in data if r['question_id'] not in devq],
        'outside_development_videos2100':[r for r in data if r['video_id'] not in devv],
        'all_methods_16_frames':[r for r in data if r['question_id'] not in boundary]}
    accuracy={}; performance={}; issues={}; flat=[]
    for method,group in groups.items():
        accuracy[method]={}
        for subset,items in subsets.items():
            accuracy[method][subset]={}
            for layer in ('all','short','medium','long'):
                chosen=[group[r['question_id']] for r in items if layer=='all' or r['stratum']==layer]
                if not chosen:continue
                c=sum(r['correct'] for r in chosen)
                accuracy[method][subset][layer]=dict(n=len(chosen),correct=c,accuracy=c/len(chosen),wilson_95=wilson(c,len(chosen)))
        issues[method]=dict(parse_failures=[q for q,r in group.items() if not r['result']['answer']['parsed_answer']],
            limit_hits=[q for q,r in group.items() if len(r['result']['observed']['generated_token_ids'])==16],
            strict_parser_differences=[q for q,r in group.items() if parse_answer(r['result']['answer']['raw_output'])!=r['result']['answer']['parsed_answer']],
            recovered=[q for q,r in group.items() if r['reused_answer']],
            frame_counts=dict(Counter(len(r['selection']['selected_frames']) for r in group.values())))
        performance[method]={}
        for layer in ('all','short','medium','long'):
            values=[r for r in group.values() if not r['reused_answer'] and (layer=='all' or r['stratum']==layer)]
            performance[method][layer]=dict(timings={k:distribution([r['timings'][k] for r in values]) for k in values[0]['timings']},
                qa_seconds=distribution([r['result']['observed']['qa_seconds'] for r in values]),
                peak_allocated_gib=distribution([r['result']['observed']['peak_allocated_gib'] for r in values])) if values else {}
        for row in data:
            q=row['question_id']; r=group[q]; a=r['result']['answer']; o=r['result']['observed']
            flat.append(dict(method=method,question_id=q,video_id=row['video_id'],stratum=row['stratum'],
                raw_output=a['raw_output'],parsed_answer=a['parsed_answer'],reference_answer='ABCD'[row['answer_index']],correct=r['correct'],
                frames=len(r['selection']['selected_frames']),visual_tokens=o['visual_tokens'],prefill_tokens=o['prefill_tokens']))
    # 步骤2：主要比较只用新协议；开发/非开发与短池影响均单列，不另跑问题。
    comparisons={}
    for left,right in [(METHODS[0],'BASE-Uniform'),(METHODS[1],'BASE-Uniform'),(METHODS[1],METHODS[0])]:
        comparisons[left+'_minus_'+right]={s:compare(rs,groups[left],groups[right]) for s,rs in subsets.items()}
    oldspec=read_json(SOURCE/'protocol.json'); hist=read_json(SOURCE/'historical_baselines.json')
    assert sha256(SOURCE/'historical_baselines.json')==oldspec['selection_exports']['historical_baselines.json']
    history={METHODS[0]:{r['question_id']:r for r in hist if r['method']==METHODS[0]},
             METHODS[1]:{q:read_json(SOURCE/'llava/results'/METHODS[1]/(q+'.json')) for q in ids}}
    old_comparisons={m:compare(data,groups[m],history[m]) for m in METHODS}
    report=dict(status='completed',utc=utc(),protocol_sha256=sha256(run/'protocol.json'),accuracy=accuracy,
        comparisons=comparisons,old_protocol_comparisons=old_comparisons,issues=issues,performance=performance,
        boundary_question_ids=sorted(boundary),answer_call_counts={m:2700 for m in METHODS},
        failures=[str(p.relative_to(run)) for p in (run/'failures').glob('*.json')],
        limitations=['Cached frozen selections: not raw selection E2E','Uniform is a different batch under identical QA protocol',
            'Old-to-new comparison changes QA protocol, not selection algorithm','Full data previously observed, not an unseen blind test'])
    durable(run/'summary.json',report)
    # 步骤3：逐题CSV和可读报告可重建，原始答案和历史快照不改。
    with (run/'per_question.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=['# LMMs-Eval同协议：均匀、Top-K与RD-1.2','', '| 方法 | 总体 | 短 | 中 | 长 |','|---|---:|---:|---:|---:|']
    for m in groups:
        v=accuracy[m]['all2700'];lines.append('| '+m+' | '+' | '.join(f"{v[s]['correct']}/{v[s]['n']} ({100*v[s]['accuracy']:.3f}%)" for s in ('all','short','medium','long'))+' |')
    lines+=['','## 同协议配对','']
    for name, result in comparisons.items():
        r=result['all2700'];lines.append(f"- {name}: {r['difference_pp']:+.3f}个百分点；改善{len(r['cells']['left_only'])}、退化{len(r['cells']['right_only'])}；95%区间{r['stratified_video_cluster_bootstrap_95']}。")
    lines+=['','仅新增5400次问答，均匀复用本日同协议结果。短池题单列，全部方法16帧子集另报。',
        '调用/短池/性能/旧协议对照及全部题号见summary.json。耗时为冻结选帧读取与问答，不称完整选帧E2E。']
    (run/'report.md').write_text('\n'.join(lines)+'\n');print({m:accuracy[m]['all2700']['all'] for m in groups},flush=True)
    return report
