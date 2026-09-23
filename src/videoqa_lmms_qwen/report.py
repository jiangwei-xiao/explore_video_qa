"""同协议六组统计：Qwen三组新答案与LLaVA三组冻结对照。"""
import csv
from collections import Counter
from pathlib import Path
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import rows,durable,utc
from videoqa_full.report import compare
from videoqa_runtime.baseline_report import wilson,distribution
from videoqa_runtime.protocol import parse_answer
from .run import METHODS,UNIFORM,LLAVA,validate


def summarize(run):
    """接口：8100条齐全才汇总；跨模型同源帧，不混淆不同视觉Token及计时口径。"""
    run=Path(run);spec=read_json(run/'protocol.json');data=rows();ids={r['question_id'] for r in data}
    groups={};issues={};performance={};tables={};flat=[]
    # 步骤1：按完整题号核验新账本及真实输入，模型内外都必须完全配对。
    for m in METHODS:
        for folder in ('results','calls','raw_generations'):
            ps=list((run/folder/m).glob('*.json'));assert len(ps)==2700 and {p.stem for p in ps}==ids
        groups['qwen__'+m]={r['question_id']:validate(run,r,m,spec) for r in data}
        groups['llava__'+m]={}
        for r in data:
            q=r['question_id'];path=(UNIFORM/'results' if m==METHODS[0] else LLAVA/'results'/m)/(q+'.json')
            assert sha256(path)==spec['source_results'][str(path)]
            old=read_json(path);fresh=groups['qwen__'+m][q]
            assert old['selection']==fresh['selection'] and old['rgb_sha256']==fresh['rgb_sha256']
            groups['llava__'+m][q]=old
    for q in ids:assert len({groups['qwen__'+m][q]['gpu'] for m in METHODS})==1
    dev=read_json(ROOT/'data/manifests/video_mme_development_200_20260920.json')['rows']
    dq={r['question_id'] for r in dev};dv={r['video_id'] for r in dev}
    boundary={q for q in ids if any(len(groups['qwen__'+m][q]['selection']['selected_frames'])!=16 for m in METHODS)}
    subsets={'all2700':data,'development200':[r for r in data if r['question_id'] in dq],
        'other2500':[r for r in data if r['question_id'] not in dq],
        'outside_development_videos2100':[r for r in data if r['video_id'] not in dv],
        'all_methods_16_frames2694':[r for r in data if r['question_id'] not in boundary]}
    assert [len(x) for x in subsets.values()]==[2700,200,2500,2100,2694]
    # 步骤2：总体/分时长/已观察开发子集分开；不把抽样方差和视觉Token差异省略。
    for key,group in groups.items():
        tables[key]={}
        for name,rs in subsets.items():
            tables[key][name]={}
            for layer in ('all','short','medium','long'):
                values=[group[r['question_id']] for r in rs if layer=='all' or r['stratum']==layer]
                if not values:continue
                c=sum(r['correct'] for r in values)
                tables[key][name][layer]=dict(n=len(values),correct=c,accuracy=c/len(values),wilson_95=wilson(c,len(values)))
        answers={q:(r['answer'] if key.startswith('qwen') else r['result']['answer']) for q,r in group.items()}
        issues[key]=dict(parse_failures=[q for q,a in answers.items() if not a['parsed_answer']],
            strict_parser_differences=[q for q,a in answers.items() if parse_answer(a['raw_output'])!=(a['parsed_answer'] or None)])
        if key.startswith('qwen'):
            issues[key].update(limit_hits=[q for q,a in answers.items() if len(a['observed']['generated_token_ids'])==16],
                recovered=[q for q,r in group.items() if r['reused_answer']],
                source_frame_counts=dict(Counter(a['observed']['source_frames'] for a in answers.values())),
                encoded_frame_counts=dict(Counter(a['observed']['encoded_frames'] for a in answers.values())))
            performance[key]={}
            for layer in ('all','short','medium','long'):
                values=[r for r in group.values() if not r['reused_answer'] and (layer=='all' or r['stratum']==layer)]
                if not values:continue
                performance[key][layer]=dict(timings={k:distribution([r['timings'][k] for r in values]) for k in values[0]['timings']},
                    qa_seconds=distribution([r['answer']['observed']['qa_seconds'] for r in values]),
                    visual_tokens=distribution([r['answer']['observed']['visual_tokens'] for r in values]),
                    prefill_tokens=distribution([r['answer']['observed']['prefill_tokens'] for r in values]),
                    peak_allocated_gib=distribution([r['answer']['observed']['peak_allocated_gib'] for r in values]))
        for row in data:
            q=row['question_id'];r=group[q];a=answers[q]
            o=a['observed'] if key.startswith('qwen') else r['result']['observed']
            flat.append(dict(model=key.split('__')[0],method=r['method'],question_id=q,video_id=row['video_id'],stratum=row['stratum'],
                raw_output=a['raw_output'],parsed_answer=a['parsed_answer'],reference_answer='ABCD'[row['answer_index']],correct=r['correct'],
                source_frames=len(r['selection']['selected_frames']),visual_tokens=o['visual_tokens'],prefill_tokens=o['prefill_tokens']))
    comparisons={}
    for model in ('qwen','llava'):
        for left,right in [(METHODS[2],METHODS[0]),(METHODS[2],METHODS[1]),(METHODS[1],METHODS[0])]:
            comparisons[model+'__'+left+'_minus_'+right]={n:compare(rs,groups[model+'__'+left],groups[model+'__'+right]) for n,rs in subsets.items()}
    summary=dict(status='completed',accuracy=tables,comparisons=comparisons,issues=issues,performance=performance,
        new_answer_calls={m:2700 for m in METHODS},boundary_question_ids=sorted(boundary),
        failures=[str(p.relative_to(run)) for p in (run/'failures').glob('*.json')],
        protocol_sha256=sha256(run/'protocol.json'),utc=utc(),
        limitations=['Same source RGB, not equal visual tokens across models','Encoding FPS=2 is not original PTS; no timestamp text',
            'Cached selection timings, not raw selection E2E','Observed full dataset, not an unseen blind test'])
    durable(run/'summary.json',summary)
    # 步骤3：派生CSV和Markdown可复算，原始答案及旧模型结果保持字节不变。
    with (run/'per_question.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=['# Qwen与LLaVA：Video-MME同协议六组对照','', '| 模型/方法 | 总体 | 短 | 中 | 长 |','|---|---:|---:|---:|---:|']
    for key,t in tables.items():
        v=t['all2700'];lines.append('| '+key+' | '+' | '.join(f"{v[s]['correct']}/{v[s]['n']} ({100*v[s]['accuracy']:.3f}%)" for s in ('all','short','medium','long'))+' |')
    lines+=['','## Qwen配对差值','']
    for key,v in comparisons.items():
        if not key.startswith('qwen'):continue
        r=v['all2700'];lines.append(f"- {key}: {r['difference_pp']:+.3f}个百分点，改善{len(r['cells']['left_only'])}、退化{len(r['cells']['right_only'])}；视频簇95%区间{r['stratified_video_cluster_bootstrap_95']}。")
    lines+=['','完整得失/子集/Token/性能及解析失败见summary.json。只有Qwen8100条为本轮新增；LLaVA答案不重答。']+summary['limitations']
    (run/'report.md').write_text('\n'.join(lines)+'\n');print({k:v['all2700']['all'] for k,v in tables.items()},flush=True)
    return summary
