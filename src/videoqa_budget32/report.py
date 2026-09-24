"""Qwen同模型16/32帧六组统计与完成后安全恢复验收。"""
from collections import Counter
import csv
from pathlib import Path
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import durable,rows,utc
from videoqa_full.report import compare
from videoqa_runtime.baseline_report import wilson,distribution
from .selection import OLD_QWEN,METHODS


def summarize(run):
    """接口：全量完整配对后统计，离线诊断不附加任何历史预测。"""
    from .run import validate
    run=Path(run);spec=read_json(run/'protocol.json');data=rows();ids={r['question_id'] for r in data}
    groups={};flat=[]
    # 步骤1：新三组实际输入/调用/原始输出闭合，旧三组只读并检查来源指纹。
    for method in METHODS:
        new={};old={}
        for folder in ('results','calls','raw_generations'):
            paths=list((run/folder/method).glob('*.json'));assert len(paths)==2700 and {p.stem for p in paths}==ids
        for row in data:
            q=row['question_id'];new[q]=validate(run,row,method,spec)
            p=OLD_QWEN/'results'/method/(q+'.json');assert sha256(p)==spec['source_results'][str(p)]
            old[q]=read_json(p)
        groups['K32__'+method]=new;groups['K16__'+method]=old
    for q in ids:assert len({groups['K32__'+m][q]['gpu'] for m in METHODS})==1
    dev=read_json(ROOT/'data/manifests/video_mme_development_200_20260920.json')['rows']
    dq={r['question_id'] for r in dev};dv={r['video_id'] for r in dev}
    common={q for q in ids if all(len(groups['K32__'+m][q]['selection']['selected_frames'])==32 for m in METHODS)}
    subsets=dict(all2700=data,development200=[r for r in data if r['question_id'] in dq],
        other2500=[r for r in data if r['question_id'] not in dq],
        outside_development_videos2100=[r for r in data if r['video_id'] not in dv],
        all_methods_32_frames=[r for r in data if r['question_id'] in common])
    for task in sorted({r.get('task_type','unknown') for r in data}):
        subsets['task__'+task]=[r for r in data if r.get('task_type','unknown')==task]
    accuracy={};performance={};issues={}
    # 步骤2：显式分母；边际Wilson与视频簇配对区间不混为一谈。
    for key,group in groups.items():
        accuracy[key]={}
        for name,subset in subsets.items():
            accuracy[key][name]={}
            for layer in ('all','short','medium','long'):
                values=[group[r['question_id']] for r in subset if layer=='all' or r['stratum']==layer]
                if not values:continue
                c=sum(r['correct'] for r in values)
                accuracy[key][name][layer]=dict(n=len(values),correct=c,accuracy=c/len(values),wilson_95=wilson(c,len(values)))
        issues[key]=dict(parse_failures=[q for q,r in group.items() if not r['answer']['parsed_answer']],
            source_frame_counts=dict(Counter(len(r['selection']['selected_frames']) for r in group.values())))
        performance[key]={}
        for layer in ('all','short','medium','long'):
            values=[r for r in group.values() if not r['reused_answer'] and (layer=='all' or r['stratum']==layer)]
            if not values:continue
            performance[key][layer]={k:distribution([r['answer']['observed'][k] for r in values])
                for k in ('qa_seconds','visual_tokens','prefill_tokens','peak_allocated_gib')}
            performance[key][layer]['final_decode_seconds']=distribution([r['timings']['final_decode_seconds'] for r in values])
        for row in data:
            q=row['question_id'];r=group[q]
            flat.append(dict(configuration=key,question_id=q,video_id=row['video_id'],stratum=row['stratum'],
                raw_output=r['answer']['raw_output'],parsed_answer=r['answer']['parsed_answer'],correct=r['correct'],
                source_frames=len(r['selection']['selected_frames'])))
    # 步骤3：固定比较集合，不按最有利结果挑选检验。
    pairs=[('K32__'+a,'K32__'+b) for a,b in [(METHODS[2],METHODS[0]),(METHODS[2],METHODS[1]),(METHODS[1],METHODS[0])]]
    pairs += [('K32__'+m,'K16__'+m) for m in METHODS]
    comparisons={a+'_minus_'+b:{name:compare(rs,groups[a],groups[b]) for name,rs in subsets.items() if rs}
        for a,b in pairs}
    summary=dict(status='completed',accuracy=accuracy,comparisons=comparisons,performance=performance,issues=issues,
        common_32_count=len(common),boundary_question_ids=sorted(ids-common),new_answer_calls={m:2700 for m in METHODS},
        failures=[str(p.relative_to(run)) for p in (run/'failures').glob('*.json')],utc=utc(),
        limitations=['M32/K32 changes seeds, windows and refinement, not only final budget',
            'Cached coarse assets: timings are not independent raw-video E2E',
            'Fixed-old-pool M16/K32 has zero answers and no diagnostic accuracy'])
    durable(run/'summary.json',summary)
    with (run/'per_question.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# Qwen32帧全量结果','', '| 配置 | 正确数 | 准确率 |','|---|---:|---:|']
    for key in groups:
        a=accuracy[key]['all2700']['all'];lines.append(f"| {key} | {a['correct']}/2700 | {a['accuracy']:.2%} |")
    lines+=['','M32/K32是完整预算配置变化；离线M16/K32无问答成绩。',f'三方法均32帧共同子集：{len(common)}题。',
        '完整配对题号、视频簇区间及成本见summary.json；人工阶段归因另行有界复核。']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def audit(run):
    """接口：完成后实际恢复，确认不新增模型调用且历史冻结文件未变。"""
    from .run import execute,verify
    from videoqa_full.prepare import verify_frozen_rd
    run=Path(run);verify(run)
    # 步骤1：结果、账本、原始生成的全部字节指纹先冻结。
    before={str(p.relative_to(run)):sha256(p) for folder in ('results','calls','raw_generations') for p in (run/folder).glob('*/*.json')}
    assert len(before)==24300
    execute(run)
    after={str(p.relative_to(run)):sha256(p) for folder in ('results','calls','raw_generations') for p in (run/folder).glob('*/*.json')}
    assert before==after;verify_frozen_rd()
    # 步骤2：完成恢复只校验与汇总，不加载模型、增加生成。
    durable(run/'completion_audit.json',dict(status='passed',unchanged_files=len(before),extra_generations=0,utc=utc()))
    from .stages import analyze
    analyze(run)
