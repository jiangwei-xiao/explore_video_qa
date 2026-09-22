"""密度v1结果复核、统计和轻量输入图板；不运行任何问答模型。"""
import csv
import html
import math
from pathlib import Path
import numpy as np
from videoqa_runtime.common import ROOT, read_json, write_json, sha256


def render_board(path, images, selected, processor, qid):
    """接口：复用本次最终RGB，渲染实际384预处理可见内容；属于答题后的展示成本。"""
    from PIL import Image, ImageDraw
    pixels=processor.preprocess(images,return_tensors='np')['pixel_values']
    canvas=Image.new('RGB',(4*384,32+math.ceil(len(images)/4)*410),'white');draw=ImageDraw.Draw(canvas)
    draw.text((8,8),f'{qid} Density-soft: ordered 384x384 QA input',fill='black')
    # 步骤1：反归一化显示实际预处理内容，不以原帧缩略图冒充模型输入。
    for j,(pixel,row) in enumerate(zip(pixels,selected)):
        rgb=pixel.transpose(1,2,0)*np.asarray(processor.image_std)+np.asarray(processor.image_mean)
        view=Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype('uint8'))
        x,y=j%4*384,32+j//4*410;canvas.paste(view,(x,y))
        draw.text((x+4,y+386),f'#{row["candidate_index"]}  {row["timestamp_seconds"]:.5f}s  PTS {row["source_pts"]}',fill='black')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def distribution(values):
    """报告耗时/数量分布；空的恢复或首次子集不伪造为零。"""
    if not values:
        return dict(n=0,mean=None,p50=None,p90=None,p95=None)
    values=np.asarray(values,dtype=float)
    return dict(n=len(values),mean=float(values.mean()),p50=float(np.percentile(values,50)),
                p90=float(np.percentile(values,90)),p95=float(np.percentile(values,95)))


def accuracy(records):
    """原始计分与95% Wilson区间，解析失败保留并计错。"""
    n=len(records);k=sum(r['correct'] for r in records);p=k/n;z=1.959963984540054
    denominator=1+z*z/n;center=(p+z*z/(2*n))/denominator
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return dict(correct=k,n=n,accuracy=p,wilson95=[center-radius,center+radius])


def summarize(run):
    """接口：全50题完整性、调用账本与重放通过后生成派生报告，不改逐题原始结果。"""
    from .density_run import MANIFEST,HISTORY_PATHS,execution_protocol,validate_result
    run=Path(run);rows=read_json(MANIFEST)['rows'];ids=[r['question_id'] for r in rows]
    spec=read_json(run/'protocol.json');fingerprint=sha256(run/'protocol.json')
    # 步骤1：核对固定题集、代码/历史身份、生成调用和所有新帧/特征资产。
    if spec!=execution_protocol() or read_json(run/'execution.json')['status']!='completed':
        raise RuntimeError('Execution not complete or frozen fingerprint changed')
    if {p.stem for p in (run/'results').glob('*.json')}!=set(ids):
        raise RuntimeError('Expected exactly 50 paired results')
    calls=[read_json(p) for p in (run/'calls/answer').glob('*.json')]
    if len(calls)!=50 or {c['question_id'] for c in calls}!=set(ids) or any(c['status']!='completed' for c in calls):
        raise RuntimeError('Answer ledger incomplete or budget exceeded')
    if any((run/'calls'/kind).exists() and list((run/'calls'/kind).glob('*.json')) for kind in ('scope','query')):
        raise RuntimeError('Unexpected text-model calls')
    current={q:read_json(run/'results'/f'{q}.json') for q in ids}
    display_processor=None
    for row in rows:
        q=row['question_id'];validate_result(run,current[q],row,fingerprint,verify_assets=True)
        if not (run/'boards'/f'{q}.png').exists():
            # 已有答案不重生成；展示落盘中断只重建CPU图板。
            from transformers import SiglipImageProcessor
            from videoqa_methods.region_context import read_exact_frames
            if display_processor is None:
                display_processor=SiglipImageProcessor.from_pretrained(read_json(ROOT/'configs/local_model_inventory.json')['models']['vision']['path'],local_files_only=True)
            selection=current[q]['selection'];images=read_exact_frames(selection['video'],selection['selected_frames'])
            render_board(run/'boards'/f'{q}.png',images,selection['selected_frames'],display_processor,q)
            for image in images:
                image.close()
    methods={'Density-soft':current,**{name:{q:read_json(path/f'{q}.json') for q in ids} for name,path in HISTORY_PATHS.items()}}
    groups={'all':ids,**{s:[r['question_id'] for r in rows if r['stratum']==s] for s in ('short','medium','long')}}
    summary=dict(run_id=run.name,accuracy={},comparisons={},performance={},mechanisms={},
                 calls=dict(answer_started=len(calls),answer_completed=len(calls),scope=0,query=0),
                 execution=read_json(run/'execution.json'),preflight=read_json(run/'preflight.json'),
                 failures=[read_json(p) for p in (run/'failures').glob('*.json')],
                 scope='observed development50; historical performance descriptive across batches')
    # 步骤2：分层统计以及固定种子的配对bootstrap，不删答错/解析失败题。
    for name,records in methods.items():
        summary['accuracy'][name]={s:accuracy([records[q] for q in qids]) for s,qids in groups.items()}
    for name in HISTORY_PATHS:
        rng=np.random.default_rng(2027);samples=np.zeros(10000)
        for stratum in ('short','medium','long'):
            differences=np.array([int(current[q]['correct'])-int(methods[name][q]['correct']) for q in groups[stratum]])
            samples+=rng.choice(differences,size=(10000,len(differences)),replace=True).sum(axis=1)/50*100
        improved=[q for q in ids if current[q]['correct'] and not methods[name][q]['correct']]
        regressed=[q for q in ids if not current[q]['correct'] and methods[name][q]['correct']]
        summary['comparisons'][name]=dict(delta_pp=2*(len(improved)-len(regressed)),improved=improved,regressed=regressed,
            bootstrap95_pp=np.percentile(samples,[2.5,97.5]).tolist(),
            both_correct=[q for q in ids if current[q]['correct'] and methods[name][q]['correct']],
            both_wrong=[q for q in ids if not current[q]['correct'] and not methods[name][q]['correct']],
            prediction_changes=[q for q in ids if current[q]['answer']['parsed_answer']!=methods[name][q]['answer']['parsed_answer']])
    for stratum,qids in groups.items():
        fresh=[current[q] for q in qids if current[q]['timing_kind']=='direct_no_application_cache']
        summary['performance'][stratum]={key:distribution([r['timings'][key] for r in fresh]) for key in current[ids[0]]['timings']}
    summary['resumed_questions']=[q for q in ids if current[q]['timing_kind']!='direct_no_application_cache']
    summary['memory']={key:distribution([r['memory'][key] for r in current.values()]) for key in current[ids[0]]['memory']}
    summary['workers']=[read_json(p) for p in sorted((run/'workers').glob('*.json')) if not p.name.endswith('_final.json')]
    summary['blip_real_compute']={key:sum(r['blip_calls'][key] for r in current.values()) for key in current[ids[0]]['blip_calls']}
    summary['parse_failures']=[q for q in ids if current[q]['answer']['parsed_answer'] is None]
    mechanisms=[]
    for q in ids:
        s=current[q]['selection'];selected=set(s['selected_indices']);old_pts={r['source_pts'] for r in methods['TopK'][q]['selection']['selected_frames']}
        mechanisms.append(dict(question_id=q,regions=len(s['plan']['regions']),cores=len(s['plan']['cores']),
            refined_regions=sum(r['refine'] for r in s['plan']['regions']),new_candidates=s['new_candidate_count'],
            new_frames_selected=sum(i>=s['initial_count'] for i in selected),protected=len(s['selection_trace']['protected_indices']),
            topk_overlap=sum(r['source_pts'] in old_pts for r in s['selected_frames']),
            zero_diversity_fill=sum(t['winner']['utility'] == 0 for t in s['selection_trace']['rounds']),
            budget=s['selection_trace']['effective_frame_budget'],region_budgets=s['selection_trace']['region_budgets']))
    summary['mechanisms']=dict(per_question=mechanisms,refinement_questions=sum(m['refined_regions']>0 for m in mechanisms),
        total_new_candidates=sum(m['new_candidates'] for m in mechanisms),total_new_frames_selected=sum(m['new_frames_selected'] for m in mechanisms),
        changed_from_topk=[m['question_id'] for m in mechanisms if m['topk_overlap']!=m['budget']],
        distributions={k:distribution([m[k] for m in mechanisms]) for k in ('regions','cores','refined_regions','new_candidates','new_frames_selected','topk_overlap','zero_diversity_fill')})
    write_json(run/'summary.json',summary)
    # 步骤3：保存逐题答案、机制CSV及可点击图板索引，只报告现有证据支持的结论。
    with (run/'answers.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['question_id','stratum','reference','raw_output','prediction','correct','TopK_correct','Uniform_correct','B_correct','C_local_correct','R_correct','new_candidates','new_selected','e2e_seconds'])
        for row,m in zip(rows,mechanisms):
            q=row['question_id'];record=current[q]
            writer.writerow([q,row['stratum'],record['reference_answer'],record['answer']['raw_output'],record['answer']['parsed_answer'],record['correct'],
                *[methods[name][q]['correct'] for name in ('TopK','Uniform','B','C_local','R')],m['new_candidates'],m['new_frames_selected'],record['timings']['end_to_end_seconds']])
    lines=['# 检索密度软折中：开发50题初版结果','',f'运行：`{run.name}`。冻结参数单版本；50题均为已观察开发集，旧方法不重新生成。','',
           '| 方法 | 总体 | 短 | 中 | 长 |','|---|---:|---:|---:|---:|']
    for name,result in summary['accuracy'].items():
        lines.append('| '+name+' | '+' | '.join(f'{result[g]["correct"]}/{result[g]["n"]} ({result[g]["accuracy"]:.0%})' for g in ('all','short','medium','long'))+' |')
    a=summary['accuracy']['Density-soft']['all'];comp=summary['comparisons']['TopK']
    lines+=['',f'本方法95% Wilson区间：{a["wilson95"][0]:.1%}–{a["wilson95"][1]:.1%}。',
            f'相对Top-K：{comp["delta_pp"]:+.1f}个百分点，分时长配对bootstrap（10000次，种子2027）95%区间为{comp["bootstrap95_pp"]}个百分点。',
            f'新增正确：{comp["improved"]}；退化：{comp["regressed"]}。','',
            f'补查触发{summary["mechanisms"]["refinement_questions"]}/50题，新增候选{summary["mechanisms"]["total_new_candidates"]}张，其中{summary["mechanisms"]["total_new_frames_selected"]}张进入最终输入。',
            f'最终问答{len(calls)}次，分类/查询均0；失败记录{len(summary["failures"])}条，解析失败{summary["parse_failures"]}。','',
            '| 性能（首次执行，秒） | 均值 | P50 | P90 | P95 |','|---|---:|---:|---:|---:|']
    for metric in ('selection_core_seconds','selection_total_seconds','qa_total_seconds','end_to_end_seconds'):
        d=summary['performance']['all'][metric]
        lines.append('| '+metric+' | '+' | '.join('—' if d[k] is None else f'{d[k]:.3f}' for k in ('mean','p50','p90','p95'))+' |')
    lines+=['',f'批次活跃墙钟：{summary["execution"]["total_active_wall_seconds"]/60:.2f}分钟；包含预核验、模型加载、预热、展示和落盘。',
            'E2E为模型就绪后从原视频处理到答案解析的直接墙钟，包含本方法必要候选资产/输入检查点；与旧批次时延仅作描述性比较。',
            '补入帧数量与视觉差异不是证据充分性。下一步只对配对改善/退化及共同失败题查看选帧轨迹，不根据本轮答案自动改参或追加版本。',
            '完整置信区间、各组逐题配对、阶段耗时、显存和失败信息见summary.json；实际输入图板见index.html。']
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    blocks=['<!doctype html><meta charset="utf-8"><style>body{font:16px/1.7 sans-serif;max-width:1100px;margin:auto}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>检索密度软折中：50题输入与结果</h1>']
    for row,m in zip(rows,mechanisms):
        q=row['question_id'];r=current[q]
        blocks.append(f'<details id="{q}"><summary>{q}：预测{html.escape(str(r["answer"]["parsed_answer"]))}，正确={r["correct"]}，新增帧入选{m["new_frames_selected"]}</summary><pre>{html.escape(row["question"])}\n{html.escape(chr(10).join(row["options"]))}</pre><p>标准答案：{r["reference_answer"]} · <a href="results/{q}.json">完整轨迹</a></p><a href="boards/{q}.png"><img loading="lazy" src="boards/{q}.png"></a></details>')
    (run/'index.html').write_text(''.join(blocks))
    files=[run/'results'/f'{q}.json' for q in ids]+[run/'summary.json',run/'answers.csv',run/'report.md',run/'index.html',Path(__file__)]
    write_json(run/'analysis_manifest.json',{str(p.relative_to(ROOT)):sha256(p) for p in files})
    return summary
