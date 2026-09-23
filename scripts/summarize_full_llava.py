#!/usr/bin/env python3
"""LLaVA阶段独立审计和三组比较；不将Qwen未完成伪装为整轮完成。"""
import argparse
import csv
import os
import sys
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_runtime.baseline_report import wilson,distribution
from videoqa_full.state import rows,METHODS,durable,utc
from videoqa_full.prepare import verify_runtime,verify_frozen_rd
from videoqa_full.run import validate_result
from videoqa_full.report import compare


def summarize(run):
    """接口：2700新结果逐条重放，冻结来源复核后输出阶段报告，不调用任何模型。"""
    run=Path(run);spec=verify_runtime(run,'llava');release=verify_frozen_rd()
    fingerprint=sha256(run/'protocol.json');data=rows()
    assert read_json(run/'llava/execution.json')['status']=='completed'
    calls=list((run/'llava/calls/RD-1.2').glob('*.json'))
    files=list((run/'llava/results/RD-1.2').glob('*.json'))
    assert len(files)==len(calls)==2700
    assert {p.stem for p in files}=={r['question_id'] for r in data}
    assert {p.stem for p in calls}=={r['question_id'] for r in data}
    # 步骤1：旧实验来源与新选择池均按内容哈希检查，不以完成状态代替输入验收。
    for name,h in spec['source_results'].items():assert sha256(ROOT/name)==h
    history=read_json(run/'historical_baselines.json')
    groups={m:{r['question_id']:r for r in history if r['method']==m} for m in METHODS[:2]}
    groups['RD-1.2']={}
    for ordinal,row in enumerate(data):
        record=validate_result(run,'llava','RD-1.2',row,spec,fingerprint,deep=True)
        checkpoint=read_json(run/'llava/checkpoints'/(row['question_id']+'.json'))
        pool=read_json(run/checkpoint['pool_file'])
        for asset in pool['new_assets']:
            assert sha256(Path(asset['path']))==asset['sha256']
        groups['RD-1.2'][row['question_id']]=record
        if (ordinal+1)%100==0:print('Verified',ordinal+1,'/2700',flush=True)
    assert all(len(g)==2700 for g in groups.values())
    dev=read_json(ROOT/'data/manifests/video_mme_development_200_20260920.json')['rows']
    devq={r['question_id'] for r in dev};devv={r['video_id'] for r in dev}
    subsets={'all2700':data,'development200':[r for r in data if r['question_id'] in devq],
        'other2500':[r for r in data if r['question_id'] not in devq],
        'outside_development_videos2100':[r for r in data if r['video_id'] not in devv]}
    assert [len(x) for x in subsets.values()]==[2700,200,2500,2100]
    accuracy={};performance={};flat=[];pairs={};agreement=[]
    # 步骤2：在同一分母上比较三组，使用已有视频簇配对统计，不把三题视为独立视频。
    for method,g in groups.items():
        accuracy[method]={};performance[method]={}
        for subset,rs in subsets.items():
            accuracy[method][subset]={}
            for layer in ('all','short','medium','long'):
                items=[g[r['question_id']] for r in rs if layer=='all' or r['stratum']==layer]
                c=sum(r['correct'] for r in items)
                accuracy[method][subset][layer]=dict(n=len(items),correct=c,accuracy=c/len(items),wilson_95=wilson(c,len(items)))
        for layer in ('all','short','medium','long'):
            items=[r for r in g.values() if (layer=='all' or r['stratum']==layer) and r.get('timing_kind')!='resumed_not_primary']
            entry=dict(n=len(items),timings={},memory={})
            if items:
                for field in ('timings','memory'):
                    shared=set.intersection(*(set(r[field]) for r in items))
                    entry[field]={k:distribution([r[field][k] for r in items]) for k in sorted(shared)}
            performance[method][layer]=entry
        for row in data:
            r=g[row['question_id']];a=r.get('answer',r)
            flat.append(dict(method=method,question_id=row['question_id'],video_id=row['video_id'],stratum=row['stratum'],
                parsed_answer=a['parsed_answer'],reference_answer='ABCD'[row['answer_index']],correct=r['correct'],
                raw_output=a['raw_output'],development_question=row['question_id'] in devq,development_video=row['video_id'] in devv))
    for baseline in METHODS[:2]:
        pairs[baseline]={name:compare(rs,groups['RD-1.2'],groups[baseline]) for name,rs in subsets.items()}
    for row in dev:
        q=row['question_id'];oldpath=ROOT/'outputs/methods/density_e1_200_20260922_r1/results/e1'/(q+'.json')
        assert sha256(oldpath)==spec['historical_rd'][q]
        old=read_json(oldpath);new=groups['RD-1.2'][q]
        if old['answer']['parsed_answer']!=new['answer']['parsed_answer']:
            agreement.append(dict(question_id=q,old_answer=old['answer']['parsed_answer'],new_answer=new['answer']['parsed_answer'],old_correct=old['correct'],new_correct=new['correct']))
    # 步骤3：阶段完成不等于整体完成；明确记录Qwen尚未执行及全部失败/恢复成本。
    summary=dict(status='llava_stage_complete_qwen_pending',overall_experiment_complete=False,
        protocol_sha256=fingerprint,analysis_code_sha256=sha256(Path(__file__)),created_at=utc(),
        accuracy=accuracy,paired_comparisons=pairs,performance=performance,
        execution=read_json(run/'llava/execution.json'),answer_calls=len(calls),
        failures=[str(p.relative_to(run)) for p in (run/'llava/failures').glob('*.json')],
        resumed_ids=[q for q,r in groups['RD-1.2'].items() if r['timing_kind']=='resumed_not_primary'],
        parse_failures={m:[q for q,r in g.items() if r.get('answer',r)['parsed_answer'] is None] for m,g in groups.items()},
        development200_answer_changes=agreement,old_development200_correct=release['result']['correct'],
        qwen_status=read_json(run/'qwen25vl/status.json'),
        verification=dict(new_results=2700,source_baseline_results=5400,all_selection_replayed=True,
            refinement_asset_hashes_checked=True,frozen_algorithm_unchanged=True,new_model_calls=0))
    durable(run/'llava_summary.json',summary)
    with (run/'llava_per_question.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# LLaVA＋RD-1.2全量阶段结果','',
        '本报告仅确认LLaVA阶段完成；Qwen三组尚待注意力实现确认，不能称整轮10800条已完成。','',
        '| 方法 | 全2700 | short900 | medium900 | long900 |','|---|---:|---:|---:|---:|']
    for method in METHODS:
        a=accuracy[method]['all2700']
        lines.append('| '+method+' | '+' | '.join(f"{a[k]['correct']}/{a[k]['n']}（{100*a[k]['accuracy']:.2f}%）" for k in ('all','short','medium','long'))+' |')
    lines+=['','## 泛化子集','', '| 方法 | 开发200 | 其余2500 | 开发视频外2100 |','|---|---:|---:|---:|']
    for method in METHODS:
        a=accuracy[method]
        lines.append('| '+method+' | '+' | '.join(f"{a[k]['all']['correct']}/{a[k]['all']['n']}" for k in ('development200','other2500','outside_development_videos2100'))+' |')
    lines+=['','## 配对结果','']
    for baseline,p in pairs.items():
        for name in subsets:
            v=p[name];lo,hi=v['stratified_video_cluster_bootstrap_95']
            lines.append(f"- RD-1.2−{baseline}，{name}：{v['difference_pp']:+.3f}个百分点，95%视频簇配对区间[{lo:.3f}, {hi:.3f}]；新增正确{len(v['cells']['left_only'])}、退化{len(v['cells']['right_only'])}。")
    lines+=['','## 成本与复测边界','',f"- 阶段墙钟{summary['execution']['elapsed_seconds']/3600:.3f}小时；最终生成{len(calls)}次，失败{len(summary['failures'])}，恢复结果{len(summary['resumed_ids'])}。",
        f"- 开发200题相对历史预测变化{len(agreement)}题，详情见llava_summary.json；不因差异重答。",
        '- 所有历史基线与新方法时延只作不同批次描述；Wilson为边际区间，方法差值用按时长分层的视频簇配对bootstrap10000次。',
        '- 其余2500题仍包括部分开发视频的问题；视频级排除后的2100题另列。此前全量基线已被观察，不声称完全未见盲测。',
        '- 本轮只验证冻结版本，不根据结果新增调参或改变选帧。']
    (run/'llava_report.md').write_text('\n'.join(lines)+'\n')
    print('RD-1.2:',accuracy['RD-1.2']['all2700']['all'],flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_directory');args=parser.parse_args()
    summarize(Path(args.run_directory))
