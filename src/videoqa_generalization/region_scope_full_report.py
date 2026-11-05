"""exp27全量统计与独立收缩判断；引用、物理成本及有界中长视频面板分开报告。"""
from collections import Counter
from pathlib import Path
import html
import numpy as np
from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import rows, durable, utc
from videoqa_full.report import compare
from videoqa_runtime.baseline_report import wilson, distribution
from . import region_scope_full as study
from . import region_scope_full_runtime as runtime
from .component_ablation_report import baseline


def decision(nets):
    """接口：两工作点点估计均不降才保留候选；不把该门槛写成统计等价。"""
    if set(nets)!=set(study.PROFILES):raise ValueError('Both profiles required')
    return dict(observed_non_degrade_both=all(v>=0 for v in nets.values()),
        net_correct=nets,default_replacement=False,exp12_transfer_validated=False,
        scope='Video-MME exp07-based refinement-scope contraction only')


def summarize(root):
    """接口：5400完整结果独立复算后汇总；不通过中途分数改变规则或停止另一配置。"""
    root=Path(root)
    if (root/'summary.json').exists():
        value=read_json(root/'summary.json')
        for p,h in value['source_result_hashes'].items():
            if sha256(p)!=h:raise ValueError('Reported source changed')
        for p in study.PROFILES:runtime.audit(root/'qa'/p)
        render(root,value);return value
    data=rows();dev=read_json(study.DEV)['rows'];dq={r['question_id'] for r in dev};dv={r['video_id'] for r in dev}
    layers={'all2700':data,'development200':[r for r in data if r['question_id'] in dq],
        'other2500':[r for r in data if r['question_id'] not in dq],
        'outside_development_videos2100':[r for r in data if r['video_id'] not in dv]}
    report={};nets={};sources={};total=0;synth=0
    # 步骤1正式调用与引用完全闭合，再读取固定历史对照；标准答案只用于计分。
    for profile in study.PROFILES:
        run=root/'qa'/profile;accept=runtime.audit(run);spec=read_json(run/'protocol.json');total+=accept['physical_new_calls'];synth+=accept['synthetic_calls']
        groups={}
        for method in ('BASE-Uniform','BASE-BLIP-TopK','RD-1.2-exp07','RD-1.2-exp12'):
            bank={}
            for row in data:
                q=row['question_id'];path=baseline(profile,method,q);v=read_json(path)
                if any(v[k]!=row[k] for k in ('question_id','video_id','question','options','stratum')):raise ValueError('Baseline identity differs')
                if v['correct']!=(v['answer']['parsed_answer']=='ABCD'[row['answer_index']]):raise ValueError('Baseline scoring differs')
                bank[q]=v;sources[str(path)]=sha256(path)
            groups[method]=bank
        current={r['question_id']:runtime.validate_result(run,r,'S',spec) for r in data};groups[study.METHOD]=current
        subsets=dict(layers,common_full_budget=[r for r in data if all(len(g[r['question_id']]['selection']['selected_frames'])==spec['frame_budget'] for g in groups.values())])
        for kind in ('development','parent','new'):
            subsets['input_'+kind]=[r for r in data if read_json(run/'inputs'/(r['question_id']+'.json'))['reference_kind']==kind]
        metrics={};pairs={}
        for method,group in groups.items():
            metrics[method]={}
            for name,rs in subsets.items():
                count=sum(group[r['question_id']]['correct'] for r in rs)
                metrics[method][name]=dict(n=len(rs),correct=count,accuracy=count/len(rs) if rs else None,wilson_95=wilson(count,len(rs)) if rs else None)
            for label in ('short','medium','long'):
                rs=[r for r in data if r['stratum']==label];n=sum(group[r['question_id']]['correct'] for r in rs)
                metrics[method][label]=dict(n=len(rs),correct=n,accuracy=n/len(rs))
            metrics[method]['official_tasks']={}
            for task in sorted({r['task_type'] for r in data}):
                rs=[r for r in data if r['task_type']==task];n=sum(group[r['question_id']]['correct'] for r in rs)
                metrics[method]['official_tasks'][task]=dict(n=len(rs),correct=n,accuracy=n/len(rs))
        # 步骤2配对抽样仅以视频为簇，时长分层10000次；子集分母与主分母分别保留。
        for ref in ('RD-1.2-exp07','RD-1.2-exp12','BASE-Uniform','BASE-BLIP-TopK'):
            scopes=subsets if ref=='RD-1.2-exp07' else {'all2700':data}
            pairs[ref]={name:compare(rs,current,groups[ref]) for name,rs in scopes.items() if rs}
        nets[profile]=metrics[study.METHOD]['all2700']['correct']-metrics['RD-1.2-exp07']['all2700']['correct']
        fresh=[v for v in current.values() if v['timing_kind']=='new_physical_call']
        hist=[read_json(root/'selections'/profile/study.METHOD/(r['question_id']+'.json')) for r in data]
        offline=read_json(root/'offline_summary.json')['metrics'][profile]
        report[profile]=dict(metrics=metrics,comparisons=pairs,call_audit=accept,reference_counts=spec['reference_counts'],
            actual_frame_counts={m:dict(Counter(str(len(v['selection']['selected_frames'])) for v in g.values())) for m,g in groups.items()},
            parse_failure_ids=[q for q,v in current.items() if v['answer']['parsed_answer'] is None],
            performance=dict(new_physical_answers=len(fresh),qa_seconds=distribution([v['answer']['observed']['qa_seconds'] for v in fresh]),
                decode_and_identity_seconds=distribution([v['timings']['final_decode_and_identity_seconds'] for v in fresh]),
                peak_allocated_gib=distribution([v['answer']['observed']['peak_allocated_gib'] for v in fresh]),
                asset_read_seconds=distribution([v['timings']['asset_read_seconds'] for v in hist]),
                selection_seconds=distribution([v['timings']['selection_seconds'] for v in hist]),
                preparation=read_json(run/'prepare_completed.json'),execution=read_json(run/'execution.json')),
            candidate_scope=offline)
    if total!=1096 or synth>16:raise ValueError('Global call budget differs')
    value=dict(status='completed',method=study.METHOD,profiles=report,decision=decision(nets),
        new_real_qa=total,synthetic_generations=synth,source_result_hashes=sources,new_blip_calls=0,
        new_selection_visual_calls=0,new_refinement_candidates=0,new_training_updates=0,utc=utc())
    study.freeze(root/'summary.json',value);render(root,value);return value


def render(root,value):
    """接口：表格先展示同协议精度/配对，再说明限定条件和资产共享成本。"""
    lines=['# exp27双配置全量确认','', '| 方法 | LLaVA8正确/2700 | 准确率 | Qwen32正确/2700 | 准确率 |','|---|---:|---:|---:|---:|']
    for method in ('BASE-Uniform','BASE-BLIP-TopK','RD-1.2-exp07','RD-1.2-exp12',study.METHOD):
        a=value['profiles']['llava8']['metrics'][method]['all2700'];b=value['profiles']['qwen32']['metrics'][method]['all2700']
        lines.append(f"| {method} | {a['correct']} | {a['accuracy']*100:.2f}% | {b['correct']} | {b['accuracy']*100:.2f}% |")
    lines+=['','| exp27−exp07 | 新增/退化 | Δ百分点 | 时长分层视频簇95%区间 |','|---|---:|---:|---|']
    for p,v in value['profiles'].items():
        pair=v['comparisons']['RD-1.2-exp07']['all2700'];lo,hi=pair['stratified_video_cluster_bootstrap_95']
        lines.append(f"| {p} | {len(pair['cells']['left_only'])}/{len(pair['cells']['right_only'])} | {pair['difference_pp']:+.3f} | [{lo:+.3f},{hi:+.3f}] |")
    lines+=['',f"两配置观测精度均不下降：{value['decision']['observed_non_degrade_both']}。这是候选保留门，不是统计等价或默认方法替换。",
        f"本轮新真实QA={value['new_real_qa']}，合成={value['synthetic_generations']}，其余结果按完整输入身份引用。",
        '补查范围只由触发核心决定；原选择区域、支持、锚点、密度和等权覆盖DP保留。',
        '开发200题及历史基准均已观察，不称独立盲测；不将开发引用视为重复性验收。',
        '缓存读取/选择、CPU输入核验、解码与QA成本单列；没有新增BLIP或细池生成，不声称原片E2E加速。',
        '未验证exp12移植或MLVU迁移，不自动执行新版本。','']
    path=Path(root)/'report.md'
    if path.exists():
        if path.read_text()!='\n'.join(lines):raise ValueError('Completed report differs')
    else:path.write_text('\n'.join(lines))


def panel_entries(data,profiles,development_videos):
    """接口：中/长×改善/退化各一个，优先开发视频外、避免重复视频，空槽不补。"""
    entries=[];used=set()
    # 步骤1先按固定槽位筛选；步骤2只用冻结题序，不按画面或效果大小挑题。
    for profile in study.PROFILES:
        cells=profiles[profile]['comparisons']['RD-1.2-exp07']['all2700']['cells']
        for stratum in ('medium','long'):
            for kind in ('left_only','right_only'):
                ids=set(cells[kind]);eligible=[r for r in data if r['stratum']==stratum and r['question_id'] in ids and r['video_id'] not in used]
                preferred=[r for r in eligible if r['video_id'] not in development_videos]
                if not (preferred or eligible):continue
                row=(preferred or eligible)[0];used.add(row['video_id'])
                entries.append(dict(profile=profile,question_id=row['question_id'],stratum=stratum,kind=kind))
    return entries


def render_panel(root):
    """接口：只生成最多八题的差异复核页；画面事实由有界人工复核单独记录。"""
    from videoqa_audit.region_scope_review import draw_board
    root=Path(root);value=read_json(root/'summary.json');data=rows();bank={r['question_id']:r for r in data}
    dev={r['video_id'] for r in read_json(study.DEV)['rows']};entries=panel_entries(data,value['profiles'],dev)
    page=['<!doctype html><meta charset="utf-8"><title>exp27中长视频差异复核</title><style>body{max-width:1400px;margin:auto;font:16px/1.6 sans-serif}img{max-width:100%}</style>','<h1>exp27全量：中长视频差异</h1>']
    for e in entries:
        q,p=e['question_id'],e['profile'];row=bank[q]
        e['parent_path']=str(baseline(p,'RD-1.2-exp07',q));e['control_path']=str(root/'qa'/p/'results'/study.METHOD/(q+'.json'))
        page+=['<h2>'+html.escape(q+' '+p+' '+e['stratum']+' '+e['kind'])+'</h2>','<p>'+html.escape(row['question'])+'</p>','<p>'+html.escape(' / '.join(row['options']))+'</p>']
        for label,path in [('parent',e['parent_path']),('exp27',e['control_path'])]:
            draw_board((str(root),q,p,label,path));page.append(f'<a href="{p}/{q}/{label}.jpg"><img src="{p}/{q}/{label}.jpg"></a>')
    study.freeze(root/'review/panel.json',dict(entries=entries,maximum_questions=8,minutes_limit=45,semantic_review='pending',new_model_calls=0))
    (root/'review/index.html').write_text('\n'.join(page));return entries
