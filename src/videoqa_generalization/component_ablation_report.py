"""全量组件消融统计：配对结果、成本边界、观测不降门和最多八题复核面板。"""
from collections import Counter
from pathlib import Path
import html
import time

import numpy as np

from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import rows, durable, utc
from videoqa_full.report import compare
from videoqa_runtime.baseline_report import wilson, distribution
from . import component_ablations as study
from . import component_ablation_runtime as runtime


def baseline(profile, method, q):
    """接口：固定同协议已有对照，不新问答、不根据正确性选择来源。"""
    base=(study.PARENTS['llava8'] if profile=='llava8' else study.PARENTS['qwen32']
          if method=='RD-1.2-exp12' else study.PREFIX32 if method=='RD-1.2-exp07' else study.BASE)
    choices=[base/'results'/method/(q+'.json'),base/'imported/results'/method/(q+'.json')]
    found=[p for p in choices if p.exists()]
    if len(found)!=1:raise ValueError('Baseline source missing/ambiguous')
    return found[0]


def recommend(differences):
    """接口：两配置观测精度均不降才列候选；统计等价和默认替换均不自动成立。"""
    if set(differences)!=set(study.PROFILES):raise ValueError('Both configurations are required')
    def net(value):
        return value.get('net_correct',len(value.get('cells',{}).get('left_only',[]))-
                         len(value.get('cells',{}).get('right_only',[])))
    nondecreasing=all(net(v)>=0 for v in differences.values())
    return dict(observed_non_degrade_both_profiles=nondecreasing,
                lower_confidence_bound_nonnegative_both=all(v['stratified_video_cluster_bootstrap_95'][0]>=0 for v in differences.values()),
                replacement_candidate=nondecreasing,default_replacement_authorized=False,
                limitation='Point estimates are not equivalence; B also requires whole-design complexity review')


def summarize(root):
    """接口：四组完整结果才统计；不把缺失回答当成已完成全量或继承父预测。"""
    root=Path(root);data=rows();dev=read_json(study.DEV)['rows']
    dq={r['question_id'] for r in dev};dv={r['video_id'] for r in dev}
    layers={'all2700':data,'development200':[r for r in data if r['question_id'] in dq],
            'other2500':[r for r in data if r['question_id'] not in dq],
            'outside_development_videos2100':[r for r in data if r['video_id'] not in dv]}
    report={};paired_by_control={c:{} for c in study.CONTROLS};total_calls=0;total_synthetic=0;sources={}
    # 步骤1完整物理调用审计后，加载原对照；所有分母按冻结身份确定。
    for profile in study.PROFILES:
        run=root/'qa'/profile;acceptance=runtime.audit(run);spec=read_json(run/'protocol.json')
        total_calls+=acceptance['physical_new_calls'];total_synthetic+=acceptance['synthetic_calls']
        groups={}
        for method in ('BASE-Uniform','BASE-BLIP-TopK','RD-1.2','RD-1.2-exp07','RD-1.2-exp12'):
            bank={}
            for row in data:
                path=baseline(profile,method,row['question_id']);item=read_json(path)
                if any(item[k]!=row[k] for k in ('question_id','video_id','question','options','stratum')):
                    raise ValueError('Baseline question identity changed')
                if item['correct']!=(item['answer']['parsed_answer']=='ABCD'[row['answer_index']]):
                    raise ValueError('Historical baseline scoring differs')
                bank[row['question_id']]=item;sources[str(path)]=sha256(path)
            groups[method]=bank
        for control,method in study.CONTROLS.items():
            groups[method]={r['question_id']:runtime.validate_result(run,r,control,spec) for r in data}
        common=[r for r in data if all(len(g[r['question_id']]['selection']['selected_frames'])==spec['frame_budget'] for g in groups.values())]
        subsets=dict(layers,common_full_budget=common);metrics={};comparisons={};performance={};parsed={}
        for method,group in groups.items():
            metrics[method]={}
            for name,rs in subsets.items():
                if not rs:continue
                count=sum(group[r['question_id']]['correct'] for r in rs)
                metrics[method][name]=dict(n=len(rs),correct=count,accuracy=count/len(rs),wilson_95=wilson(count,len(rs)))
            for label in ('short','medium','long'):
                rs=[r for r in data if r['stratum']==label];count=sum(group[r['question_id']]['correct'] for r in rs)
                metrics[method][label]=dict(n=len(rs),correct=count,accuracy=count/len(rs))
            metrics[method]['official_tasks']={}
            for task in sorted({r['task_type'] for r in data}):
                rs=[r for r in data if r['task_type']==task];count=sum(group[r['question_id']]['correct'] for r in rs)
                metrics[method]['official_tasks'][task]=dict(n=len(rs),correct=count,accuracy=count/len(rs))
        # 步骤2主要四项比较保留开发范围；其余比较只作全量描述，避免重复重抽无关组合。
        for control,method in study.CONTROLS.items():
            comparisons[method]={}
            for ref in ('RD-1.2-exp12','RD-1.2-exp07','BASE-Uniform','BASE-BLIP-TopK','RD-1.2'):
                scopes=subsets if ref=='RD-1.2-exp12' else {'all2700':data}
                comparisons[method][ref]={name:compare(rs,groups[method],groups[ref]) for name,rs in scopes.items() if rs}
            paired_by_control[control][profile]=comparisons[method]['RD-1.2-exp12']['all2700']
            values=list(groups[method].values());fresh=[v for v in values if v['timing_kind']=='new_physical_call']
            parsed[method]=[q for q,v in groups[method].items() if v['answer']['parsed_answer'] is None]
            performance[method]=dict(physical_fresh_results=len(fresh),result_kinds=dict(Counter(v['timing_kind'] for v in values)),
                qa_seconds=distribution([v['answer']['observed']['qa_seconds'] for v in fresh]) if fresh else None,
                final_decode_and_identity_seconds=distribution([v['timings']['final_decode_and_identity_seconds'] for v in fresh]) if fresh else None,
                peak_allocated_gib=distribution([v['answer']['observed']['peak_allocated_gib'] for v in fresh]) if fresh else None)
        comparisons['A_minus_B']={'all2700':compare(data,groups[study.CONTROLS['A']],groups[study.CONTROLS['B']])}
        frame_counts={m:dict(Counter(str(len(v['selection']['selected_frames'])) for v in group.values())) for m,group in groups.items()}
        report[profile]=dict(metrics=metrics,comparisons=comparisons,performance=performance,
                             actual_frame_counts=frame_counts,parse_failures=parsed,call_audit=acceptance)
    if total_calls>10800 or total_synthetic>21:raise ValueError('Global experiment call cap exceeded')
    # 步骤3仅生成候选建议，不自动删模块、组合两个删减或运行MLVU。
    decisions={c:recommend(v) for c,v in paired_by_control.items()}
    result=dict(status='completed',profiles=report,decisions=decisions,new_final_qa=total_calls,
                synthetic_generations=total_synthetic,new_blip_calls=0,new_selection_visual_calls=0,
                code_control_correction='Stored reference_quotas may be feasibility placeholders; B recomputes genuine global first-pass quotas',
                automatic_combination_or_default_replacement=False,source_result_hashes=sources,utc=utc())
    durable(root/'summary.json',result)
    render_summary(root,result)
    return result


def render_summary(root,result):
    """接口：结果先给总体及配对，再解释删减候选和成本边界。"""
    root=Path(root);lines=['# RD组件双配置全量消融','',
        '| 方法 | LLaVA8正确/2700 | 准确率 | Qwen32正确/2700 | 准确率 |','|---|---:|---:|---:|---:|']
    methods=('BASE-Uniform','BASE-BLIP-TopK','RD-1.2','RD-1.2-exp07','RD-1.2-exp12',*study.CONTROLS.values())
    for method in methods:
        a=result['profiles']['llava8']['metrics'][method]['all2700'];b=result['profiles']['qwen32']['metrics'][method]['all2700']
        lines.append(f"| {method} | {a['correct']} | {a['accuracy']*100:.2f}% | {b['correct']} | {b['accuracy']*100:.2f}% |")
    lines+=['','| 控制−exp12 | 配置 | 新增/退化 | Δ百分点 | 视频簇95%区间 |','|---|---|---:|---:|---|']
    for control,method in study.CONTROLS.items():
        for profile,value in result['profiles'].items():
            pair=value['comparisons'][method]['RD-1.2-exp12']['all2700'];lo,hi=pair['stratified_video_cluster_bootstrap_95']
            lines.append(f"| {control} | {profile} | {len(pair['cells']['left_only'])}/{len(pair['cells']['right_only'])} | "
                         f"{pair['difference_pp']:+.3f} | [{lo:+.3f},{hi:+.3f}] |")
    lines+=['','## 事前门槛与解释','']
    for c,decision in result['decisions'].items():
        lines.append(f"- 控制{c}：两配置观测均不降={decision['observed_non_degrade_both_profiles']}；不自动替换父版。")
    lines+=['','B使用真实第一遍重算配额，不使用轨迹中的容量占位reference_quotas；最终帧由同一父局部前缀截取。',
        'A移除细池会连带改变局部次序/配额/短池帧数；B衡量预算组件，不单独归因DP求解器。',
        '两项单独无损不证明组合无损；MLVU迁移、组合删减和默认方法变更另行review。',
        '配对bootstrap按时长分层、视频为簇10000次，seed2027；公开基准已观察，多组比较为探索性。',
        '新生成、历史输入核验引用、同轮共享分开统计，引用不算独立重复实验。',
        '缓存读取/CPU选择、解码/身份核验、QA与批次墙钟分开；不是原片独立E2E，不推导异机速度比。',
        '',f"正式新QA={result['new_final_qa']}；合成={result['synthetic_generations']}；新增BLIP/选帧视觉前向=0。",'']
    (root/'report.md').write_text('\n'.join(lines))


def render_panel(root):
    """接口：每控制/配置首个改善与退化，避免重复视频，最多8题，不跨空槽补足。"""
    root=Path(root);summary=read_json(root/'summary.json');data=rows();bank={r['question_id']:r for r in data}
    used=set();entries=[]
    for control,method in study.CONTROLS.items():
        for profile in study.PROFILES:
            cells=summary['profiles'][profile]['comparisons'][method]['RD-1.2-exp12']['all2700']['cells']
            for kind in ('left_only','right_only'):
                ids=set(cells[kind]);chosen=next((r for r in data if r['question_id'] in ids and r['video_id'] not in used),None)
                if chosen:
                    used.add(chosen['video_id']);entries.append(dict(question_id=chosen['question_id'],profile=profile,
                        control=control,kind=kind,parent_path=str(baseline(profile,'RD-1.2-exp12',chosen['question_id'])),
                        control_path=str(root/'qa'/profile/'results'/method/(chosen['question_id']+'.json'))))
    from videoqa_audit.region_scope_review import draw_board
    boards=[]
    for entry in entries:
        q=entry['question_id'];profile=entry['profile']
        for label,path in [('parent',entry['parent_path']),('control_'+entry['control'],entry['control_path'])]:
            boards.append(draw_board((str(root),q,profile,label,path)))
    durable(root/'review/panel.json',dict(entries=entries,maximum_questions=8,review_minutes_limit=45,
                                          semantic_review='pending',new_model_calls=0))
    page=['<!doctype html><meta charset="utf-8"><title>RD组件消融复核</title><style>body{max-width:1400px;margin:auto;font:16px/1.6 sans-serif}img{max-width:100%}</style>',
          '<h1>RD组件消融：事后差异复核</h1><p>不将答错当作证据丢失；原RGB缩略图不是完整模型可见性证明。</p>']
    for entry in entries:
        q=entry['question_id'];profile=entry['profile'];row=bank[q]
        page+=['<h2>'+html.escape(q+' '+profile+' '+entry['control']+' '+entry['kind'])+'</h2>',
               '<p>'+html.escape(row['question'])+'</p><p>'+html.escape(' / '.join(row['options']))+'</p>']
        for label in ['parent','control_'+entry['control']]:
            page.append('<a href="'+profile+'/'+q+'/'+label+'.jpg"><img src="'+profile+'/'+q+'/'+label+'.jpg"></a>')
    (root/'review/index.html').write_text('\n'.join(page))
    return entries
