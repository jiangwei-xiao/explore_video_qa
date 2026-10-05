"""三项冻结全量验证的统一计分、配对统计和性能归档。"""
from pathlib import Path
from collections import Counter
from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import rows, durable, utc
from videoqa_full.report import compare
from videoqa_runtime.baseline_report import wilson, distribution
from .inputs import OLD_QWEN, BASE, PARENT, LLAVA_BASE, LLAVA_UNIFORM, DEV
from .run import STAGES, verify, validate


def reference(directory, method, data, uniform=False):
    """接口：读取同协议旧答案；复算正确性，不能混用不同预算的基线。"""
    group={}
    for row in data:
        q=row['question_id']; path=directory/'results'/(q+'.json') if uniform else directory/'results'/method/(q+'.json')
        item=read_json(path); a=item['answer'] if 'answer' in item else item['result']['answer']
        if item['question']!=row['question'] or item['options']!=row['options'] or item['correct']!=(a['parsed_answer']=='ABCD'[row['answer_index']]):
            raise ValueError('Historical scoring or question differs')
        group[q]=item
    return group


def summarize(root):
    """接口：三组全部完整后提供全量及开发范围配对，缺组直接停止报告。"""
    root=Path(root); data=rows(); groups={}; reports={}
    dev=read_json(DEV)['rows']; dq={r['question_id'] for r in dev}; dv={r['video_id'] for r in dev}
    subsets={'all2700':data, 'development200':[r for r in data if r['question_id'] in dq],
             'other2500':[r for r in data if r['question_id'] not in dq],
             'outside_development_videos2100':[r for r in data if r['video_id'] not in dv]}
    sources={}; total=0
    # 步骤1：每组逐条校验实际输入与调用后，再加载对应模型/预算的历史对照。
    for stage in STAGES:
        run=root/stage; spec=verify(run); audit=read_json(run/'completion_audit.json')
        if audit['status']!='passed': raise ValueError('Stage audit incomplete')
        results={r['question_id']:validate(run,r,spec) for r in data}; groups[stage]=results
        total+=audit['new_calls']; sources[str(run/'protocol.json')]=sha256(run/'protocol.json')
        if stage=='qwen16':
            refs={m:reference(OLD_QWEN,m,data) for m in ['BASE-Uniform','BASE-BLIP-TopK','RD-1.2']}
            refs['exp12_M32_K32']={r['question_id']:read_json(PARENT/'results/RD-1.2-exp12'/(r['question_id']+'.json'))
                if (PARENT/'results/RD-1.2-exp12'/(r['question_id']+'.json')).exists()
                else read_json(PARENT/'imported/results/RD-1.2-exp12'/(r['question_id']+'.json')) for r in data}
        elif stage=='llava16':
            refs={m:reference(LLAVA_BASE,m,data) for m in ['BASE-BLIP-TopK','RD-1.2']}
            refs['BASE-Uniform']=reference(LLAVA_UNIFORM,'BASE-Uniform',data,uniform=True)
        else:
            refs={m:reference(BASE,m,data) for m in ['BASE-Uniform','BASE-BLIP-TopK','RD-1.2']}
            refs['RD-1.2-exp12']=groups.get('qwen32_parent') or {
                r['question_id']:read_json(PARENT/('results' if (PARENT/'results/RD-1.2-exp12'/(r['question_id']+'.json')).exists() else 'imported/results')/'RD-1.2-exp12'/(r['question_id']+'.json')) for r in data}
        same=[r for r in data if len(results[r['question_id']]['selection']['selected_frames'])==spec['source_frame_budget']
              and all(len(g[r['question_id']]['selection']['selected_frames'])==spec['source_frame_budget'] for name,g in refs.items() if name!='exp12_M32_K32')]
        layers=dict(subsets, common_full_budget=same); metrics={}; pairs={}
        # 步骤2：比较分母先固定，官方题型单列；配对按视频簇分层而非题目独立抽样。
        for name, rs in layers.items():
            metrics[name]={}
            for layer in ['all','short','medium','long']:
                subset=[r for r in rs if layer=='all' or r['stratum']==layer]
                if not subset: continue
                c=sum(results[r['question_id']]['correct'] for r in subset)
                metrics[name][layer]=dict(n=len(subset),correct=c,accuracy=c/len(subset),wilson_95=wilson(c,len(subset)))
        for label, group in refs.items():
            pairs[label]={name:compare(rs,results,group) for name,rs in layers.items() if rs}
        tasks={}
        for task in sorted({r['task_type'] for r in data}):
            rs=[r for r in data if r['task_type']==task]; c=sum(results[r['question_id']]['correct'] for r in rs)
            tasks[task]=dict(n=len(rs),correct=c,accuracy=c/len(rs))
        new=[v for v in results.values() if not v.get('imported') and v.get('timing_kind')!='resumed_not_primary']
        perf={}
        for layer in ['all','short','medium','long']:
            values=[v for v in new if layer=='all' or v['stratum']==layer]; timing={}; memory={}
            if values:
                for field in sorted(set.intersection(*(set(v['timings']) for v in values))):
                    timing[field]=distribution([v['timings'][field] for v in values])
                for field in ['qa_seconds','generation_seconds','peak_allocated_gib','peak_reserved_gib']:
                    nums=[v['answer']['observed'][field] for v in values if field in v['answer']['observed']]
                    if nums: (memory if 'gib' in field else timing)[field]=distribution(nums)
            perf[layer]=dict(n=len(values),timings=timing,memory=memory)
        reports[stage]=dict(metrics=metrics,paired=pairs,official_tasks=tasks,performance=perf,
            actual_frame_counts=dict(Counter(len(v['selection']['selected_frames']) for v in results.values())),
            new_calls=audit['new_calls'],imported=audit['imported'],
            parse_failures=[q for q,v in results.items() if v['answer']['parsed_answer'] is None],
            execution=read_json(run/'execution.json'))
    # 步骤3：跨模型必须逐题源帧/RGB一致；不要求预处理Tensor或Token相同。
    for row in data:
        q=row['question_id']; a,b=groups['qwen16'][q],groups['llava16'][q]
        if a['selection']!=b['selection'] or a['rgb_sha256']!=b['rgb_sha256']: raise ValueError('Cross-model source mismatch')
    if total!=7900: raise ValueError('Total new generation count differs')
    result=dict(status='completed',reports=reports,new_answer_calls=7900,sources=sources,cross_model_rgb_exact=2700,
        utc=utc(),limits=['Previously observed dataset; exploratory comparisons',
            'Published WFS sources do not prove author video byte identity or full paper reproduction',
            'Cached input timings are not independent raw-video selection E2E',
            'M16/K16 versus M32/K32 changes complete budget configuration'])
    durable(root/'summary.json',result)
    lines=['# exp12跨模型跨预算与公开参考全量结果','',
           '| 工作点 | 正确数 | 准确率 | 新增问答 | 历史导入 |','|---|---:|---:|---:|---:|']
    for stage,report in reports.items():
        m=report['metrics']['all2700']['all']
        lines.append(f'| {stage} | {m["correct"]}/2700 | {m["accuracy"]*100:.2f}% | {report["new_calls"]} | {report["imported"]} |')
    for stage,report in reports.items():
        lines+=['',f'## {stage}配对比较','']
        for ref,comparisons in report['paired'].items():
            c=comparisons['all2700']; lo,hi=c['stratified_video_cluster_bootstrap_95']
            lines.append(f'- 对{ref}：{c["difference_pp"]:+.3f}个百分点，区间[{lo:.3f},{hi:.3f}]；新增正确{len(c["cells"]["left_only"])}、退化{len(c["cells"]["right_only"])}。')
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
