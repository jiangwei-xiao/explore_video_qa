"""同粗池小测统计与事后结构映射；全量离线选择不产生全量准确率。"""
from collections import Counter
import html
from pathlib import Path

from . import selection_structure as method
from . import selection_structure_run as study


def panel_entries(data, comparisons):
    """接口：最多8题，固定开发顺序、不同视频；歧义题及空槽不替补跨类别。"""
    used=set(); entries=[]
    for profile in study.PROFILES:
        for control in method.CONTROLS:
            pair=comparisons[profile][method.PARENT+'__minus__'+control]
            for kind in ('left_only','right_only'):
                ids=set(pair['cells'][kind])
                row=next((r for r in data if r['question_id'] in ids and r['video_id'] not in used
                          and r['question_id'] not in {'491-2','321-2'}),None)
                if row is not None:
                    used.add(row['video_id']); entries.append(dict(profile=profile,control=control,kind=kind,
                        question_id=row['question_id'],video_id=row['video_id']))
    return entries


def summarize(root):
    """接口：两配置各600条主结果验收后统计，P−G/P−W主要、G−W次要。"""
    from videoqa_full.report import compare
    from videoqa_runtime.baseline_report import wilson,distribution
    from . import selection_structure_runtime as runtime
    root=Path(root); data=study.development_rows(); profiles={}; comparisons={}; source_hashes={}
    total=synthetic=0; all_methods=(method.PARENT,*method.CONTROLS)
    # 步骤1主表只用同粗池P/G/W；完整细池父版只作为另表背景，不混同信息条件。
    for profile in study.PROFILES:
        run=root/'qa'/profile; acceptance=runtime.audit(run); spec=study.read(run/'protocol.json')
        total+=acceptance['new_real_qa']; synthetic+=acceptance['synthetic_qa']
        groups={m:{} for m in all_methods}
        for m in all_methods:
            for row in data:
                path=run/'results'/m/(row['question_id']+'.json')
                groups[m][row['question_id']]=study.read(path); source_hashes[str(path)]=study.sha(path)
        common=[r for r in data if all(len(groups[m][r['question_id']]['selection']['selected_frames'])==spec['frame_budget'] for m in all_methods)]
        subsets=dict(development200=data,common_full_budget=common)
        subsets.update({s:[r for r in data if r['stratum']==s] for s in ('short','medium','long')})
        metrics={}; performance={}; background={}
        for m,bank in groups.items():
            scopes=dict(subsets)
            scopes.update({'task/'+t:[r for r in data if r['task_type']==t] for t in sorted({r['task_type'] for r in data})})
            metrics[m]={}
            for name,rs in scopes.items():
                if not rs: continue
                n=len(rs); c=sum(bank[r['question_id']]['correct'] for r in rs)
                metrics[m][name]=dict(n=n,correct=c,accuracy=c/n,wilson_95=wilson(c,n))
            fresh=[v for v in bank.values() if v['timing_kind']=='new_physical_call']
            performance[m]=dict(kinds=dict(Counter(v['timing_kind'] for v in bank.values())),
                qa_seconds=distribution([v['answer']['observed']['qa_seconds'] for v in fresh]) if fresh else None,
                peak_allocated_gib=distribution([v['answer']['observed']['peak_allocated_gib'] for v in fresh]) if fresh else None,
                parse_failures=[q for q,v in bank.items() if v['answer']['parsed_answer'] is None],
                frame_counts=dict(Counter(str(len(v['selection']['selected_frames'])) for v in bank.values())))
        pairs=[(method.PARENT,m) for m in method.CONTROLS]+[(method.GLOBAL,method.WFS)]
        comparisons[profile]={a+'__minus__'+b:compare(data,groups[a],groups[b]) for a,b in pairs}
        from videoqa_audit.region_scope_runtime import parent_result
        for m in ('BASE-Uniform','BASE-BLIP-TopK','RD-1.2-exp07','RD-1.2-exp12'):
            paths=[parent_result(profile,r['question_id'],m) for r in data]
            values=[study.read(p) for p in paths]
            background[m]=dict(n=200,correct=sum(v['correct'] for v in values),
                scope='Background only: different selection domain/fine-pool eligibility')
            source_hashes.update({str(p):study.sha(p) for p in paths})
        profiles[profile]=dict(metrics=metrics,performance=performance,background=background,acceptance=acceptance)
    if total>800 or synthetic>8: raise ValueError('Global fixed call budget exceeded')
    # 步骤2不以200题分数自动触发全量；名义配对区间不证明模块独立因果效应。
    value=dict(status='completed',profiles=profiles,comparisons=comparisons,new_real_qa=total,
        synthetic_qa=synthetic,source_result_hashes=source_hashes,
        limitation='Observed development samples; whole-selection contrast on coarse pool; no automatic full QA',
        bootstrap=dict(repetitions=10000,seed=2027,stratification='duration',cluster='video'),
        automatic_next_experiment=False)
    study.freeze(root/'summary.json',value)
    entries=panel_entries(data,comparisons)
    study.freeze(root/'review/panel.json',dict(entries=entries,maximum_questions=8,minutes_limit=45,
        semantic_review='pending',new_model_calls=0))
    lines=['# 同粗池选择组织净价值诊断','',
           '| 配置 | 方法 | 正确／200 | 准确率 |','|---|---|---:|---:|']
    for p in study.PROFILES:
        for m in all_methods:
            v=profiles[p]['metrics'][m]['development200'];lines.append(f"| {p} | {m} | {v['correct']} | {100*v['accuracy']:.2f}% |")
    lines+=['','| 配置 | 配对 | 改善／退化 | Δ百分点 | 视频簇95%区间 |','|---|---|---:|---:|---|']
    for p,cs in comparisons.items():
        for name,v in cs.items():
            lo,hi=v['stratified_video_cluster_bootstrap_95']
            lines.append(f"| {p} | {name} | {len(v['cells']['left_only'])}／{len(v['cells']['right_only'])} | {v['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines+=['',f'新增正式QA：{total}；合成：{synthetic}。P导入、历史同输入导入和本轮共享分别统计。',
        '全量5400项父重放和10800项新选择只验收输入，不代表2700题已测试准确率。',
        '主比较为完整选择组织层，不把差值归因于单独去除区域、前缀或DP。',
        'REF-WFS-SB-S1使用本机BLIP-1分数/特征及固定官方选择器，不是作者默认BLIP-2完整复现。',
        '同输入引用不构成独立重复性检验。公开开发题已观察，结果只支持后续投入review。',
        '性能按缓存选择、CPU身份核验、解码和QA分别列示；共享答案只记一次物理成本，不称原片E2E。',
        '语义复核尚待完成；不得由图板或结构代理直接认定事实丢失。','']
    report=root/'report.md'; content='\n'.join(lines)
    if report.exists() and report.read_text()!=content: raise ValueError('Completed report differs')
    if not report.exists(): report.write_text(content)
    return value


def render_panel(root):
    """接口：在具备现有解码依赖的LLaVA环境渲染最多8题，不自动作语义判定。"""
    from videoqa_audit.region_scope_review import draw_board
    root=Path(root); entries=study.read(root/'review/panel.json')['entries']
    bank={r['question_id']:r for r in study.development_rows()}; page=['<!doctype html><meta charset="utf-8">',
        '<title>同粗池选择组织复核</title><style>body{max-width:1400px;margin:auto;font:16px/1.6 sans-serif}img{max-width:100%}</style>',
        '<h1>同粗池选择组织复核</h1><p>事后审核；每题至多两个10秒无声窗口，累计最多45分钟。结构差异不是语义证据损失证明。</p>']
    for entry in entries:
        q=entry['question_id'];p=entry['profile'];row=bank[q]
        page+=['<h2>'+html.escape(q+' '+p+' '+entry['kind'])+'</h2>',
               '<p>'+html.escape(row['question'])+'</p><p>'+html.escape(' / '.join(row['options']))+'</p>']
        for m in (method.PARENT,entry['control']):
            path=root/'qa'/p/'results'/m/(q+'.json')
            draw_board((str(root),q,p,m,str(path)))
            relative=p+'/'+q+'/'+m+'.jpg'
            page.append('<p>'+html.escape(m)+'</p><a href="'+relative+'"><img src="'+relative+'"></a>')
    target=root/'review/index.html';content='\n'.join(page)
    if not target.exists(): target.write_text(content)
    elif target.read_text()!=content: raise ValueError('Frozen panel page changed')
    return entries
