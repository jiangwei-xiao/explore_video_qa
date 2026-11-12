"""H−P为主的固定配额局部对照统计；不自动接续新实验。"""
from collections import Counter
from pathlib import Path
import html
from . import fixed_quota_local_mmr as method
from . import fixed_quota_local_mmr_run as study


def summarize(root):
    """接口：两配置200题及排除旧8题的192题配对统计，来源逐项绑定。"""
    from videoqa_full.report import compare
    from videoqa_runtime.baseline_report import wilson,distribution
    from . import fixed_quota_local_mmr_runtime as runtime
    root=Path(root);data=study.development_rows();profiles={};sources={};panel=[]
    fixed=[('llava8','128-1'),('llava8','306-2'),('qwen32','383-2'),
           ('qwen32','597-1'),('qwen32','691-3'),('qwen32','705-1')]
    used={q.split('-')[0] for _,q in fixed};improved_videos=set();gates=[]
    # 步骤1H/P读取本轮结果；G/W仅引用旧固定输入实测，不混入完整RD背景。
    for profile in study.PROFILES:
        run=root/'qa'/profile;acceptance=runtime.audit(run);spec=study.read(run/'protocol.json')
        banks={}
        for m in (method.CONTROL,method.PARENT,method.GLOBAL,method.WFS):
            base=run if m in (method.CONTROL,method.PARENT) else study.PREVIOUS/'qa'/profile
            paths={r['question_id']:base/'results'/m/(r['question_id']+'.json') for r in data}
            banks[m]={q:study.read(p) for q,p in paths.items()};sources.update({str(p):study.sha(p) for p in paths.values()})
        subset=[r for r in data if r['question_id'] not in study.OLD_PANEL]
        if len(subset)!=192: raise ValueError('Supplementary denominator differs')
        metrics={};performance={};comparisons={}
        for m,bank in banks.items():
            scopes=dict(all200=data,excluding_previous_panel192=subset)
            scopes.update({s:[r for r in data if r['stratum']==s] for s in ('short','medium','long')})
            scopes.update({'task/'+t:[r for r in data if r['task_type']==t] for t in sorted({r['task_type'] for r in data})})
            if m==method.CONTROL:
                for kind in sorted({v['timing_kind'] for v in bank.values()}):
                    scopes['input_kind/'+kind]=[r for r in data if bank[r['question_id']]['timing_kind']==kind]
            metrics[m]={}
            for scope,rs in scopes.items():
                n=len(rs);c=sum(bank[r['question_id']]['correct'] for r in rs)
                if n: metrics[m][scope]=dict(n=n,correct=c,accuracy=c/n,wilson_95=wilson(c,n))
            fresh=[v for v in bank.values() if v['timing_kind']=='new_physical_call']
            performance[m]=dict(kinds=dict(Counter(v['timing_kind'] for v in bank.values())),
                qa_seconds=distribution([v['answer']['observed']['qa_seconds'] for v in fresh]) if fresh else None,
                peak_allocated_gib=distribution([v['answer']['observed']['peak_allocated_gib'] for v in fresh]) if fresh else None,
                parse_failures=[q for q,v in bank.items() if v['answer']['parsed_answer'] is None],
                actual_frames=dict(Counter(str(len(v['selection']['selected_frames'])) for v in bank.values())))
        for other in (method.PARENT,method.GLOBAL,method.WFS):
            comparisons[other]=dict(all200=compare(data,banks[method.CONTROL],banks[other]),
                excluding_previous_panel192=compare(subset,banks[method.CONTROL],banks[other]))
        from videoqa_audit.region_scope_runtime import parent_result
        background={}
        for m in ('BASE-Uniform','BASE-BLIP-TopK','RD-1.2-exp07','RD-1.2-exp12'):
            paths=[parent_result(profile,r['question_id'],m) for r in data]
            background[m]=dict(n=200,correct=sum(study.read(p)['correct'] for p in paths),scope='Background only; full/fine domain differs')
            sources.update({str(p):study.sha(p) for p in paths})
        primary=comparisons[method.PARENT];cells=primary['all200']['cells']
        outside=[q for q in cells['left_only'] if q not in study.OLD_PANEL]
        improved_videos.update(q.split('-')[0] for q in outside)
        regress=set(cells['right_only']);gain=set(cells['left_only'])
        extra=next((r for r in data if r['question_id'] in regress and r['question_id'] not in study.OLD_PANEL
                    and r['video_id'] not in used),None)
        if extra is None and not regress:
            extra=next((r for r in data if r['question_id'] in gain and r['question_id'] not in study.OLD_PANEL
                        and r['video_id'] not in used),None)
        if extra:
            used.add(extra['video_id']);panel.append(dict(profile=profile,question_id=extra['question_id'],kind='new_regression' if regress else 'new_improvement'))
        gates.append((primary['all200']['difference_pp']>=0,primary['excluding_previous_panel192']['difference_pp']>=0,
                      len(cells['left_only'])-len(cells['right_only'])))
        profiles[profile]=dict(metrics=metrics,performance=performance,comparisons=comparisons,
            background=background,acceptance=acceptance,preparation_wall_seconds=spec['preparation_wall_seconds'],
            execution=study.read(run/'execution.json'))
    # 步骤2资源门槛仅建议后续review；不自动新增版本、参数或全量问答。
    gate=all(a and b for a,b,_ in gates) and max(c for _,_,c in gates)>=4 and len(improved_videos)>=3
    value=dict(status='completed',profiles=profiles,source_result_hashes=sources,
        recommendation_gate=bool(gate),outside_design_improvement_videos=sorted(improved_videos),
        bootstrap=dict(repetitions=10000,seed=2027,stratification='duration',cluster='video'),
        automatic_next_experiment=False,limitation='Fixed parent allocations; coarse-only; observed development data',
        new_real_qa=sum(v['acceptance']['new_real_qa'] for v in profiles.values()),synthetic_qa=8)
    study.freeze(root/'summary.json',value)
    entries=[dict(profile=p,question_id=q,kind='existing_material') for p,q in fixed]+panel
    study.freeze(root/'review/panel.json',dict(entries=entries,maximum_questions=8,minutes_limit=45,semantic_review='pending'))
    lines=['# 固定区域与实际配额局部MMR对照','', '| 配置 | P | H | G | W | H−P题数 |','|---|---:|---:|---:|---:|---:|']
    for p,v in profiles.items():
        counts=[v['metrics'][m]['all200']['correct'] for m in (method.PARENT,method.CONTROL,method.GLOBAL,method.WFS)]
        lines.append('| '+p+' | '+' | '.join(map(str,counts))+f' | {counts[1]-counts[0]:+d} |')
    lines+=['',f'新问答 {value["new_real_qa"]}；合成8。固定投入建议门：{gate}。',
        '完整配对四格题号、分层区间、192题补充与背景见summary.json。',
        '本轮只替换父实际资源分配下的整个局部规则；没有取消贪心前缀，没有重新求解DP。',
        '收益不代表可直接处理无ITM细帧或已得到独立可部署算法；成本为缓存诊断，不是原片E2E。',
        '有界语义复核另行记录，单一反例不修改规则。']
    dest=root/'report.md';content='\n'.join(lines)+'\n'
    if not dest.exists(): dest.write_text(content)
    elif dest.read_text()!=content: raise ValueError('Frozen report changed')
    return value


def render_panel(root):
    """接口：只渲染事前固定及最多两道新增案例，不自动给出语义结论。"""
    from videoqa_audit.region_scope_review import draw_board
    root=Path(root);entries=study.read(root/'review/panel.json')['entries'];rows={r['question_id']:r for r in study.development_rows()}
    page=['<!doctype html><meta charset="utf-8"><title>固定配额局部MMR复核</title>',
          '<style>body{max-width:1500px;margin:auto;font:16px/1.6 sans-serif}img{max-width:100%}</style>',
          '<h1>最多8题、45分钟；事后AI初审</h1>']
    for e in entries:
        p,q=e['profile'],e['question_id'];row=rows[q]
        page+=['<h2>'+html.escape(p+' '+q+' '+e['kind'])+'</h2>',
               '<p>'+html.escape(row['question'])+'</p><p>'+html.escape(' / '.join(row['options']))+'</p>']
        for m in (method.PARENT,method.CONTROL):
            path=root/'qa'/p/'results'/m/(q+'.json');draw_board((str(root),q,p,m,str(path)))
            rel=p+'/'+q+'/'+m+'.jpg';page.append('<p>'+m+'</p><a href="'+rel+'"><img src="'+rel+'"></a>')
    out=root/'review/index.html';content='\n'.join(page)
    if not out.exists():out.write_text(content)
    elif out.read_text()!=content:raise ValueError('Frozen panel changed')
    return entries
