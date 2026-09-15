"""证据审计导出与机制对照：未标注帧保留pending，不以代理分数推断语义。"""
import csv
from collections import Counter
from pathlib import Path
from .core import read,write,selected,summarize,validate_annotation


def csv_file(path, rows):
    """导出UTF-8 BOM表格，供中文Excel读取；仅写本轮派生产物。"""
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def export(out):
    """导出50题、300组、4800槽位及B/C、A/B追踪；保留审核覆盖率。"""
    out=Path(out);questions=read(out/'questions.json');outcomes=read(out/'outcomes.json')
    source=Path(read(out/'protocol.json')['source']);frames=[];question_rows=[];changes=[];adaptation=[]
    scope_reviews=read(out/'scope_reviews.json').get('questions',{}) if (out/'scope_reviews.json').exists() else {}
    # 步骤1：逐槽位映射人工标注，组内覆盖与源帧标签严格分开。
    for q in questions:
        qid=q['question_id'];annotation=read(out/'annotations'/f'{qid}.json');validate_annotation(q,annotation)
        answer_map={v['method']:v for v in outcomes[qid].values()}
        row=dict(question_id=qid,stratum=q['stratum'],question=q['question'],scope=q['requirements']['scope'],
                 operation=q['requirements']['operation'],evidence_need=q['requirements']['need'],
                 scope_status='provisional',ambiguity=q['requirements'].get('ambiguity',''),review_status=annotation['status'])
        row['scope_suggested_after_view']=scope_reviews.get(qid,{}).get('suggested_scope','')
        row['scope_revision_reason']=scope_reviews.get(qid,{}).get('reason','')
        for method,r in answer_map.items():row[method+'_prediction']=r['prediction'];row[method+'_correct']=r['correct']
        question_rows.append(row)
        info={r['source_pts']:r for r in q['frames']}
        for alias,pts in q['groups'].items():
            group=annotation.get('groups',{}).get(alias,{})
            for slot,p in enumerate(pts,1):
                f=group.get('overrides',{}).get(str(p),annotation.get('frames',{}).get(str(p),{}));dup=group.get('redundancy',{}).get(str(p),{})
                frames.append(dict(question_id=qid,method=outcomes[qid][alias]['method'],alias=alias,slot=slot,
                                   source_pts=p,source_frame_index=info[p]['source_frame_index'],seconds=info[p]['timestamp_seconds'],
                                   label=f.get('label','pending'),reason=f.get('reason',''),duplicate=dup.get('duplicate'),
                                   new_information=dup.get('new_information'),review_status=annotation['status']))
        # 步骤2：计算集合差，不虚构旧帧到新帧的一对一替换关系。
        records={m:read(source/'results'/m/f'{qid}.json') for m in ('A','B','C')}
        sets={m:{r['source_pts'] for r in selected(records[m])} for m in records}
        pool=records['C']['pool'];windows=[(h['left_seconds'],h['right_seconds']) for h in pool['hotspot_trace'] if h['selected']]
        alias_by_method={v['method']:alias for alias,v in outcomes[qid].items()}
        for kind,pts,method in [('removed',sets['B']-sets['C'],'B'),('added',sets['C']-sets['B'],'C')]:
            for p in sorted(pts):
                t=info[p]['timestamp_seconds'];inside=any(l<=t<=r for l,r in windows)
                dist=min((max(l-t,0,t-r) for l,r in windows),default=None)
                group=annotation.get('groups',{}).get(alias_by_method[method],{})
                f=group.get('overrides',{}).get(str(p),annotation.get('frames',{}).get(str(p),{}))
                changes.append(dict(question_id=qid,kind=kind,source_pts=p,seconds=t,inside_hotspot=inside,distance_to_window=dist,
                                    label=f.get('label','pending'),reason=f.get('reason',''),review_status=annotation['status']))
        if sets['A']!=sets['B']:
            adaptation.append(dict(question_id=qid,quotas_changed=records['A']['pool']['quotas']!=records['B']['pool']['quotas'],
                                   A_quotas=records['A']['pool']['quotas'],B_quotas=records['B']['pool']['quotas'],
                                   removed_pts=sorted(sets['A']-sets['B']),added_pts=sorted(sets['B']-sets['A']),
                                   review_status=annotation['status'],correct_A=records['A']['correct'],correct_B=records['B']['correct']))
    # 步骤3：只报告已审核比例及标签分布；直接证据帧被移除不等于证据单元完全丢失。
    groups=read(out/'per_group.json')
    group_rows=[]
    for g in groups:
        group_rows.append({**{k:v for k,v in g.items() if k!='counts'},**{label:g['counts'].get(label,0) for label in ('direct','context','topic','irrelevant','uncertain','pending')}})
    csv_file(out/'per_question.csv',question_rows);csv_file(out/'per_frame.csv',frames);csv_file(out/'per_group.csv',group_rows)
    write(out/'bc_replacements.json',changes);write(out/'ab_adaptation.json',adaptation)
    removed=[x for x in changes if x['kind']=='removed']
    stats=dict(bc_removed=len(removed),bc_added=sum(x['kind']=='added' for x in changes),
               bc_removed_outside=sum(not x['inside_hotspot'] for x in removed),
               bc_removed_far=sum(x['distance_to_window'] is not None and x['distance_to_window']>1 for x in removed),
               removed_label_counts=dict(Counter(x['label'] for x in removed)),
               added_label_counts=dict(Counter(x['label'] for x in changes if x['kind']=='added')),
               ab_changed=len(adaptation),ab_quotas_changed=sum(x['quotas_changed'] for x in adaptation),
               warning='未审计项不参与语义结论；帧级标签增减不代表完整证据单元的因果损益。')
    write(out/'mechanism_audit.json',stats)
    lines=['# 首批10题逐组初审数量','',
           '仅首批原帧图板初审，尚待用户校准；384输入仅关键帧抽查。直接+上下文不能简单称为全部合理帧，重复信息单列于CSV。',
           '其余40题保持pending，不混入以下语义表。表中对错仍是原始计分，未调用模型。','']
    for q in questions:
        if not q['pilot']:continue
        lines.extend([f'## {q["question_id"]}', '', q['question'],'',
                      '| 方法 | 对错 | 直接 | 上下文 | 仅主题 | 无关 | 不确定 | 整体证据 |',
                      '|---|---|---:|---:|---:|---:|---:|---|'])
        for g in group_rows:
            if g['question_id']==q['question_id']:
                lines.append('| '+ ' | '.join(str(x) for x in [g['method'],'对' if g['correct'] else '错',g['direct'],g['context'],g['topic'],g['irrelevant'],g['uncertain'],g['sufficiency']])+' |')
        lines.append('')
    (out/'pilot_frame_counts.md').write_text('\n'.join(lines),encoding='utf-8')
    return stats
