"""全量证据定位汇总：统计人工标签、真实选择轨迹和已冻结诊断，不触发模型。"""
import argparse
import csv
from collections import Counter,defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_audit.core import read,write,digest,selected,validate_annotation

PILOT_PROFILE={
 '264-2':('recognition','single_frame'),'872-2':('characterization','distributed_events'),
 '004-3':('comparison','distributed_events'),'395-2':('anchored_outcome','distributed_events'),
 '491-2':('ordering','single_frame_summary'),'538-1':('causal','single_frame_summary'),
 '012-3':('color','single_frame'),'383-2':('analogy','single_frame'),
 '260-2':('ordering','local_sequence'),'844-1':('ordering','distributed_events')}

ERRORS={
 '844-1':('relationship_missing','四事件与年代顺序未完整对应'),
 '395-2':('selection_omission|relationship_missing','读书高分，宣传册低分；补宣传册后仍可能只理解为阅读，不能据此认定模型没有关系能力'),
 '815-2':('relationship_missing|uncertain','用环方式可见，节目序号尚未定位'),
 '871-2':('uncertain','玩具身份与兴趣比较判据未确定'),
 '304-1':('relationship_missing|uncertain','重复姿态/提问文字缺原因解释'),
 '170-2':('relationship_missing|model_or_evidence_limit','起跑与过线/回放需区分；增加过线帧后仍错，不追加修复'),
 '158-3':('evidence_present_answer_wrong','原帧有比分时间，需保留输入可读性与推理两种可能'),
 '705-1':('entity_attribution|uncertain','署名与Doctor标题可能被混淆，职业独立证据仍待核实'),
 '306-2':('selection_omission','原候选88/89含首次梳头，原B未选'),
 '443-1':('relationship_missing|selection_omission','多选后续局次；第一局三分尝试与回放未完整对应'),
 '736-2':('uncertain','专业策略依据尚无法从现有视觉证据确认'),
 '103-1':('selection_omission','原候选5含叉子，B未选而保留后续筷子'),
 '316-3':('relationship_missing|uncertain','人物介绍重复，缺改信与免死因果'),
 '870-1':('uncertain','出现任务卡不等于完成；未完成状态尚待确认'),
 '204-3':('uncertain','坐姿/站姿不等价，书页人物数值对应未核实'),
 '601-1':('uncertain','全片主线及标准选项解释尚有疑点，不擅自改计分'),
 '004-3':('selection_omission|uncertain','C丢目标阶段名，但A保留仍错；快速变化判据待复核'),
 '491-2':('annotation_ambiguity','原题两个事件名称重复'),
 '128-1':('evidence_present_answer_wrong|uncertain','B/C仍有干裂地表，不能称完全漏旱灾；法国/最严重限定不完整'),
 '538-1':('selection_omission','D缺规则文字；加入规则后答对'),
 '544-1':('relationship_missing|uncertain','摩托车离开时间、出门与路过者尚有边界问题'),
 '012-3':('selection_omission','Top-K缺清晰灰胶带；加入后答对'),
 '383-2':('selection_omission|relationship_missing','ABC漏船只等号；均匀有潜水艇仍错，不作单一归因'),
 '893-1':('relationship_missing|uncertain','全局主线与消费背景不完整，尚非完整原片证据'),
 '693-1':('selection_omission','均匀缺Village Defense招募牌'),
 '260-2':('relationship_missing','A/B/C都有喷雾，缺刷牙锚点，C退化不等于丢喷雾')}


def output_csv(path,rows):
    """统一导出中文可读CSV；空集合不伪造记录。"""
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


def summarize(out,diagnostics):
    """仅读取所有人工记录与冻结结果，产生可复算全量统计和证据引用。"""
    out=Path(out);full=out/'full_review';diag=Path(diagnostics)
    qs=read(out/'questions.json');outcomes=read(out/'outcomes.json');source=Path(read(out/'protocol.json')['source'])
    observations=read(full/'full_observations.json');pixels={(r['question_id'],r['source_pts']):r for r in read(full/'pixel_statistics.json')}
    categories=defaultdict(lambda:defaultdict(lambda:dict(n=0,correct=0,wrong=0)))
    slot_counts=defaultdict(Counter);black=[];errors=[];profiles=[];unit_changes=[]
    # 步骤1：两条独立人工轴，不沿用实验分类器标签；未知问题保持未知。
    for q in qs:
        qid=q['question_id'];a=read(out/'annotations'/f'{qid}.json');validate_annotation(q,a)
        if a['status']!='initial_reviewed':raise ValueError('仍有未审核题，不能出全量报告')
        if qid in observations:op,layout=observations[qid]['operation'],observations[qid]['layout'];needs=observations[qid]['required_evidence']
        else:op,layout=PILOT_PROFILE[qid];needs=[q['requirements']['need']]
        profiles.append(dict(question_id=qid,operation=op,evidence_layout=layout,required_evidence='；'.join(needs),
                             confidence='assistant_initial_review',question=q['question']))
        records={}
        for m in ('uniform','topk','A','B','C','D'):
            records[m]=read(source/('baseline_reference/results' if m in ('uniform','topk') else 'results')/m/f'{qid}.json')
        for alias,pts in q['groups'].items():
            o=outcomes[qid][alias];m=o['method'];group=a['groups'][alias]
            for axis,key in [('operation',op),('layout',layout)]:
                stat=categories[axis+':'+key][m];stat['n']+=1;stat['correct']+=int(o['correct']);stat['wrong']+=int(not o['correct'])
            if not o['correct']:
                kind,reason=ERRORS[qid]
                errors.append(dict(question_id=qid,method=m,error_hypotheses=kind,reason=reason,
                                   sufficiency=group['sufficiency'],evidence=f'annotations/{qid}.json',certainty='hypothesis_not_causal_proof'))
            # 步骤2：像素阈值只识别待讨论的近黑子集；必须同时有人工无关标签才记为确认无效。
            pool=records[m].get('pool',records[m].get('selection'))
            scoring=pool if m!='uniform' else records['C']['pool']
            positions={f['source_pts']:i for i,f in enumerate(scoring['candidates'])}
            ranks={i:rank+1 for rank,i in enumerate(sorted(range(len(scoring['scores'])),key=lambda i:(-scoring['scores'][i],scoring['candidates'][i]['source_pts'],i)))}
            for slot,p in enumerate(pts,1):
                label=group.get('overrides',{}).get(str(p),a['frames'][str(p)])['label'];slot_counts[m][label]+=1
                px=pixels[(qid,p)]
                if px['near_black_pixel_fraction']>=.995:
                    i=positions[p];trace=next((t for t in pool.get('competition_trace',[]) if t['candidate_index']==i),None)
                    black.append(dict(question_id=qid,method=m,slot=slot,source_pts=p,label=label,
                        score_source='C_frozen_pool_posthoc_not_uniform_compute' if m=='uniform' else m,
                        confirmed_irrelevant=label=='irrelevant',relevance=scoring['scores'][i],rank=ranks[i],
                        near_black_fraction=px['near_black_pixel_fraction'],coarse_pick_step=None if trace is None else trace['step']+1,
                        segment_count_before=None if trace is None else trace['segment_count_before'],
                        duplicate_penalty=None if trace is None else trace['duplicate'],
                        hotspot=any(h.get('selected') and h['source_pts']==p for h in pool.get('hotspot_trace',[]))))
        # 步骤3：仅追踪人工明确命名的同信息簇；未合并的unique编号不冒充独立必要事实。
        def units(method):
            """按实际选帧查找明确命名的信息簇，避免把移除单帧等同丢失全部证据。"""
            return {a['frames'][str(f['source_pts'])]['evidence_unit'] for f in selected(records[method])
                    if a['frames'][str(f['source_pts'])]['label'] in ('direct','context') and not a['frames'][str(f['source_pts'])]['evidence_unit'].startswith('unique_')}
        for l,r in [('A','B'),('B','C')]:
            before,after=units(l),units(r)
            unit_changes.append(dict(question_id=qid,comparison=l+'-'+r,lost_named_clusters=sorted(before-after),gained_named_clusters=sorted(after-before),
                                     warning='同信息簇是人工代理分组，不代表所有必要事实已完整标注'))
    # 步骤4：导出分母、错误率和人工证据数量；诊断结果绝不合入六组旧准确率。
    for methods in categories.values():
        for stat in methods.values():stat['error_rate']=stat['wrong']/stat['n']
    output_csv(full/'question_profiles.csv',profiles);output_csv(full/'error_localization.csv',errors);output_csv(full/'near_black_trace.csv',black)
    write(full/'categories.json',dict(categories));write(full/'named_information_changes.json',unit_changes)
    diagnostic=read(diag/'summary.json');pairs=[p for p in diagnostic['pairs'] if p['status']=='paired']
    summary=dict(question_count=50,unique_frames=2235,qa_results=300,frame_slots=4800,unreviewed=0,
                 frame_labels_by_method={m:dict(c) for m,c in slot_counts.items()},
                 near_black_analysis_threshold='>=99.5% pixels maxRGB<=3; diagnostic only, not a filtering algorithm',
                 near_black_slots=len(black),confirmed_irrelevant_near_black_by_method=dict(Counter(r['method'] for r in black if r['confirmed_irrelevant'])),
                 wrong_results=len(errors),all_wrong_questions=16,
                 diagnostic=dict(calls=diagnostic['budget_used'],paired=len(pairs),improved=sum(p['improvement']>0 for p in pairs),
                                 regressed=sum(p['improvement']<0 for p in pairs),wrong_unchanged=sum(not p['old_correct'] and not p['new_correct'] for p in pairs),
                                 correct_unchanged=sum(p['old_correct'] and p['new_correct'] for p in pairs)),
                 limits=['AI初审，不是双人一致性真值','只核查记录内原片窗口，不证明未查区域无证据','384像素资产全量存在，人工可读性专项范围另列','目的性人工修复不等于算法提升'])
    write(full/'summary.json',summary)
    output_csv(diag/'paired_results.csv',diagnostic['pairs'])
    print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--diagnostics',required=True);a=p.parse_args();summarize(a.out,a.diagnostics)
