"""关联全部A/B及B/C换帧与人工事实见证，分开记录数值机制和语义证据，不运行模型。"""
import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import read_json,write_json,sha256


def fact_transitions(facts,before,after):
    """比较明确列出的事实见证集合；多张替代帧仍有一张时，不误记整个事实见证丢失。"""
    result=[]
    for fact in facts:
        all_pts={f['source_pts'] for f in fact['witnesses']};old=all_pts&before;new=all_pts&after
        state=('retained' if new else 'listed_witness_lost') if old else ('listed_witness_added' if new else 'not_witnessed_in_either')
        result.append(dict(fact_id=fact['id'],description=fact['description'],role=fact['role'],state=state,
            before_pts=sorted(old),after_pts=sorted(new),limitation=fact['limitation']))
    return result


def analyze(audit,source,witness_version='witnesses_r5',output_version='semantic_changes_r5'):
    """输入审计与冻结实验目录，输出50题、所有指定变化的帧/事实/配额关联报告。"""
    audit=Path(audit);source=Path(source);out=audit/'full_review'/output_version;out.mkdir(exist_ok=True)
    outcomes=read_json(audit/'outcomes.json');questions=read_json(audit/'questions.json');totals=Counter();details=[]
    # 步骤1：逐题关联原答案、人工标签、事实目录及历史候选池；不改任何原结果。
    for q in questions:
        qid=q['question_id'];annotations=read_json(audit/'annotations'/f'{qid}.json')
        witness=read_json(audit/'full_review'/witness_version/f'{qid}.json')
        stage=read_json(audit/'full_review'/'stage_attribution'/f'{qid}.json')
        records={m:read_json(source/'results'/m/f'{qid}.json') for m in 'ABC'}
        aliases={x['method']:a for a,x in outcomes[qid].items()}
        sets={m:set(q['groups'][aliases[m]]) for m in 'ABC'}
        result={'question_id':qid,'question':q['question'],'reasoning_limits':witness['reasoning_limits'],'comparisons':{}}
        # 步骤2：所有变化逐帧落盘，保存本帧事实及同事实剩余见证，不用“直接帧数量”替代证据单元。
        for before,after in [('A','B'),('B','C')]:
            key=before+after;changed=sets[before]!=sets[after]
            transitions=fact_transitions(witness['facts'],sets[before],sets[after])
            comp={'changed':changed,'before_prediction':outcomes[qid][aliases[before]],'after_prediction':outcomes[qid][aliases[after]],
                'facts':transitions,'removed':[],'added':[],
                'quotas_before':records[before]['pool']['quotas'],'quotas_after':records[after]['pool']['quotas']}
            comp['quotas_changed']=comp['quotas_before']!=comp['quotas_after']
            for direction,pts_list,method in [('removed',sets[before]-sets[after],before),('added',sets[after]-sets[before],after)]:
                pool=records[method]['pool'];positions={f['source_pts']:i for i,f in enumerate(pool['candidates'])}
                for pts in sorted(pts_list):
                    i=positions[pts];frame=pool['candidates'][i];label=annotations['frames'][str(pts)]
                    related=[t for t in transitions if pts in t['before_pts'] or pts in t['after_pts']]
                    role='no_catalog_fact_identified'
                    if related:role=';'.join(sorted({r['role'] for r in related}))
                    elif label['label'] in {'irrelevant','topic'}:role='no_answer_fact_identified_'+label['label']
                    refinement=stage['methods']['C']['refinement']['targets'].get(str(pts),{})
                    comp[direction].append(dict(frame=frame,candidate_index=i,source_method=method,score=pool['scores'][i],
                        segment_id=pool['segment_ids'][i],is_added_fine_candidate=i>=pool['initial_count'],
                        initial_frame_label=label['label'],initial_frame_reason=label['reason'],semantic_role=role,
                        related_fact_transitions=related,distance_to_C_hotspot_seconds=refinement.get('distance_to_hotspot_seconds'),
                        input_view=f'../../frames/{qid}/{pts}.input.png',
                        review_basis='已查看原帧与384图板；语义解释依人工事实目录，未命中目录不等同无效或不存在证据'))
            if len(comp['removed'])!=len(comp['added']):raise ValueError('16帧预算交换数不闭合')
            if key=='BC' and comp['quotas_changed']:raise ValueError('B/C冻结配额不一致，需暂停归因')
            comp['answer_information_lost_candidates']=[t for t in transitions if t['state']=='listed_witness_lost' and t['role'] in {'required','partial'}]
            comp['answer_information_added_candidates']=[t for t in transitions if t['state']=='listed_witness_added' and t['role'] in {'required','partial'}]
            comp['interpretation']='见证集合变化不自动证明必要事实完全丢失；候选信息增减需结合限制、原片与原/新答案，不作因果归因。'
            result['comparisons'][key]=comp
            totals[key+'_changed_questions']+=changed;totals[key+'_removed_frames']+=len(comp['removed']);totals[key+'_added_frames']+=len(comp['added'])
            totals[key+'_quota_changed_questions']+=comp['quotas_changed']
        result['first_AB_divergence']=stage['first_AB_divergence']
        write_json(out/f'{qid}.json',result);details.append(result)
    # 步骤3：验收要求的集合规模逐项核对；“50文件存在”不替代具体120进/120出闭合。
    if len(details)!=50 or totals['AB_changed_questions']!=33 or totals['BC_removed_frames']!=120 or totals['BC_added_frames']!=120:
        raise ValueError('指定变化集合与冻结实验不一致')
    summary={'questions':50,**totals,'model_calls':0,'facts_spec_sha256':sha256(audit/'full_review'/witness_version/'spec.json'),
        'warning':'统计明确列出的事实见证变化；不是独立人工金标准，不自动宣布机制因果或完整审计结束。'}
    # 步骤4：可读总表保留反例，按冻结题序不挑选最有利样本。
    lines=['# 全量A/B与B/C语义变化关联','','数值重放与人工可见事实分开；表内“失去/新增”仅指目录中该事实的全部已列见证不再保留/首次进入。不是答案改善因果。','','| 题号 | 比较 | 替换帧数 | 配额变 | 所列事实见证失去 | 所列事实见证新增 | 原答案→新答案 |','|---|---|---:|---|---|---|---|']
    for r in details:
        for key,comp in r['comparisons'].items():
            if not comp['changed']:continue
            lost='；'.join(t['description'] for t in comp['answer_information_lost_candidates']) or '无（限已列事实）'
            added='；'.join(t['description'] for t in comp['answer_information_added_candidates']) or '无（限已列事实）'
            a,b=comp['before_prediction'],comp['after_prediction']
            answer=f"{a['prediction']}({'对' if a['correct'] else '错'})→{b['prediction']}({'对' if b['correct'] else '错'})"
            lines.append(f"| [{r['question_id']}]({r['question_id']}.json) | {key} | {len(comp['removed'])} | {'是' if comp['quotas_changed'] else '否'} | {lost} | {added} | {answer} |")
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8');write_json(out/'summary.json',summary);print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);p.add_argument('--source',required=True)
    p.add_argument('--witness-version',default='witnesses_r5');p.add_argument('--output-version',default='semantic_changes_r5')
    a=p.parse_args();analyze(a.audit,a.source,a.witness_version,a.output_version)
