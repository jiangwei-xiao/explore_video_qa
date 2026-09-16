"""汇合已审核的50题事实、300组标签、128条错误定位及冻结诊断；只读实验资产。"""
import argparse
from collections import Counter, defaultdict
import csv
import html
import re
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import read_json,write_json,sha256

METHODS=['uniform','topk','A','B','C','D']
LABELS=['direct','context','topic','irrelevant','uncertain']
LAYOUTS={'single_frame':'单帧','local_sequence':'局部连续过程','distributed_events':'跨时间分散','single_frame_summary':'多事件集中总结','uncertain':'暂不确定'}


def csv_write(path,rows):
    """以稳定字段顺序输出中文可读CSV，列表/字典显式序列化不丢信息。"""
    import json
    if not rows:raise ValueError('空统计表')
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def assemble(audit,diagnostic,rules_path):
    """核对冻结身份后组装最终审核资产；不根据输出重跑任何模型。"""
    audit=Path(audit);diagnostic=Path(diagnostic);rules=read_json(rules_path);out=audit/'final_review'
    if (out/'acceptance.json').exists():raise RuntimeError('最终审核已封存；请在新版本/副本重建，禁止覆盖当前交付。')
    out.mkdir(exist_ok=True);(out/'questions').mkdir(exist_ok=True)
    # 步骤1：原300条实验身份必须仍与审计建立时一致。
    original=read_json(audit/'protocol.json')['result_hashes']
    if len(original)!=300 or any(sha256(p)!=h for p,h in original.items()):raise ValueError('历史结果数量或身份改变')
    questions=read_json(audit/'questions.json');outcomes=read_json(audit/'outcomes.json')
    profiles={r['question_id']:r for r in csv.DictReader((audit/'full_review'/'question_profiles.csv').open(encoding='utf-8-sig'))}
    groups=[];errors=[];question_rows=[];category=defaultdict(lambda:{m:Counter() for m in METHODS});counts=Counter();common_wrong=[]
    # 步骤2：每个16帧集合显式保存角色、理由、重复维度、事实见证及整体充分性。
    for q in questions:
        qid=q['question_id'];a=read_json(audit/'annotations'/f'{qid}.json');v=read_json(audit/'full_review'/'visibility'/f'{qid}.json')
        seen={t for p in v['pages'].values() for t in p['source_pts']}
        if seen!={f['source_pts'] for f in q['frames']}:raise ValueError('384覆盖不闭合')
        for p in v['pages'].values():
            if sha256(audit/p['board_file'])!=p['sha256']:raise ValueError('384图板已改变')
        witness=read_json(audit/'full_review'/'witnesses_r6'/f'{qid}.json');profile=profiles[qid]
        layout=rules['layout_revisions'].get(qid,{}).get('layout',profile['evidence_layout'])
        operation=rules['operation_map'][profile['operation']]
        result={'question_id':qid,'question':q['question'],'options':q['options'],'source_video_sha256':q['source_video_sha256'],
                'operation':operation,'operation_subtype':profile['operation'],'evidence_layout':layout,
                'layout_revision':rules['layout_revisions'].get(qid),'required_evidence':profile['required_evidence'],
                'facts':witness['facts'],'visibility':v,'groups':{},'review_mode':'assistant_posthoc_initial_not_gold'}
        all_correct=all(x['correct'] for x in outcomes[qid].values());all_wrong=not any(x['correct'] for x in outcomes[qid].values())
        partition='all_correct' if all_correct else 'all_wrong' if all_wrong else 'discordant';counts[partition]+=1
        if all_wrong:common_wrong.append(qid)
        question_rows.append(dict(question_id=qid,question=q['question'],operation=operation,subtype=profile['operation'],evidence_layout=layout,
            required_evidence=profile['required_evidence'],partition=partition,judgment='AI初审；布局为当前视觉核查下的判断，非穷尽原片真值'))
        by_pts={f['source_pts']:f for f in q['frames']}
        for alias,pts in q['groups'].items():
            if len(pts)!=16 or len(set(pts))!=16:raise ValueError('输入不是16唯一帧')
            outcome=outcomes[qid][alias];m=outcome['method'];ga=a['groups'][alias];c=Counter();frames=[]
            for slot,t in enumerate(pts,1):
                frame_label={**a['frames'][str(t)],**ga.get('overrides',{}).get(str(t),{})};label=frame_label['label']
                if label not in LABELS:raise ValueError('未审核/非法标签')
                c[label]+=1
                frames.append(dict(slot=slot,frame=by_pts[t],annotation=frame_label,redundancy=ga['redundancy'][str(t)],
                    source_image=f'../../frames/{qid}/{t}.original.png',input_image=f'../../frames/{qid}/{t}.input.png'))
            g={'method':m,'alias':alias,**outcome,'frames':frames,'counts':{l:c[l] for l in LABELS},
               'sufficiency':ga['sufficiency'],'sufficiency_reason':ga['reason'],'facts':witness['groups'][alias]['facts']}
            if not outcome['correct']:
                rule=rules['error_rules'][qid];codes=rule['codes'];reason=rule['reason'];switch=rule.get('witness_switch')
                if switch and not g['facts'][switch['fact']]['count']:codes=switch['absent_codes'];reason=switch['absent_reason']
                g['error_localization']={'codes':codes,'reason':reason,'status':'reviewed_hypotheses_not_causal_proof',
                    'input_fact_counts':{fid:row['count'] for fid,row in g['facts'].items()},'evidence':f'questions/{qid}.json'}
                errors.append(dict(question_id=qid,method=m,prediction=outcome['prediction'],reference=outcome['reference'],codes=codes,
                    reason=reason,sufficiency=g['sufficiency'],input_fact_counts=g['error_localization']['input_fact_counts'],evidence=f'questions/{qid}.json'))
            result['groups'][m]=g;groups.append(dict(question_id=qid,method=m,prediction=outcome['prediction'],reference=outcome['reference'],correct=outcome['correct'],
                **g['counts'],sufficiency=g['sufficiency'],reason=g['sufficiency_reason']))
            for axis,value in [('operation',operation),('layout',layout)]:
                z=category[(axis,value)][m];z['n']+=1;z['correct']+=outcome['correct'];z['wrong']+=not outcome['correct']
        write_json(out/'questions'/f'{qid}.json',result)
    if len(questions)!=50 or len(groups)!=300 or len(errors)!=128 or dict(counts)!={'all_correct':24,'all_wrong':16,'discordant':10}:
        raise ValueError('历史问答统计不一致')
    # 步骤3：单独核实16次诊断清单冻结、开始状态、配对同卡及提示词，不追加调用。
    manifest=read_json(diagnostic/'manifest.json');freeze=read_json(diagnostic/'freeze.json');fingerprint=sha256(diagnostic/'manifest.json')
    if fingerprint!=freeze['manifest_sha256']:raise ValueError('诊断清单被改')
    calls=[read_json(p) for p in (diagnostic/'calls').glob('*.json')]
    if len(calls)!=16 or len(calls)>20:raise ValueError('诊断预算不符')
    indexed={(r['question_id'],r['variant']):r for r in calls};diagnostic_rows=[]
    for case in manifest['cases']:
        qid=case['question_id'];a=indexed[qid,'original'];b=indexed[qid,'repaired']
        if a['gpu']!=b['gpu']:raise ValueError('配对不同GPU')
        if a['answer']['parsed_answer']!=case['historical_prediction']:raise ValueError('原输入不稳定，应暂停归因')
        for variant,r in [('original',a),('repaired',b)]:
            if r['status']!='completed' or r['manifest_sha256']!=fingerprint or not r['generation_state'].get('returned'):raise ValueError('有不确定生成')
            if r['started_utc']<=manifest['frozen_utc']:raise ValueError('诊断先于冻结')
            if r['answer']['visual_tokens']!=3360 or r['answer']['pixel_shape']!=[16,3,384,384]:raise ValueError('输入协议异常')
            expected=case[variant+'_frames'];actual=r['selected_frames']
            if actual!=expected or len({f['source_pts'] for f in actual})!=16:raise ValueError('诊断帧不匹配冻结')
            if case['question'] not in r['answer']['prompt'] or any(o not in r['answer']['prompt'] for o in case['options']):raise ValueError('题面选项变化')
            if variant=='original' and r['answer']['prompt']!=case['historical_prompt']:raise ValueError('原提示词不一致')
        if len(set(f['source_pts'] for f in case['original_frames'])-set(f['source_pts'] for f in case['repaired_frames']))>4:raise ValueError('替换超过4')
        expected_order=['original','repaired'] if case['order_index']%2==0 else ['repaired','original']
        if indexed[qid,expected_order[0]]['started_utc']>=indexed[qid,expected_order[1]]['started_utc']:raise ValueError('交替顺序不符')
        diagnostic_rows.append(dict(question_id=qid,method=case['method'],category=case['stratum'],original=a['answer']['parsed_answer'],repaired=b['answer']['parsed_answer'],
            original_correct=a['correct'],repaired_correct=b['correct'],improvement=int(b['correct'])-int(a['correct']),original_repeat_stable=True,replacements=len(case['remove_slots'])))
    # 步骤4：分类表分别给分母/正确/错误/错误率，布局小类不得被样本数掩盖。
    category_rows=[]
    for (axis,value),methods in category.items():
        for m,z in methods.items():category_rows.append(dict(axis=axis,category=value,method=m,n=z['n'],correct=z['correct'],wrong=z['wrong'],error_rate=z['wrong']/z['n']))
    csv_write(out/'questions.csv',question_rows);csv_write(out/'groups.csv',groups);csv_write(out/'errors.csv',errors);csv_write(out/'categories.csv',category_rows);csv_write(out/'diagnostics.csv',diagnostic_rows)
    write_json(out/'attribution_rules.json',rules);write_json(out/'common_failure_ids.json',common_wrong)
    summary={'questions':50,'groups':300,'slots':4800,'errors':128,**counts,'diagnostic_started_calls':len(calls),
        'improved_pairs':sum(r['improvement']>0 for r in diagnostic_rows),'regressed_pairs':sum(r['improvement']<0 for r in diagnostic_rows),
        'historical_result_hashes_verified':len(original),'operation_categories':len({r['operation'] for r in question_rows}),
        'user_quantitative_agreement':None,'status':'assembled_pending_final_review','rules_sha256':sha256(rules_path)}
    write_json(out/'summary.json',summary)
    lines=['# 按独立任务操作和证据布局的历史错误率','','每格为错误数/题数（错误率），不按分类器LOCAL/GLOBAL标签分组。题数小，不宣称稳定能力差异。','']
    for axis in ['layout','operation']:
        lines += [f'## {axis}','','| 类别 | 均匀 | Top-K | A | B | C | D |','|---|---|---|---|---|---|---|']
        for (a,value),methods in category.items():
            if a!=axis:continue
            cells=[f"{methods[m]['wrong']}/{methods[m]['n']}（{100*methods[m]['wrong']/methods[m]['n']:.1f}%）" for m in METHODS]
            lines.append('| '+LAYOUTS.get(value,value)+' | '+' | '.join(cells)+' |')
        lines.append('')
    (out/'category_report.md').write_text('\n'.join(lines),encoding='utf-8');print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);p.add_argument('--diagnostic',required=True);p.add_argument('--rules',required=True)
    a=p.parse_args();assemble(a.audit,a.diagnostic,a.rules)
