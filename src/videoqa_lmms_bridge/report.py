"""桥接完成审计与结论：只解释协议组合，不把10题当正式算法成绩。"""
import csv
from pathlib import Path
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import durable,utc
from .common import QUESTION_IDS,CONDITIONS,question_rows,load_task,task_doc,GENERATION,SOURCE


def summarize(run):
    """接口：检查20调用、10对RGB与真实prefill，结合900视频输入差异出报告。"""
    run=Path(run);spec=read_json(run/'bridge_protocol.json');fp=sha256(run/'bridge_protocol.json')
    calls=list((run/'diagnostic/calls').glob('*/*.json'));assert len(calls)==20
    assert {(p.stem,p.parent.name) for p in calls}=={(q,c) for q in QUESTION_IDS for c in CONDITIONS}
    paths=list((run/'bridge_results').glob('*.json'));assert len(paths)==10
    assert {p.stem for p in paths}==set(QUESTION_IDS)
    task=load_task();records=[];flat=[];metrics={c:[] for c in CONDITIONS}
    # 步骤1：使用框架原任务process_results再计分；严格解析器只作旁路诊断。
    from videoqa_runtime.protocol import parse_answer
    for row in question_rows():
        q=row['question_id'];r=read_json(run/'bridge_results'/(q+'.json'))
        assert r['protocol_sha256']==fp and r['question']==row['question'] and r['options']==row['options']
        assert r['source_selection_sha256']==spec['source_selections'][q]['selection_sha256']
        inputs=read_json(run/'bridge_inputs'/(q+'.json'));assert inputs['rgb_sha256']==r['rgb_sha256']
        pair={}
        for condition in CONDITIONS:
            call=read_json(run/'diagnostic/calls'/condition/(q+'.json'))
            assert call['status']=='completed' and call['protocol_sha256']==fp
            assert call['generation_state']=={'started':True,'returned':True}
            assert call['answer']==r['conditions'][condition]
            item=r['conditions'][condition];a=item['answer'];o=item['observed']
            assert o['generate_calls']==o['prefill_calls']==1 and o['checked_logit_steps']>0
            assert o['max_new_tokens']==(8 if condition=='old' else 16)
            assert o['pixel_sha256']==o['expected_pixel_sha256'] and o['visual_tokens']==3360
            assert len(o['generated_token_ids'])<=o['max_new_tokens']
            scored=task.videomme_process_results(task_doc(row),[a['raw_output']])['videomme_perception_score']
            metrics[condition].append(scored)
            if condition=='reference':assert scored==a['framework_metric']
            pair[condition]=scored['pred_answer'];strict=parse_answer(a['raw_output'])
            flat.append(dict(question_id=q,stratum=row['stratum'],condition=condition,raw_output=a['raw_output'],
                primary_answer=scored['pred_answer'],strict_answer=strict,reference_answer=r['reference_answer'],
                correct=scored['pred_answer']==r['reference_answer'],generated_tokens=len(o['generated_token_ids']),
                generation_limit=o['max_new_tokens'],hit_token_limit=len(o['generated_token_ids'])==o['max_new_tokens'],
                text_tokens=o['text_input_tokens'],visual_tokens=o['visual_tokens'],prefill_tokens=o['prefill_tokens'],
                pixel_sha256=o['pixel_sha256'],repeat_agreement=r['repeat_agreement']))
        assert r['conditions']['old']['observed']['pixel_sha256']==r['conditions']['reference']['observed']['pixel_sha256']
        assert r['repeat_agreement']==(r['conditions']['old']['answer']['parsed_answer']==r['historical_answer'])
        r['primary_answers']=pair;records.append(r)
    # 步骤2：聚合也调用原任务函数；结果不能用来调提示词或选择最高分配置。
    scores={c:task.videomme_aggregate_results(items) for c,items in metrics.items()}
    uniform=read_json(run/'uniform_summary.json');assert uniform['videos']==900 and uniform['new_model_calls']==0
    offline_spec=read_json(run/'uniform_protocol.json')
    for name,h in offline_spec['code_sha256'].items():assert sha256(run/'offline_code_snapshot'/name)==h
    for vid,h in uniform['record_hashes'].items():
        path=run/'uniform_records'/(vid+'.json');assert sha256(path)==h
        native=read_json(path);assert sha256(run/native['pts_file'])==native['pts_sha256']
        assert native['new_count']==min(16,native['source_frame_count'])
        assert len({f['source_pts'] for f in native['new_frames']})==native['new_count']
    agreement=[r['question_id'] for r in records if not r['repeat_agreement']]
    changed=[r['question_id'] for r in records if r['primary_answers']['old']!=r['primary_answers']['reference']]
    # 步骤3：Qwen只有合成记录；缺少合成验收不能把统一入口宣称全部就绪。
    qpath=run/'qwen_framework_synthetic.json';qwen=read_json(qpath) if qpath.exists() else {'status':'missing'}
    status='completed' if qwen['status']=='passed' else 'llava_bridge_complete_qwen_synthetic_pending'
    summary=dict(status=status,created_at=utc(),protocol_sha256=fp,real_generations=len(calls),
        scores_diagnostic_only=scores,changed_question_ids=changed,repeatability_disagreements=agreement,
        attribution_excluded=agreement,native_uniform=uniform,qwen_synthetic_status=qwen['status'],
        token_limit_hits=[dict(question_id=r['question_id'],condition=r['condition']) for r in flat if r['hit_token_limit']],
        parser_disagreements=[dict(question_id=r['question_id'],condition=r['condition']) for r in flat if (r['primary_answer'] or None)!=r['strict_answer']],
        all_visual_inputs_identical=True,all_visual_tokens=3360,new_full_experiment_started=False,
        limitations=['10 fixed diagnostic questions, not a new algorithm score',
            'New prompt, omitted time instruction, output limit and framework stopping logic are a combined protocol intervention',
            'Native uniform source changes were not included in the 20 diagnostic generations',
            'Offline indexing is audited against its launch-time source snapshot; bridge task mapping was corrected in the later bridge snapshot'])
    durable(run/'bridge_summary.json',summary)
    with (run/'bridge_per_question.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# LMMs-Eval协议桥接结果','',f"状态：{status}。真实诊断生成{len(calls)}次，无全量问答。",'',
        f"旧/新协议诊断准确率为{scores['old']:.1f}% / {scores['reference']:.1f}%，仅描述本10题，不作为算法成绩。",
        f"同帧输入下预测变化题：{', '.join(changed) or '无'}；旧入口复测不一致题：{', '.join(agreement) or '无'}。",'',
        '## 输入对齐','', '10题两个入口RGB与BF16视觉Tensor一致；每题实际视觉Token均为3360。',
        f"900视频中，原生均匀与旧均匀完全一致{uniform['identical_videos']}个，平均源帧重合{uniform['mean_overlap']:.3f}张。",
        '新原生均匀属于独立采样协议，不能复用旧均匀的准确率作其基线。','',
        '## 逐题答案','', '| 题号 | 历史 | 旧入口复测 | 参考协议 | 标准 | 归因状态 |','|---|---|---|---|---|---|']
    for r in records:lines.append('| '+' | '.join(str(v) for v in (r['question_id'],r['historical_answer'],r['primary_answers']['old'],r['primary_answers']['reference'],r['reference_answer'],r['attribution_status']))+' |')
    lines+=['','## 边界与后续','']+['- '+v for v in summary['limitations']]
    lines+=['- 本轮不按诊断正确率决定协议去留；全量模型/方法/调用数另行确认，不自动启动。']
    (run/'bridge_report.md').write_text('\n'.join(lines)+'\n')
    return summary
