"""单一参考协议均匀基线：全量复核与旧协议配对，不追逐指定分数。"""
import csv
from pathlib import Path
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import durable,utc,rows
from videoqa_runtime.baseline_report import wilson,distribution
from videoqa_full.report import compare
from videoqa_lmms_bridge.common import load_task
from videoqa_runtime.protocol import parse_answer


def summarize(run):
    """接口：要求2700唯一成功结果及账本齐全；框架聚合与独立计数必须一致。"""
    from .run import validate,METHOD,BRIDGE
    run=Path(run);spec=read_json(run/'protocol.json');fp=sha256(run/'protocol.json');data=rows()
    paths=list((run/'results').glob('*.json'));calls=list((run/'calls'/METHOD).glob('*.json'))
    assert len(paths)==len(calls)==2700
    assert {p.stem for p in paths}=={p.stem for p in calls}=={r['question_id'] for r in data}
    result={r['question_id']:validate(run,r,spec,fp) for r in data}
    metrics=[r['result']['answer']['framework_metric'] for r in result.values()]
    native=load_task().videomme_aggregate_results(metrics);correct=sum(r['correct'] for r in result.values())
    assert abs(native-100*correct/2700)<1e-10
    oldpath=ROOT/'outputs/full_videoqa/rd_1_2__videomme2700__crossmodel_20260922__r01/historical_baselines.json'
    history=read_json(oldpath);old={r['question_id']:r for r in history if r['method']==METHOD}
    assert len(old)==2700
    # 步骤1：报告分时长、解析/截断以及上下文Token，不按答错自动重答。
    table={}
    for layer in ('all','short','medium','long'):
        group=[r for r in result.values() if layer=='all' or r['stratum']==layer];c=sum(r['correct'] for r in group)
        table[layer]=dict(correct=c,n=len(group),accuracy=c/len(group),wilson_95=wilson(c,len(group)))
    paired=compare(data,result,old)
    summary=dict(status='completed',protocol_sha256=fp,method=METHOD,accuracy=table,framework_accuracy_percent=native,
        old_protocol_comparison=paired,comparison_kind='combined protocol and source sampling change, not algorithm gain',
        paper_reference_percent=60.6,paper_exact_run_reproduced=False,
        parse_failures=[q for q,r in result.items() if not r['result']['answer']['parsed_answer']],
        strict_parser_differences=[q for q,r in result.items() if parse_answer(r['result']['answer']['raw_output'])!=r['result']['answer']['parsed_answer']],
        generation_limit_hits=[q for q,r in result.items() if len(r['result']['observed']['generated_token_ids'])==16],
        answer_calls=len(calls),failures=[str(p.relative_to(run)) for p in (run/'failures').glob('*.json')],
        recovered=[q for q,r in result.items() if r['reused_answer']],
        qa_seconds=distribution([r['result']['observed']['qa_seconds'] for r in result.values() if not r['reused_answer']]),
        created_at=utc())
    durable(run/'summary.json',summary)
    # 步骤2：保存所有预测及得失，不只展示更接近60.6%的个例。
    flat=[]
    for row in data:
        q=row['question_id'];r=result[q];a=r['result']['answer'];o=r['result']['observed']
        flat.append(dict(question_id=q,video_id=row['video_id'],stratum=row['stratum'],raw_output=a['raw_output'],
            parsed_answer=a['parsed_answer'],reference_answer='ABCD'[row['answer_index']],correct=r['correct'],
            old_answer=old[q]['parsed_answer'],old_correct=old[q]['correct'],source_frames=len(r['selection']['selected_frames']),
            visual_tokens=o['visual_tokens'],prefill_tokens=o['prefill_tokens'],generated_tokens=len(o['generated_token_ids'])))
    with (run/'per_question.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=['# LMMs-Eval参考协议：全量均匀基线','', '| 分组 | 正确数 | 准确率 |','|---|---:|---:|']
    for name,r in table.items():lines.append(f"| {name} | {r['correct']}/{r['n']} | {100*r['accuracy']:.3f}% |")
    lines+=['',f"旧协议基线1572/2700。本轮相对旧协议{paired['difference_pp']:+.3f}个百分点，新增正确{len(paired['cells']['left_only'])}、退化{len(paired['cells']['right_only'])}。",
        '差异同时包含原生均匀位置和问答协议变化，不是选帧算法收益。',
        '论文60.6%为公开对照值，缺少作者该次运行的完整命令/逐题日志，不能凭四舍五入后接近宣称精确复现。',
        '本轮只执行一次固定配置，未按分数调参或自动重答。',
        '输入位置来自已核验缓存，记录的是最终解码/问答成本，不是旧协议的原片全扫描E2E。']
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    print(table,flush=True)
    return summary
