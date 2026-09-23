"""六组全量统计：视频簇配对bootstrap，区分开发题/开发视频与缓存问答成本。"""
import csv
from pathlib import Path
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_runtime.baseline_report import wilson,distribution
from .state import rows,METHODS,durable,utc,stage_methods


def cluster_interval(data,left,right,repetitions=10000):
    """接口：按官方时长分层，以视频为簇成对重抽；允许子集内每簇题数不同。"""
    rng=np.random.default_rng(2027);numerator=np.zeros(repetitions);denominator=np.zeros(repetitions)
    for layer in ('short','medium','long'):
        groups={}
        for row in data:
            if row['stratum']!=layer:continue
            q=row['question_id'];v=row['video_id']
            groups.setdefault(v,[]).append(int(left[q]['correct'])-int(right[q]['correct']))
        if not groups:continue
        values=list(groups.values());sums=np.array([sum(v) for v in values]);sizes=np.array([len(v) for v in values])
        picks=rng.integers(0,len(values),size=(repetitions,len(values)))
        numerator+=sums[picks].sum(axis=1);denominator+=sizes[picks].sum(axis=1)
    assert np.all(denominator>0)
    return np.percentile(100*numerator/denominator,[2.5,97.5]).tolist()


def compare(data,left,right):
    """保存全量配对四格及题号，不只展示有利变化。"""
    cells={k:[] for k in ('both_correct','left_only','right_only','both_wrong')}
    for row in data:
        q=row['question_id'];a,b=left[q]['correct'],right[q]['correct']
        key='both_correct' if a and b else 'left_only' if a else 'right_only' if b else 'both_wrong'
        cells[key].append(q)
    return dict(n=len(data),cells=cells,difference_pp=100*(len(cells['left_only'])-len(cells['right_only']))/len(data),
                stratified_video_cluster_bootstrap_95=cluster_interval(data,left,right))


def summarize(run):
    """接口：四组新结果完整校验后生成六组报告；缺失则停止，不补造答案。"""
    from .run import validate_result
    run=Path(run);spec=read_json(run/'protocol.json');fingerprint=sha256(run/'protocol.json');data=rows()
    historical=read_json(run/'historical_baselines.json')
    assert sha256(run/'historical_baselines.json')==spec['selection_exports']['historical_baselines.json']
    records={};call_counts={}
    for model in ('llava','qwen25vl'):
        for method in METHODS:
            key=model+'__'+method
            if model=='llava' and method!='RD-1.2':
                values=[r for r in historical if r['method']==method]
            else:
                paths=list((run/model/'results'/method).glob('*.json'));assert len(paths)==2700
                values=[validate_result(run,model,method,row,spec,fingerprint,deep=model=='llava') for row in data]
                calls=list((run/model/'calls'/method).glob('*.json'));assert len(calls)==2700
                call_counts[key]=len(calls)
            assert len(values)==len({r['question_id'] for r in values})==2700
            records[key]={r['question_id']:r for r in values}
    assert sum(call_counts.values())==10800
    for row in data:
        q=row['question_id']
        assert len({records['qwen25vl__'+m][q]['physical_gpu'] for m in METHODS})==1
    dev=read_json(ROOT/'data/manifests/video_mme_development_200_20260920.json')['rows']
    devq={r['question_id'] for r in dev};devv={r['video_id'] for r in dev}
    subsets={'all2700':data,'development200':[r for r in data if r['question_id'] in devq],
             'other2500':[r for r in data if r['question_id'] not in devq],
             'outside_development_videos2100':[r for r in data if r['video_id'] not in devv]}
    assert [len(v) for v in subsets.values()]==[2700,200,2500,2100]
    tables={};comparisons={};performance={};flat=[]
    # 步骤1：各组各子集按相同分母计分，解析失败列题号。
    for key,group in records.items():
        tables[key]={}
        for subset,subset_rows in subsets.items():
            tables[key][subset]={}
            for layer in ('all','short','medium','long'):
                items=[group[r['question_id']] for r in subset_rows if layer=='all' or r['stratum']==layer]
                if not items:continue
                correct=sum(r['correct'] for r in items)
                tables[key][subset][layer]=dict(n=len(items),correct=correct,accuracy=correct/len(items),wilson_95=wilson(correct,len(items)))
        values=list(group.values());primary=[r for r in values if r.get('timing_kind')!='resumed_not_primary']
        performance[key]={}
        for layer in ('all','short','medium','long'):
            items=[r for r in primary if layer=='all' or r['stratum']==layer]
            performance[key][layer]=dict(n=len(items),timings={},memory={})
            if items:
                for field in ('timings','memory'):
                    shared=set.intersection(*(set(r[field]) for r in items))
                    performance[key][layer][field]={k:distribution([r[field][k] for r in items]) for k in sorted(shared)}
        for row in data:
            r=group[row['question_id']];a=r.get('answer',r)
            flat.append(dict(model=key.split('__')[0],method=r['method'],question_id=row['question_id'],
                video_id=row['video_id'],stratum=row['stratum'],development_question=row['question_id'] in devq,
                development_video=row['video_id'] in devv,parsed_answer=a['parsed_answer'],reference_answer='ABCD'[row['answer_index']],
                correct=r['correct'],raw_output=a['raw_output'],timing_kind=r.get('timing_kind','historical_batch')))
    # 步骤2：主比较按视频簇进行配对，不能把同视频三题当成独立重复。
    for model in ('llava','qwen25vl'):
        for baseline in METHODS[:2]:
            name=model+'__RD-1.2_minus_'+baseline
            comparisons[name]={s:compare(rs,records[model+'__RD-1.2'],records[model+'__'+baseline]) for s,rs in subsets.items()}
    summary=dict(status='completed',created_at=utc(),protocol_sha256=fingerprint,accuracy=tables,comparisons=comparisons,
        performance=performance,answer_call_counts=call_counts,
        failures=[str(p.relative_to(run)) for p in run.glob('*/failures/*.json')],
        executions={m:read_json(run/m/'execution.json') for m in ('llava','qwen25vl')},
        parse_failures={key:[q for q,r in g.items() if r.get('answer',r)['parsed_answer'] is None] for key,g in records.items()},
        limitations=['Qwen video uses uniform encoding FPS=2; true timestamps are textual',
            'Qwen uses cached selection: not independent raw-video E2E',
            'Historical batches are descriptive timing comparisons',
            'Full baseline results were previously observed; not a wholly unseen blind test',
            'Wilson intervals are marginal; paired uncertainty uses stratified video clusters'])
    durable(run/'summary.json',summary)
    # 步骤3：派生CSV/Markdown可重建，不修改任何原结果。
    with (run/'per_question.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# Video-MME全量RD-1.2与Qwen视频输入实验','',
           '| 模型／方法 | 全2700 | 开发200 | 其余2500 | 开发视频外2100 |','|---|---:|---:|---:|---:|']
    for key,groups in tables.items():
        values=[groups[s]['all'] for s in subsets]
        lines.append('| '+key+' | '+' | '.join(f"{v['correct']}/{v['n']}（{100*v['accuracy']:.2f}%）" for v in values)+' |')
    lines+=['','## 主配对比较','']
    for key,result in comparisons.items():
        v=result['all2700'];lo,hi=v['stratified_video_cluster_bootstrap_95']
        lines.append(f"- {key}：{v['difference_pp']:+.3f}个百分点，视频簇配对95%区间[{lo:.3f}, {hi:.3f}]；新增正确{len(v['cells']['left_only'])}、退化{len(v['cells']['right_only'])}。")
    lines+=['','## 解释边界','']+['- '+v for v in summary['limitations']]
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    return summary
