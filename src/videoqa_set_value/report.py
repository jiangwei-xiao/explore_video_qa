"""固定排序头的独立验证统计；候选可达题数不作为算法准确率。"""
from collections import Counter,defaultdict
from pathlib import Path
import html
import math
import numpy as np

from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,utc
from videoqa_set_value.data import METHODS


def wilson(correct,n):
    """接口：总体点估计的Wilson区间；视频内两题相关性另用簇bootstrap处理。"""
    z=1.959963984540054;p=correct/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [center-half,center+half]


def paired_cluster_interval(rows,differences):
    """接口：100个视频簇各含2题，固定种子10000次配对重抽，不拆开同源问题。"""
    grouped=defaultdict(list)
    for row,d in zip(rows,differences):grouped[row['video_id']].append(d)
    if len(grouped)!=100 or any(len(v)!=2 for v in grouped.values()):raise ValueError('Expected 100 two-question validation clusters')
    means=np.array([np.mean(v) for v in grouped.values()],dtype=np.float64)
    rng=np.random.default_rng(2027);values=[]
    for _ in range(50):
        samples=rng.integers(0,len(means),size=(200,len(means)));values.extend(means[samples].mean(axis=1).tolist())
    return [float(v) for v in np.quantile(values,[.025,.975])]


def report(root):
    """接口：只在200题预测冻结且答案完整后统一报告，不依据验证分数重选模型。"""
    root=Path(root);manifest=read_json(root/'manifest.json');rows=[r for r in manifest['rows'] if r['split']=='val']
    training=read_json(root/'head/training.json');frozen=read_json(root/'validation_predictions.json')
    if frozen['checkpoint_sha256']!=sha256(root/'head/checkpoint.pt'):raise ValueError('Frozen prediction/checkpoint changed')
    predictions={r['question_id']:r for r in frozen['predictions']};records=[];values={m:[] for m in METHODS}
    values.update(learned_selection=[],mean_itm_selection=[]);preference=[]
    for row in rows:
        q=row['question_id'];p=predictions[q]
        data={m:read_json(root/'qa/val/results'/m/(q+'.json')) for m in METHODS}
        ys=[bool(data[m]['correct']) for m in METHODS];plan=read_json(root/'inputs'/(q+'.json'))
        reps=[g['representative_method_index'] for g in plan['groups']];pairs=[]
        for i in reps:
            for j in reps:
                if ys[i] and not ys[j]:
                    d=p['scores'][i]-p['scores'][j];pairs.append(1. if d>1e-8 else .5 if abs(d)<=1e-8 else 0.)
        if pairs:preference.append(float(np.mean(pairs)))
        for m,y in zip(METHODS,ys):values[m].append(y)
        values['learned_selection'].append(data[p['chosen_method']]['correct'])
        values['mean_itm_selection'].append(data[p['mean_itm_method']]['correct'])
        records.append(dict(question_id=q,video_id=row['video_id'],task_type=row['task_type'],
            candidate_correct=dict(zip(METHODS,ys)),chosen_method=p['chosen_method'],mean_itm_method=p['mean_itm_method'],
            learned_correct=values['learned_selection'][-1],preference_accuracy=float(np.mean(pairs)) if pairs else None,
            any_candidate_correct=any(ys),all_candidates_correct=all(ys),candidate_scores=p['scores']))
    if len(records)!=200:raise ValueError('Independent validation denominator differs')
    best_fixed=training['training_best_fixed_method'];comparisons={};learned=np.array(values['learned_selection'],dtype=int)
    for method in [*METHODS,'mean_itm_selection']:
        base=np.array(values[method],dtype=int);gain=(learned==1)&(base==0);loss=(learned==0)&(base==1)
        comparisons[method]=dict(new_correct=[rows[i]['question_id'] for i in np.flatnonzero(gain)],
            degraded=[rows[i]['question_id'] for i in np.flatnonzero(loss)],
            both_correct=[rows[i]['question_id'] for i in np.flatnonzero((learned==1)&(base==1))],
            both_wrong=[rows[i]['question_id'] for i in np.flatnonzero((learned==0)&(base==0))],
            net_correct=int(gain.sum()-loss.sum()),difference=float((learned-base).mean()),
            cluster_95_interval=paired_cluster_interval(rows,learned-base))
    counts={k:int(sum(v)) for k,v in values.items()}
    gains=set(comparisons[best_fixed]['new_correct']);gain_videos={r['video_id'] for r in rows if r['question_id'] in gains}
    criteria=dict(net_gain_four=counts['learned_selection']-counts[best_fixed]>=4,improved_three_videos=len(gain_videos)>=3,
        preference_at_least_sixty=bool(preference) and np.mean(preference)>=.6,
        no_lower_than_best_fixed_validation=counts['learned_selection']>=max(counts[m] for m in METHODS))
    criteria={k:bool(v) for k,v in criteria.items()};criteria['continue_recommended']=all(criteria.values())
    teacher_cost=[];synthetic=[];parse_errors=dict(train=0,val=0)
    for split in ('train','val'):
        for path in (root/'qa'/split/'calls').glob('*/*.json'):
            call=read_json(path);target=synthetic if path.parent.name=='synthetic' else teacher_cost
            target.append(call['answer']['observed'])
            if path.parent.name in METHODS:parse_errors[split]+=call['answer']['parsed_answer'] is None
    blip=[read_json(p) for p in (root/'blip_calls').glob('*/*.json')]
    reuse=read_json(root/'preparation_reuse.json') if (root/'preparation_reuse.json').exists() else {'imported_blip_batches':0}
    groups=Counter(r['task_type'] for r in rows);by_task={}
    for task,n in groups.items():
        ids=[i for i,r in enumerate(rows) if r['task_type']==task]
        by_task[task]=dict(questions=n,correct={m:sum(values[m][i] for i in ids) for m in values})
    result=dict(status='completed',task='set_value_learning_pilot',validation_questions=200,validation_videos=100,
        counts=counts,accuracy={m:counts[m]/200 for m in counts},wilson_95={m:wilson(counts[m],200) for m in counts},
        training_best_fixed_method=best_fixed,preference_questions=len(preference),
        preference_accuracy=float(np.mean(preference)) if preference else None,comparisons=comparisons,
        candidate_headroom=dict(any_correct=sum(r['any_candidate_correct'] for r in records),
            all_correct=sum(r['all_candidates_correct'] for r in records),all_wrong=sum(not r['any_candidate_correct'] for r in records),
            meaning='仅这四个固定候选的已观察可达题数，不是算法成绩或同预算理论上限'),
        method_choice_counts=dict(Counter(r['chosen_method'] for r in records)),by_task=by_task,criteria=criteria,
        training_diagnostics=dict(questions=training['train_questions'],usable_questions=training['usable_questions'],
            usable_videos=training['usable_videos'],selected_correct=training['train_selected_correct'],
            selected_accuracy=training['train_selected_correct']/training['train_questions'],
            preference_accuracy=training['train_preference_accuracy'],fixed_method_correct=training['fixed_method_correct'],
            parameters=training['parameters'],epochs=training['epochs'],
            note='训练表现只用于检查拟合与泛化差距，不作为独立验证收益。'),
        performance=dict(real_teacher_calls=len(teacher_cost),synthetic_teacher_calls=len(synthetic),
            shared_method_results=3200-len(teacher_cost),teacher_generation_seconds=sum(x['generation_seconds'] for x in teacher_cost),
            output_tokens=sum(len(x['generated_token_ids']) for x in teacher_cost),
            peak_allocated_gib=max(x['peak_allocated_gib'] for x in teacher_cost),
            blip_batches=len(blip),blip_frames=sum(x['frames'] for x in blip),
            imported_blip_batches=reuse['imported_blip_batches'],new_blip_batches=len(blip)-reuse['imported_blip_batches'],
            blip_forward_seconds=sum(x['forward_seconds'] for x in blip),
            training_seconds=training['training_seconds'],physical_parse_failures=parse_errors),
        provenance=dict(manifest_sha256=sha256(root/'manifest.json'),predictions_sha256=sha256(root/'validation_predictions.json'),
            training_sha256=sha256(root/'head/training.json'),checkpoint_sha256=sha256(root/'head/checkpoint.pt')),
        limitations=['只验证四个固定输入的学习排序，尚非逐帧统一选择算法。',
            '只验证LLaVA8；跨模型、预算、数据集迁移尚未验证。',
            '教师预训练是否见过相关NExT-QA素材未排除；选择头的独立验证与教师未见数据不是同一概念。',
            '训练标签由冻结问答模型实际对错形成，不能自动解释为语义证据充分性。'],records=records,utc=utc())
    dest=root/'report.json'
    if dest.exists():
        old=read_json(dest)
        if {k:v for k,v in old.items() if k!='utc'}!={k:v for k,v in result.items() if k!='utc'}:raise ValueError('Completed report changed')
        return old
    durable(dest,result)
    esc=lambda x:html.escape(str(x))
    page='<!doctype html><meta charset="utf-8"><title>独立数据集合价值学习小试</title><style>body{max-width:1200px;margin:24px auto;font:16px/1.7 sans-serif}pre{white-space:pre-wrap;background:#eee;padding:12px}td,th{border:1px solid #aaa;padding:6px}table{border-collapse:collapse}</style><h1>800题轻量集合价值学习</h1><p>600训练/200独立验证；只训练小排序头。以下为诊断结果，不是Video-MME或MLVU新成绩。</p><pre>'+esc({k:v for k,v in result.items() if k!='records'})+'</pre><table><tr><th>题号</th><th>任务</th><th>四候选对错</th><th>预测选择</th><th>实际正确</th></tr>'
    page+=''.join('<tr><td>'+esc(r['question_id'])+'</td><td>'+esc(r['task_type'])+'</td><td>'+esc(r['candidate_correct'])+'</td><td>'+esc(r['chosen_method'])+'</td><td>'+esc(r['learned_correct'])+'</td></tr>' for r in records)
    (root/'index.html').write_text(page+'</table>');return result
