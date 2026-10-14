"""MLVU七任务宏平均、总体准确率和跨任务视频簇配对统计。"""
from pathlib import Path
from collections import defaultdict
import itertools
import numpy as np
from videoqa_runtime.common import read_json, sha256
from videoqa_runtime.baseline_report import wilson, distribution
from videoqa_full.state import durable, utc
from videoqa_mlvu.data import rows, METHODS, TASKS, PROFILES, load_task
from videoqa_mlvu import runtime, preparation


def score_table(data, bank):
    """接口：M-Avg先按七类各算准确率再等权平均，不能用总体正确率替代。"""
    per_task = {}
    for task in TASKS:
        subset = [r for r in data if r['task_type'] == task]
        if not subset: raise ValueError('Missing task in M-Avg')
        n = len(subset); c = sum(bank[r['question_id']]['correct'] for r in subset)
        per_task[task] = dict(n=n, correct=c, accuracy=c/n, wilson_95=wilson(c,n))
    n = len(data); c = sum(bank[r['question_id']]['correct'] for r in data)
    return dict(n=n, correct=c, accuracy=c/n, wilson_95=wilson(c,n),
                m_avg_percent=100*np.mean([v['accuracy'] for v in per_task.values()]), tasks=per_task)


def compare(data, left, right, repetitions=10000):
    """接口：按内容视频为簇成对重抽，跨任务同源视频不拆开；每次重算宏平均。"""
    cells = {k: [] for k in ('both_correct','left_only','right_only','both_wrong')}
    vids = sorted({r['video_id'] for r in data}); positions = {v:i for i,v in enumerate(vids)}
    sums = np.zeros((len(vids),7)); counts = np.zeros_like(sums)
    # 步骤1：四格与各视频七类贡献来自实际逐题输出，不把题型当独立视频重复。
    for r in data:
        q = r['question_id']; a, b = bool(left[q]['correct']), bool(right[q]['correct'])
        key = 'both_correct' if a and b else 'left_only' if a else 'right_only' if b else 'both_wrong'
        cells[key].append(q); i, j = positions[r['video_id']], TASKS.index(r['task_type'])
        sums[i,j] += int(a)-int(b); counts[i,j] += 1
    rng = np.random.default_rng(2027); macro = []; micro = []
    # 步骤2：抽样保持完整视频簇；某类完全缺失的抽样不定义M-Avg，重新抽至10000有效次。
    while len(macro) < repetitions:
        size = min(500, repetitions-len(macro)); weights = rng.multinomial(len(vids), np.full(len(vids),1/len(vids)), size=size)
        denominators = weights @ counts; numerators = weights @ sums
        valid = (denominators > 0).all(axis=1)
        macro.extend((100*np.mean(numerators[valid]/denominators[valid],axis=1)).tolist())
        micro.extend((100*numerators[valid].sum(axis=1)/denominators[valid].sum(axis=1)).tolist())
    return dict(n=len(data), cells=cells,
        accuracy_difference_pp=100*(len(cells['left_only'])-len(cells['right_only']))/len(data),
        m_avg_difference_pp=score_table(data,left)['m_avg_percent']-score_table(data,right)['m_avg_percent'],
        accuracy_video_cluster_bootstrap_95=np.percentile(micro,[2.5,97.5]).tolist(),
        m_avg_video_cluster_bootstrap_95=np.percentile(macro,[2.5,97.5]).tolist(),
        repetitions=repetitions, seed=2027, clusters=len(vids))


def summarize(root):
    """接口：八组完整并通过审计后才产生正式报告，生成与共享成本分开。"""
    root = Path(root); data = rows(root); clean = [r for r in data if not r['ambiguous_correct_options']]
    if len(clean) != 2169: raise ValueError('Ambiguity subset changed')
    profiles = {}
    # 步骤1：读取实际结果与官方逐题计分，独立复算宏平均并对照原聚合器。
    for profile in PROFILES:
        run = root / 'qa' / profile; audit = runtime.audit(run)
        banks = {}; scores = {}; performance = {}; parse_failures = {}
        for method in METHODS:
            bank = {}; physical = []; frame_counts = defaultdict(int)
            for row in data:
                item = read_json(run / 'results' / method / (row['question_id']+'.json'))
                bank[row['question_id']] = dict(correct=item['correct'], framework_metric=item['answer']['framework_metric'])
                frame_counts[str(len(item['selection']['selected_frames']))] += 1
                if item['timing_kind'] == 'physical_primary': physical.append(item)
            banks[method] = bank; table = score_table(data, bank)
            official = load_task().mlvu_aggregate_results_dev([v['framework_metric'] for v in bank.values()])
            if abs(official-table['m_avg_percent']) > 1e-10: raise ValueError('Native M-Avg differs')
            scores[method] = dict(all2174=table, ambiguity_excluded2169=score_table(clean,bank))
            parse_failures[method] = [q for q,v in bank.items() if v['framework_metric']['pred_answer'] not in 'ABCD' or len(v['framework_metric']['pred_answer']) != 1]
            performance[method] = dict(physical_calls=len(physical),
                actual_frame_counts=dict(frame_counts),
                qa_seconds=distribution([x['answer']['observed']['qa_seconds'] for x in physical]) if physical else None,
                decode_and_identity_seconds=distribution([x['timings']['final_decode_and_identity_seconds'] for x in physical]) if physical else None,
                cached_input_to_answer_seconds=distribution([x['timings']['cached_input_to_answer_seconds'] for x in physical]) if physical else None,
                peak_allocated_gib=distribution([x['answer']['observed']['peak_allocated_gib'] for x in physical]) if physical else None)
            for key in ('peak_reserved_gib','visual_tokens','actual_prefill_tokens','prefill_tokens'):
                values = [x['answer']['observed'][key] for x in physical if key in x['answer']['observed']]
                if values: performance[method][key] = distribution(values)
        paired = {a+'_minus_'+b: dict(all2174=compare(data,banks[a],banks[b]),
                    ambiguity_excluded2169=compare(clean,banks[a],banks[b]))
                  for a,b in [(METHODS[1],METHODS[0]), (METHODS[2],METHODS[0]), (METHODS[2],METHODS[1]),
                              (METHODS[3],METHODS[0]), (METHODS[3],METHODS[1]), (METHODS[2],METHODS[3])]}
        profiles[profile] = dict(scores=scores, paired=paired, performance=performance, parse_failures=parse_failures,
            call_audit=dict(method_results=audit['method_results'], physical_calls=audit['physical_calls']),
        input_summary=read_json(run / 'input_summary.json'),
        model_workers=[{k:v for k,v in read_json(p).items() if k in ('gpu','load_seconds','warmup_seconds','synthetic_generations','load_report')}
                       for p in (run / 'workers').glob('*.json')])
    # 步骤2：额外合成、恢复和准备成本引用原账本，不冒充独立原片E2E或给共享帧重复计费。
    value = dict(status='completed', dataset='MLVU-Dev-MCQ', main_metric='M-Avg over seven task accuracies',
        profiles=profiles, ambiguity_question_ids=[r['question_id'] for r in data if r['ambiguous_correct_options']],
        preparation_stages={p.stem:read_json(p) for p in (root / 'stage_completion').glob('*.json')},
        preparation_worker_totals=[read_json(p) for p in (root / 'worker_totals').glob('*.json')],
        failures=[read_json(p) for p in (root / 'failures').glob('*.json')], utc=utc())
    durable(root / 'comparison.json', value)
    lines = ['# MLVU BLIP-1 双配置全量结果', '', '| 配置 | 方法 | 正确数 | 总体准确率 | M-Avg |', '|---|---|---:|---:|---:|']
    for profile,p in profiles.items():
        for method,s in p['scores'].items():
            t = s['all2174']; lines.append(f"| {profile} | {method} | {t['correct']}/2174 | {100*t['accuracy']:.2f}% | {t['m_avg_percent']:.2f}% |")
    lines += ['', '完整配对题号、七类统计、视频簇区间与2169题补充见comparison.json。',
        '共享输入只记一次物理QA；准备和缓存QA不称原片直接E2E。',
        '本轮未调参，不以优于基线或达到论文分数作为运行验收门槛。']
    (root / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return value


def final_audit(root):
    """接口：真实完成恢复前后校验生成文件字节，并重新核验所有原始源与冻结算法。"""
    root = Path(root); before = {}
    for profile in PROFILES:
        for folder in ('calls','raw_generations','results'):
            for p in (root / 'qa' / profile / folder).glob('*/*.json'): before[str(p.relative_to(root))] = sha256(p)
    for profile in PROFILES: runtime.audit(root / 'qa' / profile)
    after = {p:sha256(root / p) for p in before}
    if before != after: raise ValueError('Completion recovery changed generation artifacts')
    preparation.verify(root, full=True)
    value = dict(status='passed', unchanged_generation_files=len(before), new_generations=0,
        files=after, comparison_sha256=sha256(root / 'comparison.json'), utc=utc())
    durable(root / 'completion_acceptance.json', value); return value
