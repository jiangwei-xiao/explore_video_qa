import csv
import math
from pathlib import Path
import numpy as np

from .common import ROOT, read_json, sha256, write_json
from .baseline_records import METHODS, result_path, validate_pair


def wilson(correct, n):
    z = 1.959963984540054
    p = correct / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0, center - spread), min(1, center + spread)]


def distribution(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(values), mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                p90=float(np.percentile(a, 90)), p95=float(np.percentile(a, 95)), min=float(a.min()), max=float(a.max()))


def paired_interval(rows, records, repetitions=10000):
    rng = np.random.default_rng(2027)
    total = np.zeros(repetitions)
    counted = 0
    for layer in ('short', 'medium', 'long'):
        group = [r for r in rows if r['stratum'] == layer]
        if not group:
            continue  # 单分层数据集（如补跑集）没有的分层直接跳过
        differences = np.array([int(records['topk'][r['question_id']]['correct']) - int(records['uniform'][r['question_id']]['correct']) for r in group])
        indices = rng.integers(0, len(group), size=(repetitions, len(group)))
        total += differences[indices].sum(axis=1)
        counted += len(group)
    samples = total / counted * 100
    return [float(v) for v in np.percentile(samples, [2.5, 97.5])]


def summarize(run_dir):
    """从运行目录重建统计报告；题目清单、期望题数与口径说明取自该轮 protocol.json
    （旧50题运行的 protocol.json 缺少 dataset_manifest/scope 字段时按开发集默认回退）。"""
    run_dir = Path(run_dir)
    spec = read_json(run_dir / 'protocol.json')
    manifest_path = spec.get('dataset_manifest', 'data/manifests/video_mme_development.json')
    if sha256(ROOT / manifest_path) != spec['dataset_manifest_sha256']:
        raise ValueError('Frozen dataset manifest hash mismatch')
    expected = spec.get('question_count', 50)
    scope = spec.get('scope', '50-question development set; no application-cache hits; OS cache uncontrolled; resident-model E2E')
    rows = read_json(ROOT / manifest_path)['rows']
    protocol_hash = sha256(run_dir / 'protocol.json')
    records = {method: {} for method in METHODS}
    for method in METHODS:
        paths = list((run_dir / 'results' / method).glob('*.json'))
        if len(paths) != expected:
            raise ValueError(f'{method} has {len(paths)} results, expected {expected}')
        for p in paths:
            r = read_json(p)
            if r['question_id'] in records[method]:
                raise ValueError('Duplicate result identity')
            records[method][r['question_id']] = r
    for row in rows:
        validate_pair({m: records[m][row['question_id']] for m in METHODS}, row, protocol_hash, spec)
    methods = {}
    for method in METHODS:
        values = list(records[method].values())
        by_stratum = {}
        for layer in ('all', 'short', 'medium', 'long'):
            group = values if layer == 'all' else [r for r in values if r['stratum'] == layer]
            if not group:
                continue  # 单分层数据集（如补跑集）没有的分层不产生统计
            correct = sum(r['correct'] for r in group)
            by_stratum[layer] = dict(correct=correct, n=len(group), accuracy=correct / len(group),
                                    wilson_95=wilson(correct, len(group)),
                                    timings={key: distribution([r['timings'][key] for r in group]) for key in group[0]['timings']},
                                    memory={key: distribution([r['memory'][key] for r in group]) for key in group[0]['memory']})
        methods[method] = dict(by_stratum=by_stratum, parse_failures=[r['question_id'] for r in values if r['answer']['parsed_answer'] is None])
        if method == 'topk':
            n_frames = sum(len(r['selection']['candidates']) for r in values)
            forward = sum(r['timings']['blip_forward_seconds'] for r in values)
            full = sum(r['timings']['rgb_conversion_seconds'] + r['timings']['blip_preprocess_seconds'] + r['timings']['blip_forward_seconds'] for r in values)
            methods[method]['blip_throughput'] = dict(candidates=n_frames, forward_frames_per_second=n_frames / forward,
                                                    with_preprocess_frames_per_second=n_frames / full)
    cells = {name: [] for name in ('both_correct', 'uniform_only', 'topk_only', 'both_wrong')}
    for row in rows:
        qid = row['question_id']
        u, t = records['uniform'][qid]['correct'], records['topk'][qid]['correct']
        label = 'both_correct' if u and t else 'uniform_only' if u else 'topk_only' if t else 'both_wrong'
        cells[label].append(qid)
    attempts = [read_json(p) for p in (run_dir / 'attempts').glob('*/*.json')]
    execution = read_json(run_dir / 'execution.json')
    report = dict(run_id=run_dir.name, protocol_sha256=protocol_hash, results=expected * len(METHODS), methods=methods,
                  accuracy_difference_pp=100 * (methods['topk']['by_stratum']['all']['accuracy'] - methods['uniform']['by_stratum']['all']['accuracy']),
                  difference_bootstrap_95_pp=paired_interval(rows, records), bootstrap_repetitions=10000, bootstrap_seed=2027,
                  paired_cells=cells, execution=execution,
                  attempts=dict(total=len(attempts), generation_started=sum(bool(a.get('generation_state', {}).get('started')) for a in attempts),
                                generation_returned=sum(bool(a.get('generation_state', {}).get('returned')) for a in attempts),
                                failed=sum(a['status'] != 'completed' for a in attempts),
                                result_write_seconds=sum(a.get('result_write_seconds', 0) for a in attempts)),
                  scope=scope,
                  result_sha256={str(p.relative_to(run_dir)): sha256(p) for m in METHODS for p in sorted((run_dir / 'results' / m).glob('*.json'))})
    write_json(run_dir / 'summary.json', report)
    with (run_dir / 'per_question.csv').open('w', newline='') as handle:
        columns = ['question_id', 'stratum', 'method', 'correct', 'raw_output', 'parsed_answer', 'gpu', 'candidates',
                   'selection_core_seconds', 'selection_total_seconds', 'qa_total_seconds', 'end_to_end_seconds']
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            for method in METHODS:
                r = records[method][row['question_id']]
                writer.writerow(dict(question_id=row['question_id'], stratum=row['stratum'], method=method,
                                     correct=r['correct'], raw_output=r['answer']['raw_output'], parsed_answer=r['answer']['parsed_answer'],
                                     gpu=r['physical_gpu'], candidates=len(r['selection']['candidates']),
                                     **{k: r['timings'][k] for k in columns[-4:]}))
    # 标题与导语保持与历史50题运行逐字一致；全量2700题运行使用完整测试集措辞
    if expected == 2700:
        title, subtitle = '# Video-MME 完整公开测试集基线', f'两组各{expected}题、{expected * len(METHODS)}条新结果；旧单题验收未混入。'
    else:
        title, subtitle = f'# Video-MME {expected}题基线', f'两组各{expected}题、{expected * len(METHODS)}条新结果；旧单题验收未混入。以下为开发集结果。'
    lines = [f'{title}：{run_dir.name}', '',
             subtitle, '',
             '| 方法 | 正确数 | 准确率 | 95% Wilson区间 | 选帧核心均值(s) | 选帧全流程均值(s) | 端到端均值(s) | E2E P50/P90/P95(s) |',
             '|---|---:|---:|---|---:|---:|---:|---|']
    for method in METHODS:
        a = methods[method]['by_stratum']['all']
        t = a['timings']; e = t['end_to_end_seconds']
        interval = a['wilson_95']
        lines.append(f'| {method} | {a["correct"]}/{a["n"]} | {a["accuracy"]:.1%} | {interval[0]:.1%}–{interval[1]:.1%} | '
                     f'{t["selection_core_seconds"]["mean"]:.3f} | {t["selection_total_seconds"]["mean"]:.3f} | {e["mean"]:.3f} | '
                     f'{e["p50"]:.3f}/{e["p90"]:.3f}/{e["p95"]:.3f} |')
    lines += ['', f'Top-K − uniform：{report["accuracy_difference_pp"]:+.1f}个百分点；分层配对bootstrap 95%区间：{report["difference_bootstrap_95_pp"]}个百分点。', '',
              '| 时长 | uniform | topk | uniform E2E均值(s) | topk E2E均值(s) |', '|---|---:|---:|---:|---:|']
    for layer in ('short', 'medium', 'long'):
        if layer not in methods[METHODS[0]]['by_stratum']:
            continue
        u, t = [methods[m]['by_stratum'][layer] for m in METHODS]
        lines.append(f'| {layer} | {u["correct"]}/{u["n"]} ({u["accuracy"]:.1%}) | {t["correct"]}/{t["n"]} ({t["accuracy"]:.1%}) | '
                     f'{u["timings"]["end_to_end_seconds"]["mean"]:.3f} | {t["timings"]["end_to_end_seconds"]["mean"]:.3f} |')
    lines += ['', '配对结果：']
    for label, ids in cells.items():
        # 题量超过100时仅列出前50个题号，完整清单以 per_question.csv 为准
        shown = ids if len(ids) <= 100 else ids[:50] + [f'... 其余{len(ids) - 50}题见per_question.csv']
        lines += ['', f'- {label}：{len(ids)}题；' + ', '.join(shown)]
    lines += ['', f'调度器总墙钟时间（含核验、模型加载、合成预热、调度与保存）：{execution["total_active_wall_seconds"] / 60:.2f}分钟。',
              f'执行GPU：{execution["selected_gpus"]}；额外重试：{execution["extra_retries"]}。', '',
              '计时为模型常驻、原始视频读取到答案解析的直接墙钟测量。选帧全流程含解码；核心含RGB转换、BLIP预处理/评分和排名。应用缓存命中为0，未清理OS缓存。', '',
              '工作进程同时常驻LLaVA和BLIP，总显存不等于均匀方法独立部署显存；阶段增量与分组详细分位数见summary.json。'
              + ('此为完整公开测试集（2700题）成绩。' if expected == 2700 else '结果不能外推为完整Video-MME成绩或稳定方法优势。'), '']
    (run_dir / 'report.md').write_text('\n'.join(lines))
    return report
