"""合并全量基线 r1（5388条）与预算上限修订后的补跑 r2（12条）为完整 5400 条最终报告。

背景：r1（videomme_full_uniform_topk_20260916_r1）在两个短视频处因原协议的
16 帧硬性预算暂停，完成 5388/5400。2026-09-17 用户裁定固定预算为最大选帧上限，
代码修订后以独立运行 r2（videomme_full_completion_20260917_r2，协议版本
videomme-full-completion6-v1）补齐剩余 6 题。本脚本：

1. 校验两轮运行各自的 protocol.json 与其清单/协议哈希一致；
2. 用各自运行的协议哈希逐条复验全部 5400 条结果（修订后的校验对 r1 的
   ≥16 候选结果逐字段等价）；
3. 断言两轮结果按 (question_id, method) 互斥且并集恰为 2700×2；
4. 以与 baseline_report.summarize 完全相同的统计口径（Wilson 区间、
   分层配对 bootstrap、计时/显存分布）生成合并 summary.json、report.md、
   per_question.csv 与 provenance.json，写入独立的合并目录。

两轮源运行目录保持原样不动；合并目录是派生数据，归档时与源目录一同打包。
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from videoqa_runtime.baseline_records import METHODS, result_path, validate_pair
from videoqa_runtime.baseline_report import wilson, distribution, paired_interval
from videoqa_runtime.common import ROOT, read_json, sha256, write_json

R1_DIR = ROOT / 'outputs/baselines/videomme_full_uniform_topk_20260916_r1'
R2_DIR = ROOT / 'outputs/baselines/videomme_full_completion_20260917_r2'
FAILED_DIR = ROOT / 'outputs/baselines/videomme_full_completion_20260917_r1'
FULL_MANIFEST = ROOT / 'data/manifests/video_mme_full_2700.json'
COMPLETION_MANIFEST = ROOT / 'data/manifests/video_mme_full_completion_6.json'
OUT_DIR = ROOT / 'outputs/baselines/videomme_full_uniform_topk_20260916_merged'


def load_run(run_dir, manifest_path, label, expect_missing=()):
    """读取单轮运行：校验协议/清单一致性，逐条复验结果。
    expect_missing 为该轮按设计缺席的题号（r1 缺补跑集的 6 题）。"""
    spec = read_json(run_dir / 'protocol.json')
    if spec['dataset_manifest'] != str(manifest_path.relative_to(ROOT)):
        raise ValueError(f'{label}: unexpected dataset manifest {spec["dataset_manifest"]}')
    if sha256(manifest_path) != spec['dataset_manifest_sha256']:
        raise ValueError(f'{label}: frozen manifest hash mismatch')
    for name, digest in spec['code_sha256'].items():
        if sha256(run_dir / 'code_snapshot' / name) != digest:
            raise ValueError(f'{label}: frozen code snapshot changed: {name}')
    rows = read_json(manifest_path)['rows']
    protocol_hash = sha256(run_dir / 'protocol.json')
    records = {m: {} for m in METHODS}
    for method in METHODS:
        for p in sorted((run_dir / 'results' / method).glob('*.json')):
            r = read_json(p)
            if r['question_id'] in records[method]:
                raise ValueError(f'{label}: duplicate result identity {r["question_id"]}')
            records[method][r['question_id']] = r
        expected_present = {r['question_id'] for r in rows} - set(expect_missing)
        if set(records[method]) != expected_present:
            missing = sorted(expected_present - set(records[method]))
            extra = sorted(set(records[method]) - expected_present)
            raise ValueError(f'{label}/{method}: missing={missing[:8]} extra={extra[:8]}')
    for row in rows:
        if row['question_id'] in expect_missing:
            continue
        validate_pair({m: records[m][row['question_id']] for m in METHODS}, row, protocol_hash, spec)
    attempts = [read_json(p) for p in (run_dir / 'attempts').glob('*/*.json')]
    return records, spec, protocol_hash, attempts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out-dir', default=str(OUT_DIR.relative_to(ROOT)))
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=False)

    rows = read_json(FULL_MANIFEST)['rows']
    by_id = {r['question_id']: r for r in rows}
    if len(rows) != 2700 or len(by_id) != 2700:
        raise ValueError('full manifest is not the frozen 2700-question set')

    completion_rows = read_json(COMPLETION_MANIFEST)['rows']
    completion_ids = [r['question_id'] for r in completion_rows]
    r1, r1_spec, r1_hash, r1_attempts = load_run(R1_DIR, FULL_MANIFEST, 'r1', expect_missing=completion_ids)
    r2, r2_spec, r2_hash, r2_attempts = load_run(R2_DIR, COMPLETION_MANIFEST, 'r2')
    # 失败补跑没有答案，但必须核验并计入全部尝试和成本。
    _, _, _, failed_attempts = load_run(FAILED_DIR, COMPLETION_MANIFEST, 'failed completion', expect_missing=completion_ids)
    failed_exec = read_json(FAILED_DIR / 'execution.json')
    for row in completion_rows:
        if row != by_id[row['question_id']]:
            raise ValueError(f'completion row differs from full manifest: {row["question_id"]}')

    merged, provenance_results = {m: {} for m in METHODS}, []
    for method in METHODS:
        overlap = set(r1[method]) & set(r2[method])
        if overlap:
            raise ValueError(f'r1/r2 overlap for {method}: {sorted(overlap)}')
        merged[method] = {**r1[method], **r2[method]}
        if len(merged[method]) != 2700:
            raise ValueError(f'{method}: merged {len(merged[method])} results, expected 2700')
    counts = dict(r1={m: len(r1[m]) for m in METHODS}, r2={m: len(r2[m]) for m in METHODS})
    if counts['r1'] != {'uniform': 2694, 'topk': 2694} or counts['r2'] != {'uniform': 6, 'topk': 6}:
        raise ValueError(f'unexpected split: {counts}')
    for method in METHODS:
        for qid, record in merged[method].items():
            source = 'r2' if qid in r2[method] else 'r1'
            provenance_results.append(dict(question_id=qid, method=method, source_run=source,
                                           path=str(result_path(R2_DIR if source == 'r2' else R1_DIR, method, qid).relative_to(ROOT)),
                                           sha256=sha256(result_path(R2_DIR if source == 'r2' else R1_DIR, method, qid))))

    # —— 以下统计口径与 baseline_report.summarize 逐项一致 ——
    methods = {}
    for method in METHODS:
        values = list(merged[method].values())
        by_stratum = {}
        for layer in ('all', 'short', 'medium', 'long'):
            group = values if layer == 'all' else [r for r in values if r['stratum'] == layer]
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
        u, t = merged['uniform'][qid]['correct'], merged['topk'][qid]['correct']
        label = 'both_correct' if u and t else 'uniform_only' if u else 'topk_only' if t else 'both_wrong'
        cells[label].append(qid)
    attempts = r1_attempts + failed_attempts + r2_attempts
    r1_exec, r2_exec = read_json(R1_DIR / 'execution.json'), read_json(R2_DIR / 'execution.json')
    frames_used = {m: sorted({len(r['selection']['selected_frames']) for r in merged[m].values()}) for m in METHODS}

    report = dict(
        run_id=out_dir.name,
        merged_from=dict(
            r1=dict(run_dir=str(R1_DIR.relative_to(ROOT)), protocol_sha256=r1_hash,
                    protocol_version=r1_spec['version'], results=5388,
                    status=r1_exec['status'], note='paused at the two short videos under the original fixed 16-frame budget'),
            r2=dict(run_dir=str(R2_DIR.relative_to(ROOT)), protocol_sha256=r2_hash,
                    protocol_version=r2_spec['version'], results=12,
                    status=r2_exec['status'], note='completion under the 2026-09-17 budget-cap amendment')),
        protocol_amendment='2026-09-17: the 16-frame budget is a maximum cap; videos with fewer distinct '
                           '1FPS candidates (14 for uF3zNOthLAg, 11 for 6Z_XNM_iT4g) use all of them',
        frames_used_per_method=frames_used,
        methods=methods,
        accuracy_difference_pp=100 * (methods['topk']['by_stratum']['all']['accuracy'] - methods['uniform']['by_stratum']['all']['accuracy']),
        difference_bootstrap_95_pp=paired_interval(rows, merged), bootstrap_repetitions=10000, bootstrap_seed=2027,
        paired_cells=cells,
        execution=dict(total_active_wall_seconds=r1_exec['total_active_wall_seconds'] + failed_exec['total_active_wall_seconds'] + r2_exec['total_active_wall_seconds'],
                       r1=r1_exec, failed_completion=failed_exec, r2=r2_exec),
        attempts=dict(total=len(attempts), generation_started=sum(bool(a.get('generation_state', {}).get('started')) for a in attempts),
                      generation_returned=sum(bool(a.get('generation_state', {}).get('returned')) for a in attempts),
                      failed=sum(a['status'] != 'completed' for a in attempts),
                      result_write_seconds=sum(a.get('result_write_seconds', 0) for a in attempts)),
        scope='full 2700-question public Video-MME test set; no application-cache hits; OS cache uncontrolled; resident-model E2E',
        result_sha256=provenance_results)
    write_json(out_dir / 'summary.json', report)

    with (out_dir / 'per_question.csv').open('w', newline='') as handle:
        columns = ['question_id', 'stratum', 'method', 'correct', 'raw_output', 'parsed_answer', 'gpu', 'candidates',
                   'selected_frames', 'selection_core_seconds', 'selection_total_seconds', 'qa_total_seconds', 'end_to_end_seconds', 'source_run']
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            for method in METHODS:
                r = merged[method][row['question_id']]
                writer.writerow(dict(question_id=row['question_id'], stratum=row['stratum'], method=method,
                                     correct=r['correct'], raw_output=r['answer']['raw_output'], parsed_answer=r['answer']['parsed_answer'],
                                     gpu=r['physical_gpu'], candidates=len(r['selection']['candidates']),
                                     selected_frames=len(r['selection']['selected_frames']),
                                     source_run='r2' if row['question_id'] in r2[method] else 'r1',
                                     **{k: r['timings'][k] for k in columns[-5:-1]}))

    lines = [f'# Video-MME 完整公开测试集基线（r1+补跑合并）：{out_dir.name}', '',
             '两组各2700题、5400条新结果；由 r1（5388条）与 2026-09-17 预算上限修订后的补跑 r2（12条）合并；旧单题验收未混入。', '',
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
        u, t = [methods[m]['by_stratum'][layer] for m in METHODS]
        lines.append(f'| {layer} | {u["correct"]}/{u["n"]} ({u["accuracy"]:.1%}) | {t["correct"]}/{t["n"]} ({t["accuracy"]:.1%}) | '
                     f'{u["timings"]["end_to_end_seconds"]["mean"]:.3f} | {t["timings"]["end_to_end_seconds"]["mean"]:.3f} |')
    lines += ['', '配对结果：']
    for label, ids in cells.items():
        shown = ids if len(ids) <= 100 else ids[:50] + [f'... 其余{len(ids) - 50}题见per_question.csv']
        lines += ['', f'- {label}：{len(ids)}题；' + ', '.join(shown)]
    lines += ['',
              f'三轮调度器累计活跃墙钟（含失败补跑、核验、加载、预热、调度与保存）：{report["execution"]["total_active_wall_seconds"] / 60:.2f}分钟。',
              f'全部尝试{len(attempts)}次，失败{report["attempts"]["failed"]}次；实际开始生成{report["attempts"]["generation_started"]}次。',
              f'r1 执行GPU：{r1_exec["selected_gpus"]}；补跑执行GPU：{r2_exec["selected_gpus"]}；额外重试：r1={r1_exec["extra_retries"]}，r2={r2_exec["extra_retries"]}。', '',
              '2026-09-17 修订：16帧预算为最大选帧上限。两个短视频（uF3zNOthLAg 14帧、6Z_XNM_iT4g 11帧）由补跑 r2 以全部候选完成；其余 2694 题在 r1 中以 16 帧完成，行为与修订前完全一致。', '',
              '计时为模型常驻、原始视频读取到答案解析的直接墙钟测量。选帧全流程含解码；核心含RGB转换、BLIP预处理/评分和排名。应用缓存命中为0，未清理OS缓存。', '',
              '工作进程同时常驻LLaVA和BLIP，总显存不等于均匀方法独立部署显存；阶段增量与分组详细分位数见summary.json。此为完整公开测试集（2700题）成绩。', '']
    (out_dir / 'report.md').write_text('\n'.join(lines))
    print(f'merged report written to {out_dir}')
    print('Final accuracy:', {m: methods[m]['by_stratum']['all']['accuracy'] for m in METHODS})


if __name__ == '__main__':
    main()
