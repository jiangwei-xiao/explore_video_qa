"""exp32独立完成审计与开发对照：只读取冻结选择和生成，不再调用任何模型。"""
from pathlib import Path
from datetime import datetime
from videoqa_runtime.common import read_json, sha256
from videoqa_runtime.baseline_report import wilson, distribution
from videoqa_full.report import compare
from videoqa_full.state import durable, utc
from videoqa_audit.region_scope_runtime import development_rows, parent_result
from videoqa_audit import literal_development_runtime as runtime

REFERENCES = ('BASE-Uniform', 'BASE-BLIP-TopK', 'RD-1.2-exp07', 'RD-1.2-exp12')


def evaluate(data, results, references, changed, matched):
    """接口：完整同题结果→固定四参照及输入变化子集；空子集不伪造零准确率。"""
    ids = {row['question_id'] for row in data}
    if len(ids) != len(data) or set(results) != ids or any(set(group) != ids for group in references.values()) or not changed <= ids or not matched <= ids:
        raise ValueError('Complete paired population required')
    groups = {runtime.METHOD: results, **references}
    subsets = dict(all=data, changed=[r for r in data if r['question_id'] in changed],
        unchanged=[r for r in data if r['question_id'] not in changed],
        literal_matched=[r for r in data if r['question_id'] in matched])
    # 步骤1：总体与每时长使用相同分母；匹配/变化属于代理子集，不替代总体。
    accuracy = {}
    for name, records in groups.items():
        accuracy[name] = {}
        for subset, selected in subsets.items():
            accuracy[name][subset] = {}
            for layer in ('all', 'short', 'medium', 'long'):
                rows = [r for r in selected if layer == 'all' or r['stratum'] == layer]
                if not rows:
                    continue
                count = sum(records[r['question_id']]['correct'] for r in rows)
                accuracy[name][subset][layer] = dict(n=len(rows), correct=count, accuracy=count / len(rows), wilson_95=wilson(count, len(rows)))
    # 步骤2：既定四项比较保留全部得失；视频簇分层bootstrap沿用10000次/2027。
    comparisons = {name: {subset: compare(rows, results, records) for subset, rows in subsets.items() if rows}
        for name, records in references.items()}
    return dict(accuracy=accuracy, comparisons=comparisons, changed_ids=sorted(changed), matched_ids=sorted(matched))


def finish(root):
    """接口：全200完成后实际恢复一次并独立复算；生成文件与历史参照必须字节不变。"""
    root = Path(root); run = root / 'qa/qwen32'; spec = runtime.verify(run); data = development_rows()
    if not (run / 'summary.json').exists() or read_json(run / 'status.json')['status'] != 'completed':
        raise ValueError('Formal200 must complete before audit')
    if {p.stem for p in (run / 'results' / runtime.METHOD).glob('*.json')} != set(spec['question_ids']) or any({p.stem for p in (run / directory / runtime.METHOD).glob('*.json')} != set(spec['new_ids']) for directory in ('calls', 'raw_generations')):
        raise ValueError('Complete physical calls and result population required')
    results = {row['question_id']: runtime.validate_result(run, row, spec) for row in data}
    files = [p for directory in ('results', 'calls', 'raw_generations') for p in (run / directory).glob('*/*.json')]
    if (run / 'synthetic.json').exists():
        files.append(run / 'synthetic.json')
    before = {str(p): sha256(p) for p in files}
    # 步骤1：真实完成态恢复，不加载GPU；再次核验结果/调用/原始返回逐字不变。
    runtime.execute(root)
    after = {str(p): sha256(p) for p in files}
    if before != after or read_json(run / 'restore_audit.json')['new_generations'] != 0:
        raise ValueError('Completed recovery changed a generation')
    for row in data:
        runtime.validate_result(run, row, spec)
    references = {}; sources = dict(before)
    for name in REFERENCES:
        records = {}
        for row in data:
            q = row['question_id']; path = parent_result('qwen32', q, name); value = read_json(path)
            if value['question_id'] != q or value['question'] != row['question'] or value['options'] != row['options']:
                raise ValueError('Reference population identity differs')
            metric = runtime.load_task().videomme_process_results(runtime.task_doc(row), [value['answer']['raw_output']])['videomme_perception_score']
            if value['correct'] != (metric['answer'] == metric['pred_answer']):
                raise ValueError('Reference scoring differs')
            records[q] = value; sources[str(path)] = sha256(path)
        references[name] = records
    if {m: sum(r['correct'] for r in g.values()) for m, g in references.items()} != dict(zip(REFERENCES, (130, 126, 136, 138))):
        raise ValueError('Frozen development reference differs')
    selections = {row['question_id']: read_json(root / 'selections' / (row['question_id'] + '.json')) for row in data}
    changed = {q for q, item in selections.items() if item['changed_frames']}
    matched = {q for q, item in selections.items() if item['matching_frames']}
    if changed != set(spec['new_ids']):
        raise ValueError('New calls differ from frozen input changes')
    # 步骤2：统计真实调用、CPU OCR与缓存成本；导入不伪造新QA或原片E2E。
    banks = [read_json(p) for p in (root / 'ocr_banks').glob('*.json')]
    prep_spec = read_json(root / 'protocol.json'); frame_positions = sum(bank['frames'] for bank in banks)
    if len(banks) != 200 or frame_positions != prep_spec['maximum_new_ocr_frame_positions']:
        raise ValueError('OCR population or source-position count differs')
    fresh = [results[q] for q in spec['new_ids']]
    value = evaluate(data, results, references, changed, matched)
    score = value['accuracy'][runtime.METHOD]['all']['all']['correct']
    value.update(method=runtime.METHOD, status='completed', utc=utc(),
        physical_new_QA=len(spec['new_ids']), imported=len(spec['imported_ids']),
        synthetic_generations=int((run / 'synthetic.json').exists()),
        parse_failures=[q for q, r in results.items() if r['answer']['parsed_answer'] is None],
        completion_audit=dict(unchanged_generation_files=len(before), new_generations_on_resume=0),
        ocr=dict(banks=len(banks), source_frame_positions=frame_positions,
            summed_worker_seconds=sum(bank['seconds'] for bank in banks),
            wall_seconds=(datetime.fromisoformat(read_json(root / 'status.json')['utc']) - datetime.fromisoformat(read_json(root / 'launch.json')['utc'])).total_seconds()),
        performance={key: distribution([r['answer']['observed'][key] for r in fresh]) for key in ('qa_seconds', 'visual_tokens', 'prefill_tokens', 'peak_allocated_gib')} if fresh else {},
        full_score_gate_passed=score >= 142, full_started=False,
        full_requires_repeated_mechanism_evidence=True,
        limitations=['Repeatedly explored development200; not full accuracy or independent blind validation',
            'OCR literal word coverage does not guarantee temporal/role/negation fidelity',
            'Imported answers consume no new physical calls; OCR costs are not free',
            'Only Qwen32 development is evaluated; no cross-budget improvement claim'])
    # 步骤3：独立派生报告不覆盖冻结QA汇总；来源哈希随结果落盘。
    for q in selections:
        for directory in ('selections', 'ocr_banks'):
            path = root / directory / (q + '.json'); sources[str(path)] = sha256(path)
    durable(root / 'comparison_sources.json', sources)
    durable(root / 'comparison.json', value)
    return value
