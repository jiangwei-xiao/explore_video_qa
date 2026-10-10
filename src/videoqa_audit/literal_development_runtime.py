"""exp32独立开发问答：官方输入身份导入、一次生成与后台协调，不改旧矩阵。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import time
from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, exclusive, utc
from videoqa_audit.framework_contraction import RUN32
from videoqa_audit.region_scope_runtime import (
    development_rows, source_order, parent_result, parent_call_sources, cpu_identity,
)
from videoqa_lmms_bridge.common import framework_identity, load_task, task_doc, POST_PROMPT
from videoqa_full.prepare import environment
from videoqa_budget32.frames import FrameProvider, validate_frames
from videoqa_budget32.run import process, recover_call, validate

METHOD = 'RD-1.2-exp32'
PYTHON = '/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'
QA_FILES = ('src/videoqa_audit/literal_development_runtime.py', 'scripts/run_literal_development.py',
    'src/videoqa_budget32/run.py', 'src/videoqa_budget32/observe.py', 'src/videoqa_budget32/qwen_adapter.py',
    'src/videoqa_budget32/frames.py', 'src/videoqa_full/qwen_fa2_backend.py', 'src/videoqa_full/qwen_backend.py',
    'src/videoqa_lmms_bridge/common.py')


def prepare(root):
    """接口：全部200选择完整后，冻结真实像素/提示/Token和同输入导入，不看对错划分。"""
    root = Path(root); run = root / 'qa/qwen32'; run.mkdir(parents=True, exist_ok=True)
    if (run / 'protocol.json').exists():
        return verify(run)
    selection_summary = read_json(root / 'summary.json')
    if selection_summary['questions'] != 200 or not selection_summary['ready_for_separate_QA_input_review']:
        raise ValueError('Finish all200 selection preparation first')
    data = development_rows(); parent_spec = read_json(RUN32 / 'protocol.json')
    if framework_identity() != parent_spec['framework'] or environment(PYTHON) != parent_spec['environment']:
        raise ValueError('Parent evaluation environment differs')
    for path, identity in parent_spec['model_files'].items():
        if sha256(path) != identity['sha256']:
            raise ValueError('Shared model bytes changed')
    code = {name: sha256(ROOT / name) for name in QA_FILES}
    for name in QA_FILES[2:]:
        if name in parent_spec['code_sha256'] and code[name] != parent_spec['code_sha256'][name]:
            raise ValueError('Frozen inference code differs: ' + name)
    import torch
    from transformers import AutoProcessor
    from videoqa_full.state import QWEN_PATH
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(QWEN_PATH, local_files_only=True, use_fast=False,
        min_pixels=100352, max_pixels=602112, size=dict(shortest_edge=100352, longest_edge=602112))
    processor.apply_chat_template([dict(role='user', content='synthetic')], tokenize=False, add_generation_prompt=True)
    tools = (processor, None); imports = []; new = []; exports = {}
    sources = {str(root / 'protocol.json'): sha256(root / 'protocol.json'), str(root / 'summary.json'): sha256(root / 'summary.json'),
        str(RUN32 / 'protocol.json'): sha256(RUN32 / 'protocol.json')}
    # 步骤1：逐源验证视频身份；不能用从未绑定的旧结果替代同协议输入。
    for vid in sorted({row['video_id'] for row in data}):
        info = parent_spec['video_stats'][vid]; path = Path(info['path']); stat = path.stat()
        if stat.st_size != info['bytes'] or stat.st_mtime_ns != info['mtime_ns'] or sha256(path) != parent_spec['video_hashes'][vid]:
            raise ValueError('Frozen development video differs')
    def one(row):
        """接口：一题完整输入→官方CPU身份；源相同仍必须核验预处理与完整提示。"""
        q = row['question_id']; source = root / 'selections' / (q + '.json')
        if sha256(source) != selection_summary['selection_exports'][str(source)]:
            raise ValueError('Approved development selection changed')
        item = read_json(source); selection = item['selection']; validate_frames(selection, 32)
        old_path = parent_result('qwen32', q); old = read_json(old_path)
        images, rgb = FrameProvider(selection).decode()
        try:
            identity = cpu_identity('qwen32', tools, images, row)
        finally:
            for image in images:
                image.close()
        same = source_order(selection) == source_order(old['selection'])
        parent_sources = {str(old_path): sha256(old_path)}
        if same:
            observed = old['answer']['observed']
            if rgb != old['rgb_sha256'] or identity['pixel_sha256'] != observed['processor_pixel_sha256'] or identity['input_token_ids'] != observed['input_token_ids'] or identity['prompt'] != observed['prompt'] or identity['video_grid_thw'] != observed['video_grid_thw']:
                raise ValueError('Same-frame import is not identical model input: ' + q)
            parent_sources = parent_call_sources('qwen32', old_path, old)
        destination = run / 'selections' / METHOD / (q + '.json')
        value = dict(question_id=q, video_id=row['video_id'], method=METHOD, selection=selection,
            rgb_sha256=rgb, input_identity=identity, imported=same, parent_result=str(old_path),
            parent_sources=parent_sources, frame_budget=32)
        durable(destination, value)
        return value, destination, {str(source): sha256(source), **parent_sources}
    # 步骤2：先冻结200输入及导入/新调用分区；导入依据只来自实际输入，不是正确性。
    with ThreadPoolExecutor(max_workers=4) as executor:
        for number, (item, destination, extra) in enumerate(executor.map(one, data), 1):
            q = item['question_id']; (imports if item['imported'] else new).append(q)
            exports[str(destination.relative_to(run))] = sha256(destination); sources.update(extra)
            durable(run / 'prepare_status.json', dict(completed=number, total=200, utc=utc()))
    spec = dict(method=METHOD, model=parent_spec['model'], profile='qwen32', frame_budget=32,
        question_ids=[row['question_id'] for row in data], imported_ids=imports, new_ids=new,
        maximum_new_real_generations=len(new), maximum_results=200, sources=sources,
        code_sha256=code, framework=parent_spec['framework'], environment=parent_spec['environment'],
        model_files=parent_spec['model_files'], video_stats=parent_spec['video_stats'], selection_exports=exports,
        pilot_question_ids=[row['question_id'] for row in data[:5]], allowed_gpus=list(map(str, range(8))),
        synthetic_limit=1, parent_method='RD-1.2-exp07', scope='Fixed literal group rule, Qwen32 development200')
    for name in code:
        dest = run / 'code_snapshot' / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, dest)
    durable(run / 'protocol.json', spec)
    return spec


def verify(run):
    """接口：200分区、来源与正式推理协议一致；完成生成不可再次调用。"""
    run = Path(run); spec = read_json(run / 'protocol.json')
    if spec['method'] != METHOD or len(spec['question_ids']) != 200 or len(set(spec['question_ids'])) != 200:
        raise ValueError('Frozen method or population differs')
    if set(spec['new_ids']) & set(spec['imported_ids']) or set(spec['new_ids']) | set(spec['imported_ids']) != set(spec['question_ids']) or len(spec['new_ids']) != spec['maximum_new_real_generations']:
        raise ValueError('Frozen call partition differs')
    if framework_identity() != spec['framework'] or environment(PYTHON) != spec['environment']:
        raise ValueError('Evaluation environment changed')
    for name, expected in spec['code_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError('Frozen generation code changed')
    for path, identity in spec['model_files'].items():
        stat = Path(path).stat()
        if stat.st_size != identity['bytes'] or stat.st_mtime_ns != identity['mtime_ns']:
            raise ValueError('Frozen model identity changed')
    for path, expected in spec['sources'].items():
        if sha256(path) != expected:
            raise ValueError('Frozen generation source changed')
    for relative, expected in spec['selection_exports'].items():
        if sha256(run / relative) != expected:
            raise ValueError('Frozen generated input changed')
    return spec


def import_answers(run, spec):
    """接口：已经验收同输入的原答案引用，保留物理来源，不伪造新调用。"""
    data = {row['question_id']: row for row in development_rows()}
    for q in spec['imported_ids']:
        path = run / 'results' / METHOD / (q + '.json')
        if path.exists():
            continue
        item = read_json(run / 'selections' / METHOD / (q + '.json')); old = read_json(item['parent_result']); row = data[q]
        durable(path, dict(status='completed', model=spec['model'], method=METHOD, protocol_sha256=sha256(run / 'protocol.json'),
            **{key: row[key] for key in ('question_id', 'video_id', 'question', 'options', 'stratum')},
            selection=item['selection'], rgb_sha256=item['rgb_sha256'], answer=old['answer'], correct=old['correct'],
            imported=True, parent_sources=item['parent_sources'], timings={}, timing_kind='historical_input_verified_import', utc=utc()))


def validate_result(run, row, spec):
    """接口：新答案与引用统一核验真实模型像素/完整提示，标准答案仅在此计分。"""
    q = row['question_id']; item = read_json(run / 'selections' / METHOD / (q + '.json'))
    result = read_json(run / 'results' / METHOD / (q + '.json'))
    # 步骤1：引用也必须归属当前模型、方法和原题；不能只核验图像后混入旧结果元数据。
    if result['status'] != 'completed' or result['model'] != spec['model'] or result['method'] != METHOD or any(result[key] != row[key] for key in ('question_id', 'video_id', 'question', 'options', 'stratum')):
        raise ValueError('Result method, model or question identity differs')
    observed = result['answer']['observed']
    expected = item['input_identity']
    if result['protocol_sha256'] != sha256(run / 'protocol.json') or result['selection'] != item['selection'] or result['rgb_sha256'] != item['rgb_sha256']:
        raise ValueError('Result input identity differs')
    if observed['processor_pixel_sha256'] != expected['pixel_sha256'] or observed['prompt'] != expected['prompt'] or observed['input_token_ids'] != expected['input_token_ids'] or observed['video_grid_thw'] != expected['video_grid_thw']:
        raise ValueError('Actual official model input differs from frozen CPU input')
    if observed['generate_calls'] != 1 or observed['prefill_calls'] != 1 or not 1 <= len(observed['generated_token_ids']) <= 16 or observed['generation_kwargs']['max_new_tokens'] != 16 or observed['generation_kwargs']['do_sample'] is not False:
        raise ValueError('Frozen generation observations differ')
    # 步骤2：身份和真实模型输入通过后才复算计分；计分不影响导入/新调用的事前划分。
    metric = load_task().videomme_process_results(task_doc(row), [result['answer']['raw_output']])['videomme_perception_score']
    if metric != result['answer']['framework_metric'] or result['correct'] != (metric['answer'] == metric['pred_answer']):
        raise ValueError('Framework scoring differs')
    if result.get('imported'):
        if q not in spec['imported_ids'] or result['parent_sources'] != item['parent_sources']:
            raise ValueError('Import attribution differs')
    else:
        validate(run, row, METHOD, spec)
    return result


def report(run, spec):
    """接口：全200配对结果、全部得失及调用预算，未完成不出正式汇总。"""
    data = development_rows(); results = [validate_result(run, row, spec) for row in data]
    old = [read_json(parent_result('qwen32', row['question_id'])) for row in data]
    pairs = dict(improved=[], regressed=[], both_correct=[], both_wrong=[])
    for row, a, b in zip(data, old, results):
        category = 'both_correct' if a['correct'] and b['correct'] else 'both_wrong' if not a['correct'] and not b['correct'] else 'improved' if b['correct'] else 'regressed'
        pairs[category].append(row['question_id'])
    calls = list((run / 'calls' / METHOD).glob('*.json'))
    if len(calls) != spec['maximum_new_real_generations']:
        raise ValueError('Physical call count differs from frozen new-input partition')
    summary = dict(method=METHOD, n=200, correct=sum(r['correct'] for r in results), parent_correct=sum(r['correct'] for r in old),
        by_stratum={s: dict(n=sum(r['stratum'] == s for r in results), correct=sum(r['correct'] for r in results if r['stratum'] == s)) for s in ('short', 'medium', 'long')},
        paired=pairs, new_QA_calls=len(calls), imported=len(spec['imported_ids']), full_recommendation_only=False,
        note='Exploratory repeatedly-used development200; not a full dataset gain or significance claim')
    durable(run / 'summary.json', summary); durable(run / 'status.json', dict(status='completed', completed=200, utc=utc()))
    return summary


def execute(root):
    """接口：单卡长期正式Qwen进程，先五题输入验收，再按时长调度，答错不重试。"""
    run = Path(root) / 'qa/qwen32'
    with exclusive(run / 'coordinator.lock'):
        spec = prepare(root); verify(run); import_answers(run, spec)
        data = {row['question_id']: row for row in development_rows()}
        # 步骤1：恢复只允许已完整返回的后处理；未完成开始状态不重发。
        for q in spec['new_ids']:
            recover_call(run, data[q], METHOD, spec)
        if len(list((run / 'results' / METHOD).glob('*.json'))) == 200:
            report(run, spec); durable(run / 'restore_audit.json', dict(new_generations=0, utc=utc())); return
        from videoqa_runtime.baseline_run import available_gpus
        required = {read_json(p)['gpu'] for p in (run / 'calls' / METHOD).glob('*.json')}
        if len(required) > 1:
            raise ValueError('Single-worker run has multiple pinned GPUs')
        # 步骤2：只有输入变化题消耗新QA；少量任务不加载八份模型，无空闲卡则后台等。
        while True:
            try:
                free, _ = available_gpus(spec['allowed_gpus'])
            except RuntimeError:
                free = []
            if free and (not required or required <= set(free)):
                break
            durable(run / 'status.json', dict(status='waiting_resources', utc=utc())); time.sleep(15)
        gpu = next(iter(required)) if required else free[0]
        os.environ.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        from PIL import Image
        from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
        from videoqa_budget32.qwen_adapter import FrozenInputQwen
        from videoqa_budget32.observe import observe
        from videoqa_lmms_bridge.bridge import synthetic_row
        backend = QwenFA2VideoBackend(); backend.torch.set_num_threads(2); adapter = FrozenInputQwen(backend)
        durable(run / 'worker.json', dict(gpu=gpu, pid=os.getpid(), sid=os.getsid(0), load_report=backend.load_report, utc=utc()))
        if not (run / 'synthetic.json').exists():
            if (run / 'synthetic_started.json').exists():
                raise ValueError('Uncertain synthetic generation')
            pictures = [Image.new('RGB', (896, 672), (20 + i * 3, 100, 150)) for i in range(32)]
            durable(run / 'synthetic_started.json', dict(started=True, utc=utc()))
            value = observe(backend, adapter, pictures, synthetic_row(32), {})
            for picture in pictures:
                picture.close()
            durable(run / 'synthetic.json', value)
        pilot = spec['pilot_question_ids']
        pending = [q for q in pilot if q in spec['new_ids']] + sorted([q for q in spec['new_ids'] if q not in pilot],
            key=lambda q: -read_json(run / 'selections' / METHOD / (q + '.json'))['selection']['video']['duration_seconds'])
        # 步骤3：native process保存原始返回并框架计分，输入/Token正确才放行，不按准确率。
        for q in pending:
            process(run, data[q], METHOD, backend, adapter, gpu, spec); validate_result(run, data[q], spec)
            if all((run / 'results' / METHOD / (question + '.json')).exists() for question in pilot):
                for question in pilot:
                    validate_result(run, data[question], spec)
                if not (run / 'pilot_acceptance.json').exists():
                    durable(run / 'pilot_acceptance.json', dict(status='passed', ids=pilot, accuracy_not_gate=True))
            durable(run / 'status.json', dict(status='running', completed=len(list((run / 'results' / METHOD).glob('*.json'))),
                total=200, gpu=gpu, active_finished=q, pid=os.getpid(), sid=os.getsid(0), utc=utc()))
        report(run, spec)
