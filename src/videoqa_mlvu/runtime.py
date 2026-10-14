"""MLVU模型输入冻结、同题共享、一次生成账本与完整恢复验收。"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
import os
import numpy as np
from videoqa_runtime.common import read_json, sha256, offline_environment
from videoqa_full.state import durable, digest, exclusive, invoke_once, utc
from videoqa_budget32.frames import FrameProvider, validate_frames
from videoqa_mlvu.data import rows, METHODS, PROFILES, context
from videoqa_mlvu.preparation import verify
from videoqa_mlvu.backends import cpu_identity, metric, observer_module, make_adapter, observe, answer_from_raw

_TOOLS = None


def source_order(selection):
    """接口：JSON原生有序源身份，避免Python元组和持久列表造成虚假不一致。"""
    return [[int(f['source_pts']), int(f['source_frame_index'])] for f in selection['selected_frames']]


def initialize_cpu(profile):
    """接口：只加载既有官方CPU处理器与Tokenizer，不加载问答模型权重。"""
    global _TOOLS
    offline_environment()
    from videoqa_scorer_comparison.runtime import make_tools
    _TOOLS = make_tools(profile)


def prepare_question(job):
    """接口：四方法真实源RGB/官方像素/完整文字核验后按完整身份分组，不读取正确性。"""
    root, profile, row = Path(job[0]), job[1], job[2]; run = root / 'qa' / profile
    q = row['question_id']; items = {}; bank = {}; sources = {}; groups = []
    # 步骤1：相同源序列仅重复利用CPU处理结果，不跨问题/模型共享答案。
    for method in METHODS:
        path = root / 'selections' / profile / method / (q+'.json'); sources[str(path)] = sha256(path)
        selection = read_json(path)['selection']; validate_frames(selection, PROFILES[profile])
        key = digest(source_order(selection))
        if key not in bank:
            images, rgb = FrameProvider(selection).decode()
            try: ident = cpu_identity(profile, _TOOLS, images, row)
            finally:
                for image in images: image.close()
            bank[key] = (rgb, ident)
        rgb, ident = bank[key]; identity = dict(ident, source=source_order(selection), rgb_sha256=rgb)
        item = dict(question_id=q, method=method, selection=selection, input_identity=identity)
        items[method] = item
        group = next((g for g in groups if g['identity'] == identity), None)
        if group is None:
            group = dict(representative=method, methods=[], identity=identity); groups.append(group)
        group['methods'].append(method)
    # 步骤2：全部文件落盘后再发布检查点，正式生成只能使用冻结分组。
    paths = []
    for method, item in items.items():
        path = run / 'selections' / method / (q+'.json'); durable(path, item); paths.append(path)
    path = run / 'groups' / (q+'.json'); durable(path, groups); paths.append(path)
    durable(run / 'input_checkpoints' / (q+'.json'), dict(sources=sources,
        files={str(p.relative_to(run)): sha256(p) for p in paths}, utc=utc()))
    return q


def prepare_inputs(root, profile):
    """接口：2174题CPU身份和实际新调用额度事前冻结；无历史MLVU答案导入。"""
    root = Path(root); verify(root); run = root / 'qa' / profile; data = rows(root); start = time.perf_counter()
    if (run / 'protocol.json').exists(): return verify_inputs(run)
    pending = []
    for r in data:
        path = run / 'input_checkpoints' / (r['question_id']+'.json')
        if path.exists():
            c = read_json(path)
            for p, h in c['sources'].items():
                if sha256(p) != h: raise ValueError('Input source changed')
            for p, h in c['files'].items():
                if sha256(run / p) != h: raise ValueError('Input checkpoint changed')
        else: pending.append(r)
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn'), initializer=initialize_cpu, initargs=(profile,)) as executor:
        for i, q in enumerate(executor.map(prepare_question, [(str(root), profile, r) for r in pending]), 1):
            durable(run / 'prepare_status.json', dict(completed=len(data)-len(pending)+i, total=2174, utc=utc()))
    calls = []; files = {}
    for r in data:
        q = r['question_id']; calls += [[q, g['representative']] for g in read_json(run / 'groups' / (q+'.json'))]
        c = read_json(run / 'input_checkpoints' / (q+'.json'))
        for rel, h in c['files'].items(): files[rel] = h
    spec = dict(dataset='MLVU-Dev-MCQ', profile=profile, frame_budget=PROFILES[profile],
        question_ids=[r['question_id'] for r in data], new_calls=calls, maximum_new_real_generations=len(calls),
        parent_protocol_sha256=sha256(root / 'protocol.json'), inputs=files,
        task_protocol='lmms-mlvu-reference-v1', generation=dict(max_new_tokens=16, do_sample=False, seed=2027), utc=utc())
    durable(run / 'protocol.json', spec)
    durable(run / 'input_summary.json', dict(method_results=8696, new_calls=len(calls), same_input_references=8696-len(calls),
        historical_references=0, preparation_wall_seconds=time.perf_counter()-start, utc=utc()))
    return verify_inputs(run)


def verify_inputs(run):
    """接口：核验任务父协议及全部输入摘要；恢复必须沿用已冻结调用额度。"""
    run = Path(run); root = run.parent.parent; verify(root); spec = read_json(run / 'protocol.json')
    if sha256(root / 'protocol.json') != spec['parent_protocol_sha256']: raise ValueError('Parent protocol changed')
    if len(spec['question_ids']) != 2174 or len(set(spec['question_ids'])) != 2174 or len(spec['new_calls']) != spec['maximum_new_real_generations']:
        raise ValueError('Frozen input/call denominator differs')
    for rel, h in spec['inputs'].items():
        if sha256(run / rel) != h: raise ValueError('Frozen actual input changed: ' + rel)
    return spec


def observed_identity(profile, answer):
    """接口：真实GPU生成的像素和提示字段与CPU冻结身份使用同一含义。"""
    o = answer['observed']
    result = dict(prompt=o['prompt'], input_token_ids=o['input_token_ids'],
        pixel_sha256=o['processor_pixel_sha256'] if profile == 'qwen32' else o['pixel_sha256'])
    if profile == 'qwen32': result['video_grid_thw'] = o['video_grid_thw']
    return result


def persist_group(run, row, spec, group, answer, gpu, timing, resumed=False):
    """接口：共享结果引用同一物理调用，保留各方法自己的源选择与成本类别。"""
    run = Path(run); q = row['question_id']; representative = group['representative']
    for method in group['methods']:
        item = read_json(run / 'selections' / method / (q+'.json'))
        value = dict(status='completed', question_id=q, video_id=row['video_id'], task_type=row['task_type'],
            method=method, representative=representative, model=spec['profile'], gpu=gpu,
            protocol_sha256=sha256(run / 'protocol.json'), selection=item['selection'],
            rgb_sha256=item['input_identity']['rgb_sha256'], answer=answer,
            correct=answer['framework_metric']['pred_answer'] == answer['framework_metric']['answer'],
            timings=timing if method == representative else {},
            timing_kind=('resumed_postprocessing' if resumed else 'physical_primary') if method == representative else 'same_input_reference', utc=utc())
        durable(run / 'results' / method / (q+'.json'), value)


def recover_group(run, row, spec, group):
    """接口：只有完整原始返回允许恢复后处理；不确定生成禁止换方法另发一次。"""
    run = Path(run); q = row['question_id']; m = group['representative']
    if all((run / 'results' / method / (q+'.json')).exists() for method in group['methods']): return
    call_path = run / 'calls' / m / (q+'.json'); raw_path = run / 'raw_generations' / m / (q+'.json')
    if not call_path.exists():
        if raw_path.exists(): raise ValueError('Raw output without durable call start')
        return
    call = read_json(call_path)
    if call['protocol_sha256'] != sha256(run / 'protocol.json'): raise ValueError('Call protocol differs')
    if call['status'] == 'completed': answer = call['answer']
    else:
        if not raw_path.exists(): raise RuntimeError('Uncertain generation, repetition forbidden: ' + q)
        raw = read_json(raw_path)
        expected = dict(question_id=q, method=m, protocol_sha256=sha256(run / 'protocol.json'), gpu=call['gpu'])
        if raw['identity'] != expected or raw['generation_state'] != {'started': True, 'returned': True}:
            raise RuntimeError('Uncertain raw return')
        answer = answer_from_raw(spec['profile'], raw, row)
        if 'qa_seconds' not in answer['observed']:
            answer['observed']['qa_seconds'] = answer['observed']['qa_through_generation_seconds']
        call.update(status='completed', answer=answer, generation_state=raw['generation_state'], recovered_at=utc())
        durable(call_path, call)
    persist_group(run, row, spec, group, answer, call['gpu'], {}, resumed=True)


def validate_result(run, row, method, spec):
    """接口：完整提示、实际像素、源身份、原生计分与物理账本共同验收。"""
    run = Path(run); q = row['question_id']; item = read_json(run / 'selections' / method / (q+'.json'))
    result = read_json(run / 'results' / method / (q+'.json')); answer = result['answer']; o = answer['observed']
    if result['protocol_sha256'] != sha256(run / 'protocol.json') or result['video_id'] != row['video_id'] or result['task_type'] != row['task_type']:
        raise ValueError('Result protocol/video/task differs')
    actual = dict(observed_identity(spec['profile'], answer), source=source_order(result['selection']), rgb_sha256=result['rgb_sha256'])
    if actual != item['input_identity'] or result['selection'] != item['selection'] or result['method'] != method or result['question_id'] != q:
        raise ValueError('Actual input differs from frozen MLVU identity')
    score = metric(row, [answer['raw_output']])
    if score != answer['framework_metric'] or result['correct'] != (score['pred_answer'] == score['answer']): raise ValueError('MLVU score differs')
    if o['generate_calls'] != 1 or o['prefill_calls'] != 1 or not 1 <= len(o['generated_token_ids']) <= 16: raise ValueError('Generation observations differ')
    if spec['profile'] == 'llava8' and o['visual_tokens'] != 210*len(item['selection']['selected_frames']): raise ValueError('LLaVA visual tokens differ')
    m = result['representative']; call = read_json(run / 'calls' / m / (q+'.json')); raw = read_json(run / 'raw_generations' / m / (q+'.json'))
    group = next(g for g in read_json(run / 'groups' / (q+'.json')) if method in g['methods'])
    expected = dict(question_id=q, method=m, protocol_sha256=result['protocol_sha256'], gpu=result['gpu'])
    if m != group['representative'] or call['status'] != 'completed' or call['answer'] != answer or raw['generation_state'] != {'started': True, 'returned': True} or raw['identity'] != expected:
        raise ValueError('Shared call/raw source differs')
    return result


def process_question(run, row, backend, adapter, observer, gpu, spec):
    """接口：同题四方法按冻结顺序循环轮换；每个唯一完整输入只生成一次。"""
    run = Path(run); q = row['question_id']; groups = read_json(run / 'groups' / (q+'.json'))
    shift = row['ordinal'] % 4; order = METHODS[shift:]+METHODS[:shift]; seen = set()
    for method in order:
        group = next(g for g in groups if method in g['methods']); m = group['representative']
        if m in seen: continue
        seen.add(m)
        with exclusive(run / 'locks' / (q+'.lock')):
            recover_group(run, row, spec, group)
            if all((run / 'results' / x / (q+'.json')).exists() for x in group['methods']): continue
            item = read_json(run / 'selections' / m / (q+'.json')); start = time.perf_counter()
            images, rgb = FrameProvider(item['selection']).decode()
            try:
                tools = (backend.processor, backend.tokenizer if spec['profile'] == 'llava8' else None)
                ident = dict(cpu_identity(spec['profile'], tools, images, row), source=source_order(item['selection']), rgb_sha256=rgb)
                if ident != item['input_identity']: raise ValueError('Runtime actual input changed before generation')
                decoded = time.perf_counter(); fp = sha256(run / 'protocol.json')
                identity = dict(question_id=q, method=m, protocol_sha256=fp, gpu=gpu)
                def invoke(state):
                    return observe(spec['profile'], observer, backend, adapter, images, item['selection'], row, state,
                                   run / 'raw_generations' / m / (q+'.json'), identity)
                answer, _, checkpoint = invoke_once(run, q, m, fp, gpu, invoke)
                persist_group(run, row, spec, group, answer, gpu, dict(final_decode_and_identity_seconds=decoded-start,
                    cached_input_to_answer_seconds=time.perf_counter()-start, call_checkpoint_seconds=checkpoint))
            finally:
                for image in images: image.close()
    for method in METHODS: validate_result(run, row, method, spec)


def qa_initialize(run, gpu):
    """接口：长期后端先完成合成边界验收；合成不取真实题或消耗正式额度。"""
    import torch
    from PIL import Image
    offline_environment(); torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(2027)
    run = Path(run); profile = read_json(run / 'protocol.json')['profile']; start = time.perf_counter()
    if profile == 'qwen32':
        from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
        backend = QwenFA2VideoBackend(); counts = (1, 11, 31, 32)
    else:
        from videoqa_runtime.llava_backend import LlavaBackend
        backend = LlavaBackend(); counts = (1, 7, 8)
    loading = time.perf_counter()-start; adapter = make_adapter(profile, backend); observer = observer_module(profile)
    start = time.perf_counter(); records = []
    row = dict(question='Which synthetic color is visible?', qa_question='Which synthetic color is visible?\n(A) Red\n(B) Blue\n(C) Green\n(D) Black\n',
               options=['Red','Blue','Green','Black'], answer_index=1, task_type='plotQA')
    for n in counts:
        images = [Image.new('RGB', (896,672) if n % 2 == 0 else (672,896), (40,20+i*3,140)) for i in range(n)]
        selection = dict(video=dict(path='synthetic.mp4', duration_seconds=float(n)),
                         selected_frames=[dict(timestamp_seconds=float(i)) for i in range(n)])
        raw = run / 'synthetic' / gpu / (str(os.getpid())+'_'+str(n)+'_raw.json')
        try: value = observe(profile, observer, backend, adapter, images, selection, row, {}, raw, dict(synthetic=True, frames=n))
        finally:
            for image in images: image.close()
        records.append(dict(source_frames=n, answer=value))
    durable(run / 'workers' / ('qa_'+gpu+'_'+str(os.getpid())+'.json'), dict(status='passed', gpu=gpu,
        load_seconds=loading, warmup_seconds=time.perf_counter()-start, synthetic_generations=len(counts), records=records,
        load_report=backend.load_report, utc=utc()))
    return backend, adapter, observer


def audit(run):
    """接口：四组各2174结果和实际冻结U闭合，完成恢复不改任何原始生成文件。"""
    run = Path(run); spec = verify_inputs(run); data = rows(run.parent.parent); ids = set(spec['question_ids'])
    for method in METHODS:
        if {p.stem for p in (run / 'results' / method).glob('*.json')} != ids: raise ValueError('Method incomplete')
        for row in data: validate_result(run, row, method, spec)
    expected = {tuple(x) for x in spec['new_calls']}
    for folder in ('calls', 'raw_generations'):
        if {(p.stem,p.parent.name) for p in (run / folder).glob('*/*.json')} != expected: raise ValueError('Physical budget/raw set differs')
    files = {str(p.relative_to(run)): sha256(p) for folder in ('calls','raw_generations','results') for p in (run / folder).glob('*/*.json')}
    dest = run / 'completion_audit.json'
    if dest.exists() and read_json(dest)['files'] != files: raise ValueError('Completion recovery changed generation artifacts')
    value = dict(status='passed', method_results=8696, physical_calls=len(expected), files=files, utc=utc())
    durable(dest, value); return value
