"""冻结Top-K/RD选帧接入参考问答；不改均匀入口及任何选择算法。"""
import contextlib
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import time
import traceback
from videoqa_runtime.common import ROOT, read_json, sha256, offline_environment
from videoqa_full.state import durable, utc, exclusive, invoke_once, rows, digest, validate_frames
from videoqa_full.prepare import verify_frozen_rd
from videoqa_lmms_bridge.frames import FrameProvider
from videoqa_lmms_bridge.common import load_task, task_doc, POST_PROMPT
from .run import verify as verify_uniform, PYTHON

METHODS = ('BASE-BLIP-TopK', 'RD-1.2')
GPUS = ('4', '5', '6', '7')
UNIFORM = ROOT / 'outputs/baselines/base_uniform__llava__lmms_ref_v1__videomme2700__20260923__r01'
SOURCE = ROOT / 'outputs/full_videoqa/rd_1_2__videomme2700__crossmodel_20260922__r01'


def method_order(ordinal):
    """同题两方法交替先后，冻结题序决定，不参考历史正确性。"""
    return METHODS if ordinal % 2 == 0 else METHODS[::-1]


def prepare(run):
    """接口：核对已有选帧与均匀协议，冻结本轮5400调用和只读来源。"""
    run = Path(run); run.mkdir(parents=True, exist_ok=True)
    with exclusive(run / 'prepare.lock'):
        if (run / 'protocol.json').exists(): return verify(run)
        start = time.perf_counter(); base = verify_uniform(UNIFORM); verify_frozen_rd()
        assert read_json(UNIFORM / 'completion_audit.json')['status'] == 'passed'
        old = read_json(SOURCE / 'protocol.json'); data = rows(); exports = {}; prior = {}
        # 步骤1：每个来源绑定原结果哈希；只导出不带答案的精确选帧供新生成。
        for row in data:
            q = row['question_id']; v = row['video_id']
            for method in METHODS:
                path = SOURCE / 'selections' / method / (q + '.json'); item = read_json(path)
                if method == METHODS[0]:
                    expected = old['selection_exports'][str(path.relative_to(SOURCE))]
                    source = ROOT / item['source_path']; assert sha256(source) == item['source_sha256']
                    original = read_json(source)
                    assert item['selection']['selected_frames'] == original['selection']['selected_frames']
                else:
                    source = SOURCE / 'llava/results' / method / (q + '.json')
                    original = read_json(source); assert original['status'] == 'completed'
                    expected = original['selection_sha256']
                    assert original['protocol_sha256'] == sha256(SOURCE / 'protocol.json')
                    assert original['frame_identity'] == [[f['source_pts'], f['source_frame_index']] for f in item['selection']['selected_frames']]
                assert sha256(path) == expected
                assert item['question_id'] == q and item['video_id'] == v and item['method'] == method
                assert item['question_sha256'] == digest([row['question'], row['options']])
                assert item['source_video_sha256'] == base['video_hashes'][v]
                validate_frames(item['selection'])
                dest = run / 'selections' / method / (q + '.json'); durable(dest, item)
                exports[str(dest.relative_to(run))] = sha256(dest)
                prior[str(source)] = sha256(source)
        # 步骤2：只读复核共享资产与上轮指纹；本轮不生成候选或评分。
        for v, st in base['video_stats'].items():
            assert sha256(st['path']) == base['video_hashes'][v]
        for p, st in base['model_files'].items(): assert sha256(p) == st['sha256']
        files = sorted(set(list((ROOT / 'src').glob('**/*.py')) + list((ROOT / 'scripts').glob('*.py'))))
        hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files}
        for name in hashes:
            dest = run / 'code_snapshot' / name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        spec = dict(protocol_id=base['version'], model=base['model'], methods=list(METHODS),
            uniform_protocol_sha256=sha256(UNIFORM / 'protocol.json'), uniform_summary_sha256=sha256(UNIFORM / 'summary.json'),
            reference_profile=base['reference_profile'], environment=base['environment'],
            source_protocol_sha256=sha256(SOURCE / 'protocol.json'), selection_exports=exports, source_results=prior,
            code_sha256=hashes, allowed_gpus=list(GPUS), pilot_question_ids=base['pilot_question_ids'],
            question_count=2700, answer_calls_per_method=2700, total_answer_calls=5400,
            BLIP_calls=0, classification_calls=0, query_calls=0, retries=0,
            timing='cached selection + exact final decoding + reference QA; not raw selection E2E',
            prepared_seconds=time.perf_counter()-start, utc=utc())
        durable(run / 'protocol.json', spec); print('Prepared 5400 frozen selections', flush=True)
        return spec


def verify(run):
    """恢复接口：环境/协议/源码有变化则暂停，禁止静默换设置。"""
    spec = read_json(Path(run) / 'protocol.json'); verify_uniform(UNIFORM); verify_frozen_rd()
    assert sha256(UNIFORM / 'protocol.json') == spec['uniform_protocol_sha256']
    assert sha256(UNIFORM / 'summary.json') == spec['uniform_summary_sha256']
    assert sha256(SOURCE / 'protocol.json') == spec['source_protocol_sha256']
    for name, h in spec['code_sha256'].items(): assert sha256(ROOT / name) == h, name
    return spec


def selection(run, spec, q, method):
    """只读帧提供接口：校验冻结源身份，不输入答案，不二次采样。"""
    rel = f'selections/{method}/{q}.json'; path = run / rel
    assert sha256(path) == spec['selection_exports'][rel]
    item = read_json(path); validate_frames(item['selection'])
    return item['selection']


def validate(run, row, method, spec):
    """接口：逐题验证真实输入、参考计分及一次生成账本；不运行模型。"""
    q = row['question_id']; r = read_json(run / 'results' / method / (q + '.json'))
    assert r['status'] == 'completed' and r['model'] == spec['model'] and r['method'] == method
    assert r['protocol_sha256'] == sha256(run / 'protocol.json')
    for k in ('question_id', 'video_id', 'question', 'options', 'stratum'): assert r[k] == row[k]
    s = selection(run, spec, q, method); assert r['selection'] == s
    a = r['result']['answer']; o = r['result']['observed']; n = len(s['selected_frames'])
    # 步骤1：观测值必须体现原帧预算与同一问答协议，不套用16帧到短池。
    assert o['generate_calls'] == o['prefill_calls'] == 1 and o['checked_logit_steps'] > 0
    assert o['max_new_tokens'] == 16 and 1 <= len(o['generated_token_ids']) <= 16
    assert o['visual_tokens'] == n * 210 and o['prefill_tokens'] == o['text_input_tokens'] - 1 + n * 210
    assert o['pixel_sha256'] == o['expected_pixel_sha256'] and len(r['rgb_sha256']) == n
    task = load_task(); metric = task.videomme_process_results(task_doc(row), [a['raw_output']])['videomme_perception_score']
    assert metric == a['framework_metric'] and r['correct'] == (metric['pred_answer'] == metric['answer'])
    assert a['context'] == task.videomme_doc_to_text(task_doc(row), {'post_prompt': POST_PROMPT})
    # 步骤2：调用ID按方法隔离，完成答案与持久账本一一对应。
    call = read_json(run / 'calls' / method / (q + '.json'))
    assert call['status'] == 'completed' and call['answer'] == r['result']
    assert call['protocol_sha256'] == r['protocol_sha256']
    assert call['generation_state'] == {'started': True, 'returned': True}
    assert r['gpu'] in GPUS and call['gpu'] == r['gpu']
    return r


def process(run, row, method, backend, adapter, gpu, spec):
    """冻结选帧→精确RGB→调用前持久化→框架问答；仅后处理允许恢复。"""
    from videoqa_lmms_bridge.llava_adapter import observe_generate
    q = row['question_id']; dest = run / 'results' / method / (q + '.json')
    if dest.exists(): validate(run, row, method, spec); return
    with exclusive(run / 'locks' / method / (q + '.lock')):
        # 步骤1：恢复仍校验选帧；模型只接收原问题、选项和已排序RGB。
        start = time.perf_counter(); s = selection(run, spec, q, method)
        images, hashes = FrameProvider(s).decode(); decode_end = time.perf_counter()
        def answer(state):
            a, o = observe_generate(backend, images, 16, state, lambda: adapter.ask(images,
                [f['timestamp_seconds'] for f in s['selected_frames']], s['video']['duration_seconds'], row, s['video']['path']))
            return dict(answer=a, observed=o)
        # 步骤2：已开始失败占用额度，异常不重试；成功答案原子落盘。
        fp = sha256(run / 'protocol.json')
        try: result, reused, checkpoint = invoke_once(run, q, method, fp, gpu, answer)
        finally:
            for image in images: image.close()
        end = time.perf_counter(); metric = result['answer']['framework_metric']
        r = dict(status='completed', model=spec['model'], method=method, protocol_sha256=fp,
            **{k: row[k] for k in ('question_id', 'video_id', 'question', 'options', 'stratum')},
            selection=s, rgb_sha256=hashes, result=result, gpu=gpu, reused_answer=reused,
            correct=metric['pred_answer'] == metric['answer'],
            timings=dict(position_read_and_final_decode_seconds=decode_end-start,
                cached_input_to_persisted_answer_seconds=end-start, call_start_checkpoint_seconds=checkpoint), utc=utc())
        durable(dest, r); validate(run, row, method, spec)


def worker(run, gpu, jobs, events, stop):
    """每卡单一常驻LLaVA，两方法同题同进程，不加载BLIP。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    offline_environment(); run = Path(run); active = None
    with (run / 'logs' / f'gpu_{gpu}.log').open('a', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            import torch
            from videoqa_runtime.llava_backend import LlavaBackend
            from videoqa_lmms_bridge.llava_adapter import FrozenInputLlavaVid
            torch.set_num_threads(2); torch.set_num_interop_threads(1)
            # 步骤1：合成预热只作工程验收，和真实问答分开记账。
            t = time.perf_counter(); backend = LlavaBackend(); adapter = FrozenInputLlavaVid(backend)
            load = time.perf_counter()-t; t = time.perf_counter(); backend.warmup()
            durable(run / 'workers' / f'{gpu}_{os.getpid()}.json', dict(gpu=gpu, pid=os.getpid(),
                load_seconds=load, warmup_seconds=time.perf_counter()-t, synthetic_forwards=1, load_report=backend.load_report))
            spec = read_json(run / 'protocol.json'); events.put(dict(type='ready', gpu=gpu))
            # 步骤2：单任务包含两种方法，冻结题序交替顺序；任何失败停止派发。
            while not stop.is_set():
                job = jobs.get()
                if job is None: break
                row, ordinal = job; active = row['question_id']
                for method in method_order(ordinal):
                    if stop.is_set(): break
                    events.put(dict(type='active', gpu=gpu, question_id=active, method=method))
                    process(run, row, method, backend, adapter, gpu, spec)
                    events.put(dict(type='answer', gpu=gpu, question_id=active, method=method))
                events.put(dict(type='completed', gpu=gpu, question_id=active)); active = None
        except BaseException as error:
            stop.set(); traceback.print_exc()
            durable(run / 'failures' / f'{gpu}_{time.time_ns()}.json', dict(error=repr(error), question_id=active, traceback=traceback.format_exc(), utc=utc()))
            events.put(dict(type='failed', gpu=gpu, error=repr(error)))


def execute(run):
    """后四卡后台协调器；先14条验收，同题同卡，恢复不重复生成。"""
    from videoqa_runtime.baseline_run import available_gpus
    run = Path(run)
    with exclusive(run / 'coordinator.lock'):
        spec = verify(run); data = rows(); byid = {r['question_id']: r for r in data}
        ordinal = {r['question_id']: i for i, r in enumerate(data)}; done = set(); pinned = {}
        # 步骤1：先验证所有账本；不确定状态直接暂停，不能以缺失结果为由重答。
        for method in METHODS:
            calls = list((run / 'calls' / method).glob('*.json')); assert len(calls) <= 2700
            for p in calls:
                c = read_json(p); assert p.stem in byid and c['status'] == 'completed'
                assert c['protocol_sha256'] == sha256(run / 'protocol.json') and c['gpu'] in GPUS
                assert pinned.setdefault(p.stem, c['gpu']) == c['gpu']
            for p in (run / 'results' / method).glob('*.json'):
                assert p.stem in byid; validate(run, byid[p.stem], method, spec); done.add((p.stem, method))
        if len(done) == 5400:
            from .methods_report import summarize
            return summarize(run)
        gpus, states = available_gpus(list(GPUS)); assert gpus and set(gpus) <= set(GPUS)
        for q, gpu in pinned.items():
            if any((q, m) not in done for m in METHODS): assert gpu in gpus, 'Partial pair GPU unavailable'
        for d in ('logs', 'workers', 'failures'): (run / d).mkdir(exist_ok=True)
        durable(run / 'gpu_preflight.json', dict(allowed=list(GPUS), actual=gpus, states=states, utc=utc()))
        ctx = mp.get_context('spawn'); events = ctx.Queue(); stop = ctx.Event()
        queues = {g: ctx.Queue() for g in gpus}; children = {}; ready = set(); active = {}; busy = set()
        pending = []; phase = 'loading'; failure = None; start = time.perf_counter()
        def publish():
            elapsed = time.perf_counter()-start
            durable(run / 'status.json', dict(status=phase, completed=len(done), total=5400,
                counts={m: sum(k[1] == m for k in done) for m in METHODS}, pid=os.getpid(), session_id=os.getsid(0),
                active=active, gpus=gpus, elapsed_seconds=elapsed, failure=failure, utc=utc()))
        def schedule():
            nonlocal pending, phase
            pilot = spec['pilot_question_ids']
            missing = lambda q: any((q, m) not in done for m in METHODS)
            if any(missing(q) for q in pilot): phase='pilot'; pending=[q for q in pilot if missing(q)]
            else:
                for q in pilot:
                    for m in METHODS: validate(run, byid[q], m, spec)
                durable(run / 'pilot_acceptance.json', dict(status='passed', question_ids=pilot, calls=14, accuracy_not_gate=True))
                phase='running'; pending=sorted([q for q in byid if missing(q)],
                    key=lambda q: (-selection(run, spec, q, METHODS[0])['video']['duration_seconds'], q))
        def refill():
            for gpu in gpus:
                if gpu in busy: continue
                q = next((q for q in pending if q not in pinned or pinned[q] == gpu), None)
                if q is None: continue
                pending.remove(q); busy.add(gpu); queues[gpu].put((byid[q], ordinal[q]))
        # 步骤2：先固定14条输入门槛，再按时长调度；控制器与SSH会话无依赖。
        try:
            for gpu in gpus:
                p=ctx.Process(target=worker, args=(str(run), gpu, queues[gpu], events, stop)); p.start(); children[gpu]=p
            publish()
            while len(done) < 5400:
                try: e=events.get(timeout=5)
                except queue.Empty: e=None
                if e:
                    if e['type']=='failed': raise RuntimeError(str(e))
                    if e['type']=='ready':
                        ready.add(e['gpu'])
                        if len(ready)==len(gpus): schedule(); refill()
                    elif e['type']=='active': active[e['gpu']]={k:e[k] for k in ('question_id','method')}
                    elif e['type']=='answer': done.add((e['question_id'], e['method']))
                    elif e['type']=='completed':
                        busy.remove(e['gpu']); active.pop(e['gpu'], None)
                        if phase=='pilot' and not pending and not busy: schedule()
                        refill()
                for gpu, p in children.items():
                    if p.exitcode is not None: raise RuntimeError(f'Worker {gpu} exited {p.exitcode}')
                publish()
            phase='validating'; publish()
        except BaseException as error:
            stop.set(); failure=repr(error); phase='paused'; publish(); raise
        finally:
            for q in queues.values(): q.put(None)
            while any(p.is_alive() for p in children.values()):
                for p in children.values(): p.join(timeout=1)
                publish()
        # 步骤3：完整后才生成三组配对统计，异常与缺失不能伪装成完整结果。
        from .methods_report import summarize
        summarize(run); phase='completed'; publish()
        durable(run / 'execution.json', dict(status='completed', wall_seconds=time.perf_counter()-start,
            gpus=gpus, new_answer_calls=len(done), utc=utc()))
