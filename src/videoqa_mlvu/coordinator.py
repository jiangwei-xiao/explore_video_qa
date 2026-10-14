"""MLVU独立后台协调；每卡长期工作进程、先行门槛和失败暂停。"""
from pathlib import Path
import multiprocessing as mp
import os
import time
import queue
import traceback
from videoqa_runtime.common import read_json
from videoqa_full.state import durable, exclusive, utc
from videoqa_mlvu.data import rows, METHODS, TASKS
from videoqa_scorer_comparison.coordinator import choose_gpus


def worker(root, stage, gpu, jobs, events, stop):
    """接口：只在独立spawn进程内加载模型，任何异常暂停，不重发生成。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    root = Path(root); active = None; start = time.perf_counter()
    try:
        if stage in ('coarse', 'refine'):
            from videoqa_mlvu import preparation as prep
            prep.initialize_scorer(root, gpu, stage)
        else:
            from videoqa_mlvu import runtime
            run = root / 'qa' / stage; spec = read_json(run / 'protocol.json')
            backend, adapter, observer = runtime.qa_initialize(run, gpu)
        events.put(dict(type='ready', gpu=gpu))
        # 步骤1：先完成模型/合成验收，再消费固定先行题及其后长视频优先队列。
        while not stop.is_set():
            job = jobs.get()
            if job is None: break
            members = job if isinstance(job, list) else [job]; active = members[0]['question_id']
            events.put(dict(type='active', gpu=gpu, question_id=active))
            if stage == 'coarse': prep.coarse_video(root, members)
            elif stage == 'refine': prep.refine_one(root, job)
            else: runtime.process_question(run, job, backend, adapter, observer, gpu, spec)
            events.put(dict(type='completed', gpu=gpu, question_ids=[r['question_id'] for r in members])); active = None
        if stage in ('coarse','refine'):
            durable(root / 'worker_totals' / (stage+'_'+gpu+'_'+str(os.getpid())+'.json'),
                dict(stage=stage, gpu=gpu, model_calls=prep._SCORER.counts, wall_seconds=time.perf_counter()-start,
                     peak_allocated_gib=prep._SCORER.torch.cuda.max_memory_allocated()/2**30, utc=utc()))
    except BaseException as exc:
        stop.set(); traceback.print_exc()
        durable(root / 'failures' / (stage+'_'+gpu+'_'+str(os.getpid())+'.json'),
            dict(question_id=active, error=repr(exc), traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter()-start, utc=utc()))
        events.put(dict(type='failed', gpu=gpu, error=repr(exc)))


def run_stage(root, stage, allowed):
    """接口：自动放行只依据输入/调用正确性；部分完成题固定原物理GPU。"""
    from videoqa_mlvu import preparation as prep, runtime
    root = Path(root); prep.verify(root); data = rows(root); bank = {r['question_id']: r for r in data}
    done = set(); pinned = {}
    # 步骤1：恢复只读取已完成资产或完整返回；不确定调用在加载新模型前即暂停。
    if stage in ('coarse','refine'):
        for r in data:
            tags = ('coarse',) if stage == 'coarse' else ('llava8','qwen32')
            if all(prep.completed(root, t, r['question_id']) for t in tags): done.add(r['question_id'])
    else:
        run = root / 'qa' / stage; spec = runtime.verify_inputs(run)
        for r in data:
            q = r['question_id']
            for group in read_json(run / 'groups' / (q+'.json')): runtime.recover_group(run, r, spec, group)
            cards = {read_json(p)['gpu'] for p in (run / 'calls').glob('*/'+q+'.json')}
            if len(cards) > 1: raise ValueError('Question moved between physical GPUs')
            if cards: pinned[q] = next(iter(cards))
            if all((run / 'results' / m / (q+'.json')).exists() for m in METHODS):
                for m in METHODS: runtime.validate_result(run, r, m, spec)
                done.add(q)
    if len(done) == len(data):
        if stage not in ('coarse','refine'): runtime.audit(root / 'qa' / stage)
        return
    actual, states = choose_gpus(allowed, {g for q,g in pinned.items() if q not in done}, root / 'status.json')
    actual = actual[:min(len(actual), len(data)-len(done))]
    durable(root / 'stage_gpu' / (stage+'.json'), dict(actual=actual, states=states, utc=utc()))
    pilot = [next(r['question_id'] for r in data if r['task_type'] == task) for task in TASKS]
    ctx = mp.get_context('spawn'); events = ctx.Queue(); stop = ctx.Event()
    jobs = {g: ctx.Queue() for g in actual}; children = {}; ready = set(); busy = set(); active = {}
    pending = []; phase = 'loading'; start = time.perf_counter(); initial = len(done)
    def schedule():
        nonlocal pending, phase
        missing = [q for q in pilot if q not in done]; phase = 'pilot' if missing else 'running'
        pending = missing or sorted((q for q in bank if q not in done), key=lambda q: (-bank[q]['declared_duration_seconds'], bank[q]['ordinal']))
        if not missing: durable(root / 'pilot_acceptance' / (stage+'.json'), dict(status='passed', question_ids=pilot, accuracy_not_gate=True, utc=utc()))
    def refill():
        for gpu in actual:
            if gpu in busy: continue
            q = next((q for q in pending if q not in pinned or pinned[q] == gpu), None)
            if q is None: continue
            pending.remove(q); busy.add(gpu)
            if stage == 'coarse':
                members = [r for r in data if r['video_id'] == bank[q]['video_id'] and r['question_id'] not in done]
                for r in members:
                    if r['question_id'] in pending: pending.remove(r['question_id'])
                jobs[gpu].put(members)
            else: jobs[gpu].put(bank[q])
    with exclusive(root / 'locks' / (stage+'.lock')):
        try:
            for gpu in actual:
                child = ctx.Process(target=worker, args=(str(root), stage, gpu, jobs[gpu], events, stop)); child.start(); children[gpu] = child
            # 步骤2：持续写心跳；超时只是观察间隔，不触发重启或重复题目。
            while len(done) < len(data):
                try: event = events.get(timeout=5)
                except queue.Empty: event = None
                if event:
                    gpu = event['gpu']
                    if event['type'] == 'failed': raise RuntimeError(str(event))
                    if event['type'] == 'ready':
                        ready.add(gpu)
                        if len(ready) == len(actual): schedule(); refill()
                    elif event['type'] == 'active': active[gpu] = event['question_id']
                    elif event['type'] == 'completed':
                        done.update(event['question_ids']); busy.remove(gpu); active.pop(gpu, None)
                        if phase == 'pilot' and not pending and not busy: schedule()
                        refill()
                if len(done) < len(data):
                    for gpu, child in children.items():
                        if child.exitcode is not None: raise RuntimeError('Worker exited before completion: '+gpu)
                elapsed = time.perf_counter()-start; new = len(done)-initial
                durable(root / 'status.json', dict(stage=stage, status=phase, completed=len(done), total=len(data),
                    pid=os.getpid(), sid=os.getsid(0), gpus=actual, active=active, wall_seconds=elapsed,
                    eta_seconds=None if new < 7 else elapsed/new*(len(data)-len(done)), utc=utc()))
        except BaseException as exc:
            stop.set(); durable(root / 'status.json', dict(stage=stage, status='paused', completed=len(done), error=repr(exc), utc=utc())); raise
        finally:
            # 步骤3：只收尾本轮子进程，不触碰其他任务；失败现场和账本全部保留。
            for gpu in actual: jobs[gpu].put(None)
            for child in children.values():
                child.join(20)
                if child.is_alive(): child.terminate(); child.join()
    if stage not in ('coarse','refine'): runtime.audit(root / 'qa' / stage)
    durable(root / 'stage_completion' / (stage+'.json'), dict(status='completed', questions=len(data),
        actual_gpus=actual, wall_seconds=time.perf_counter()-start, utc=utc()))
