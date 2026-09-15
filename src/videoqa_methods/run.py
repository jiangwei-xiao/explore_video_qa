import datetime
import fcntl
import importlib.metadata
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import sys
import time
from pathlib import Path

from videoqa_runtime.common import ROOT, read_json, write_json, sha256
from videoqa_runtime.baseline_run import preflight, available_gpus
from .model_ops import SCOPE_PROMPT
from .records import VARIANTS, result_path, validate_group, check_recovery
from .worker import worker_main

BASELINE = ROOT / 'outputs/baselines/videomme50_uniform_topk_20260914_r1'
CORE_FILES = ['__init__.py', 'algorithm.py', 'refinement.py', 'model_ops.py', 'pipeline.py', 'records.py', 'worker.py', 'run.py']


def utc():
    """返回带时区的UTC时间，与单调时钟分别承担审计时间与耗时测量。"""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def execution_protocol():
    """接口：冻结执行代码、方法配置、依赖和历史基线身份；分析绘图代码另行记录，不混入方法推理指纹。"""
    cfg = read_json(ROOT / 'configs/method_v1.json')
    paths = [ROOT / 'src/videoqa_methods' / name for name in CORE_FILES]
    paths += [ROOT / 'scripts/run_method_experiments.py']
    return dict(version='videoqa-method-experiment-v1', config=cfg, scope_prompt=SCOPE_PROMPT,
                execution_code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in paths},
                baseline_protocol_sha256=sha256(BASELINE / 'protocol.json'),
                baseline_summary_sha256=sha256(BASELINE / 'summary.json'),
                runtime_lock_sha256=sha256(ROOT / 'requirements-runtime.lock.txt'),
                method_lock_sha256=sha256(ROOT / 'requirements-method.lock.txt'),
                model_inventory_sha256=sha256(ROOT / 'configs/local_model_inventory.json'),
                questions=50, final_answers=200, scope_calls=100, maximum_retries=10,
                primary_application_cache=False, diagnostic_donor='C',
                fine_cap_rule='deduplicate; if more than 12, closest to hotspot center then earlier PTS',
                baseline_comparison='frozen historical results; latency comparison descriptive across batches')


def verify_reference_and_baseline():
    """核验固定WFS参考、历史基线代码/结果及新增包版本，保持旧基线可复用且未被改写。"""
    cfg = read_json(ROOT / 'configs/method_v1.json')
    if sha256(ROOT / 'third_party/WFS-SB-reference/core.py') != cfg['wfs_reference_sha256']:
        raise RuntimeError('WFS reference identity mismatch')
    baseline = read_json(BASELINE / 'protocol.json')
    for name, digest in baseline['code_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f'Frozen baseline execution code changed: {name}')
    summary = read_json(BASELINE / 'summary.json')
    for name, digest in summary['result_sha256'].items():
        if sha256(BASELINE / name) != digest:
            raise RuntimeError(f'Frozen baseline result changed: {name}')
    for name, version in [('PyWavelets', '1.8.0'), ('scipy', '1.15.3'), ('matplotlib', '3.10.1')]:
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f'Method dependency differs: {name}')


def execute(run_id, requested_gpus, resume=False):
    """外部运行接口：校验运行编号并取得排他文件锁，然后执行或恢复该轮实验。"""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('Invalid run ID')
    run = ROOT / 'outputs/methods' / run_id
    run.mkdir(parents=True, exist_ok=resume)
    with (run / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute_locked(run, requested_gpus, resume)
    return run


def execute_locked(run, requested_gpus, resume):
    """核心调度流程：资产核验、启动每卡常驻进程、前5题门槛、剩余45题和安全收尾；只在四组结果全部通过核验后标记完成。"""
    # 步骤1：固定运行指纹并检查恢复状态；新运行归档执行代码与依赖。
    started, start_utc = time.perf_counter(), utc()
    for name in ['logs', 'workers', 'attempts', 'features', 'code_snapshot', *[f'results/{v}' for v in VARIANTS]]:
        (run / name).mkdir(parents=True, exist_ok=True)
    spec = execution_protocol()
    if resume:
        if read_json(run / 'protocol.json') != spec:
            raise RuntimeError('Cannot resume across changed execution code, dependencies, or protocol')
    else:
        write_json(run / 'protocol.json', spec)
        write_json(run / 'method_config.json', spec['config'])
        for name in spec['execution_code_sha256']:
            target = run / 'code_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        for name in ('requirements-runtime.lock.txt', 'requirements-method.lock.txt', 'requirements-method.txt', 'requirements-method-constraints.txt'):
            shutil.copy2(ROOT / name, run / name)
        for name in ('local_model_inventory.json', 'runtime_environment.json', 'llava_source_manifest.json'):
            shutil.copy2(ROOT / 'configs' / name, run / name)
    fingerprint = sha256(run / 'protocol.json')
    rows = read_json(ROOT / 'data/manifests/video_mme_development.json')['rows']
    done = check_recovery(run, rows, fingerprint)
    if len(done) == 200:
        print('All 200 results complete; no answer or classification repeated.', flush=True)
        return
    write_json(run / 'pid.json', dict(pid=os.getpid(), session_id=os.getsid(0), started_at_utc=start_utc, executable=sys.executable))
    write_json(run / 'status.json', dict(status='preflight', completed=len(done), pid=os.getpid(), updated_at_utc=utc()))
    # 步骤2：验证旧基线、源码、模型与视频身份，然后保存只读比较副本。
    verify_reference_and_baseline()
    evidence = preflight(rows)
    write_json(run / 'preflight.json', evidence)
    if not (run / 'baseline_reference').exists():
        shutil.copytree(BASELINE / 'results', run / 'baseline_reference/results')
        for name in ('summary.json', 'protocol.json', 'review.json'):
            shutil.copy2(BASELINE / name, run / 'baseline_reference' / name)
    # 步骤3：只使用当前空闲GPU，恢复时保持未完成视频的原GPU归属。
    gpus, states = available_gpus(requested_gpus)
    write_json(run / 'gpu_allocation.json', dict(requested=requested_gpus, selected=gpus, initial_states=states))
    owner_file = run / 'owners.json'
    owners = read_json(owner_file) if owner_file.exists() else {}
    for qid, gpu in owners.items():
        if any((qid, v) not in done for v in VARIANTS) and gpu not in gpus:
            raise RuntimeError(f'Incomplete group belongs to unavailable GPU {gpu}: {qid}')
    durations = {x['video_id']: x['duration_seconds'] for x in read_json(ROOT / 'data/manifests/video_mme_asset_validation.json')['videos']}
    jobs = [dict(row=r, order=i, video_sha256=evidence['video_hashes'][r['video_id']]) for i, r in enumerate(rows)]
    ctx = mp.get_context('spawn')
    events, stop = ctx.Queue(), ctx.Event()
    queues = {g: ctx.Queue() for g in gpus}
    processes = {g: ctx.Process(target=worker_main, args=(g, queues[g], events, stop, str(run), fingerprint)) for g in gpus}
    idle, active, progress = set(), {}, {}
    retries = read_json(run / 'retry_budget.json')['used'] if (run / 'retry_budget.json').exists() else 0
    log = (run / 'events.jsonl').open('a', buffering=1)
    phase, last = 'startup', 0.0

    def publish(force=False):
        """每10秒原子写入可观察状态；由主进程落盘，避免将日志写入开销混入工作进程单题计时。"""
        nonlocal last
        if not force and time.perf_counter() - last < 10:
            return
        last = time.perf_counter()
        status = dict(status=phase, completed=len(done), expected=200,
                      by_variant={v: sum(key[1] == v for key in done) for v in VARIANTS},
                      extra_retries=retries, active=active, progress=progress,
                      elapsed_seconds=time.perf_counter() - started, updated_at_utc=utc(), pid=os.getpid())
        write_json(run / 'status.json', status)
        print(f'[{phase}] {len(done)}/200 answers; retries={retries}; active={len(active)}; elapsed={status["elapsed_seconds"]:.1f}s', flush=True)

    def next_event():
        """等待具体工作进程事件，同时刷新状态并检查异常退出；观察超时本身不视为任务停止。"""
        while True:
            publish()
            try:
                item = events.get(timeout=5)
            except queue.Empty:
                dead = {g: p.exitcode for g, p in processes.items() if p.exitcode is not None}
                if dead:
                    raise RuntimeError(f'Worker exited; inspect journals before recovery: {dead}')
                continue
            item['observed_at_utc'] = utc()
            log.write(json.dumps(item, ensure_ascii=False) + '\n')
            return item

    def run_phase(phase_jobs):
        """按空闲GPU派发任务并固定同视频GPU归属，处理完成与失败事件；只对明确可重试的失败使用总重试预算。"""
        nonlocal retries
        pending = [j for j in phase_jobs if any((j['row']['question_id'], v) not in done for v in VARIANTS)]
        while pending or active:
            for g in sorted(idle):
                index = next((i for i, j in enumerate(pending) if owners.get(j['row']['question_id'], g) == g), None)
                if index is None:
                    continue
                job = pending.pop(index)
                qid = job['row']['question_id']
                owners[qid] = g
                write_json(owner_file, owners)
                active[g] = qid
                idle.remove(g)
                queues[g].put(job)
            item = next_event()
            g = item['gpu']
            if item['type'] == 'completed':
                done.add((item['question_id'], item['variant']))
            elif item['type'] in ('progress', 'started'):
                progress[g] = item
            elif item['type'] == 'job_completed':
                active.pop(g, None); progress.pop(g, None); idle.add(g)
            elif item['type'] == 'job_failed':
                active.pop(g, None); idle.add(g)
                if not item['retryable'] or retries >= 10:
                    raise RuntimeError(f'Method experiment paused: {item}')
                retries += 1
                write_json(run / 'retry_budget.json', dict(used=retries, maximum=10))
                pending.insert(0, next(j for j in phase_jobs if j['row']['question_id'] == item['question_id']))
            elif item['type'] == 'worker_failed':
                raise RuntimeError(str(item))

    status, error = 'paused', None
    try:
        # 步骤4：启动常驻进程，全部预热就绪后才开始前5题验收。
        for p in processes.values():
            p.start()
        while len(idle) != len(gpus):
            item = next_event()
            if item['type'] != 'ready':
                raise RuntimeError(str(item))
            idle.add(item['gpu'])
        phase = 'pilot'; publish(True)
        pilot_start = time.perf_counter()
        run_phase(jobs[:5])
        for row in rows[:5]:
            qid = row['question_id']
            group = {v: read_json(result_path(run, v, qid)) for v in VARIANTS}
            validate_group(group, row, fingerprint, read_json(BASELINE / 'results/topk' / f'{qid}.json'))
        write_json(run / 'pilot_acceptance.json', dict(status='passed', questions=[r['question_id'] for r in rows[:5]],
                   answers=20, scope_calls=10, elapsed_seconds=time.perf_counter() - pilot_start,
                   criterion='input, scope consistency, candidate identity, quota and token integrity; not accuracy'))
        # 步骤5：输入与机制验收通过才放行剩余45题，不按正确率挑选是否继续。
        phase = 'full_run'; publish(True)
        run_phase(sorted(jobs[5:], key=lambda j: -durations[j['row']['video_id']]))
        for row in rows:
            qid = row['question_id']
            validate_group({v: read_json(result_path(run, v, qid)) for v in VARIANTS}, row, fingerprint,
                           read_json(BASELINE / 'results/topk' / f'{qid}.json'))
        # 步骤6：全量四组核验通过后标记完成，随后等待进程退出并保存整批成本。
        status = 'completed'
    except BaseException as e:
        error = f'{type(e).__name__}: {e}'
        print(error, flush=True)
    finally:
        stop.set()
        for q in queues.values(): q.put(None)
        while any(p.is_alive() for p in processes.values()):
            for p in processes.values():
                if p.pid is not None: p.join(timeout=1)
        phase = status; publish(True)
        write_json(run / 'execution.json', dict(status=status, error=error, start_utc=start_utc, finish_utc=utc(),
                   wall_seconds=time.perf_counter() - started, preflight_seconds=evidence['seconds'],
                   answers=len(done), extra_retries=retries, gpus=gpus, resume=resume))
        log.close()
    if error:
        raise RuntimeError(error)
