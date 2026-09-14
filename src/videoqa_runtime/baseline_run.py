"""Detached-safe coordinator: one persistent process per GPU, pilot gate, durable results."""
import datetime
import fcntl
import importlib.metadata
import multiprocessing as mp
import os
import queue
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .common import ROOT, SOURCE_COMMIT, DEVELOPMENT_SHA256, read_json, sha256, write_json, offline_environment
from .baseline_records import METHODS, check_recovery, result_path, validate_pair
from .baseline_worker import worker_main


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def protocol():
    files = sorted((ROOT / 'src/videoqa_runtime').glob('*.py')) + [ROOT / 'scripts/run_baselines.py', ROOT / 'scripts/summarize_baselines.py']
    return dict(version='videomme50-baselines-v1', dataset_manifest_sha256=DEVELOPMENT_SHA256,
                llava_source_commit=SOURCE_COMMIT, methods=list(METHODS), question_count=50,
                candidate_fps=1, phase_seconds=0.25, frames=16, expected_visual_tokens=3360,
                llava_dtype='bfloat16', llava_attention='sdpa', blip_dtype='float32', blip_batch_size=16,
                max_new_tokens=8, seed=2027, cpu_threads=2, video_decode_threads=2,
                cache_mode='no application cache; OS page cache uncontrolled',
                method_order='zero-based even ordinal uniform-first, odd ordinal topk-first',
                timing_scope='models resident; raw video open to parsed answer; no result writes inside interval',
                prior_smoke_result='excluded; 100 newly measured answers', max_extra_retries=10,
                code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in files},
                model_inventory_sha256=sha256(ROOT / 'configs/local_model_inventory.json'),
                environment_lock_sha256=sha256(ROOT / 'requirements-runtime.lock.txt'))


def preflight(rows):
    started = time.perf_counter()
    if len(rows) != 50 or len({r['question_id'] for r in rows}) != 50 or len({r['video_id'] for r in rows}) != 50:
        raise ValueError('Expected 50 distinct frozen questions/videos')
    if sha256(ROOT / 'data/manifests/video_mme_development.json') != DEVELOPMENT_SHA256:
        raise ValueError('Frozen manifest changed')
    audit = read_json(ROOT / 'data/manifests/video_mme_asset_validation.json')
    asset_map = {f['path']: f for f in audit['files']}
    for row in rows:
        path = 'data/' + row['video_relative_path']
        if sha256(ROOT / path) != asset_map[path]['sha256']:
            raise ValueError(f'Video bytes changed: {row["video_id"]}')
    print('Preflight: all 50 video hashes match.', flush=True)
    inventory = read_json(ROOT / 'configs/local_model_inventory.json')
    for model_name, model in inventory['models'].items():
        for filename, expected in model['files'].items():
            path = Path(model['path']) / filename
            if sha256(path) != expected['sha256']:
                raise ValueError(f'Model bytes changed: {model_name}/{filename}')
        print(f'Preflight: {model_name} file hashes match.', flush=True)
    expected_packages = read_json(ROOT / 'configs/runtime_environment.json')['packages']
    for name in ('torch', 'torchvision', 'transformers', 'tokenizers', 'numpy', 'av', 'safetensors'):
        if importlib.metadata.version(name) != expected_packages[name]:
            raise ValueError(f'Runtime package changed: {name}')
    offline_environment()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(inventory['models']['blip']['path'], local_files_only=True)
    lengths = [len(tokenizer(r['question'], truncation=False)['input_ids']) for r in rows]
    if max(lengths) > 512:
        raise ValueError('BLIP question exceeds position limit')
    return dict(checked_at_utc=utc(), seconds=time.perf_counter() - started,
                video_hashes={r['video_id']: asset_map['data/' + r['video_relative_path']]['sha256'] for r in rows},
                blip_question_token_lengths=lengths, package_versions_verified=True)


def available_gpus(requested):
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    states = {}
    for line in text.splitlines():
        index, memory, utilization = [s.strip() for s in line.split(',')]
        states[index] = dict(memory_mib=int(memory), utilization=int(utilization))
    selected = [g for g in requested if g in states and states[g]['memory_mib'] < 256 and states[g]['utilization'] == 0]
    if not selected:
        raise RuntimeError('No requested GPU is currently idle')
    return selected, states


def execute(run_id, requested_gpus, resume=False):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('Invalid run ID')
    run_dir = ROOT / 'outputs/baselines' / run_id
    run_dir.mkdir(parents=True, exist_ok=resume)
    with (run_dir / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _execute_locked(run_dir, requested_gpus, resume)


def _execute_locked(run_dir, requested_gpus, resume):
    started = time.perf_counter()
    for directory in ('logs', 'workers', 'attempts', 'results/uniform', 'results/topk', 'code_snapshot'):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    spec = protocol()
    spec_path = run_dir / 'protocol.json'
    if resume:
        if read_json(spec_path) != spec:
            raise RuntimeError('Resume refused: source, model identity, environment lock, or protocol changed')
    else:
        write_json(spec_path, spec)
        for name in spec['code_sha256']:
            target = run_dir / 'code_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        for name in ('llava_source_manifest.json', 'local_model_inventory.json', 'runtime_environment.json'):
            shutil.copy2(ROOT / 'configs' / name, run_dir / name)
    spec_hash = sha256(spec_path)
    rows = read_json(ROOT / 'data/manifests/video_mme_development.json')['rows']
    completed = check_recovery(run_dir, rows, spec_hash)
    if len(completed) == 100:
        print('All 100 results already complete; no generation repeated.', flush=True)
        return
    write_json(run_dir / 'pid.json', dict(pid=os.getpid(), session_id=os.getsid(0), started_at_utc=utc(), executable=sys.executable))
    write_json(run_dir / 'status.json', dict(status='preflight', completed_results=len(completed), pid=os.getpid(), updated_at_utc=utc()))
    verification = preflight(rows)
    write_json(run_dir / 'preflight.json', verification)
    gpus, gpu_states = available_gpus(requested_gpus)
    owner_path = run_dir / 'owners.json'
    owners = read_json(owner_path) if owner_path.exists() else {}
    for qid, gpu in owners.items():
        if any((qid, m) not in completed for m in METHODS) and gpu not in gpus:
            raise RuntimeError(f'Incomplete pair owns currently unavailable GPU {gpu}: {qid}')
    write_json(run_dir / 'gpu_allocation.json', dict(requested=requested_gpus, selected=gpus, before_launch=gpu_states))
    durations = {v['video_id']: v['duration_seconds'] for v in read_json(ROOT / 'data/manifests/video_mme_asset_validation.json')['videos']}
    jobs = [dict(row=row, order=i, resumed=resume, video_sha256=verification['video_hashes'][row['video_id']]) for i, row in enumerate(rows)]
    ctx = mp.get_context('spawn')
    events, stop = ctx.Queue(), ctx.Event()
    queues = {gpu: ctx.Queue() for gpu in gpus}
    processes = {gpu: ctx.Process(target=worker_main, args=(gpu, queues[gpu], events, stop, str(run_dir), spec_hash), name=f'baseline-gpu-{gpu}') for gpu in gpus}
    active, idle, progress = {}, set(), {}
    retry_file = run_dir / 'retry_budget.json'
    extra_retries = read_json(retry_file)['used'] if retry_file.exists() else 0
    last_status = 0.0
    phase = 'worker_startup'
    wall_before = read_json(run_dir / 'execution.json').get('total_active_wall_seconds', 0.0) if resume and (run_dir / 'execution.json').exists() else 0.0
    event_log = (run_dir / 'events.jsonl').open('a', buffering=1)

    def publish(force=False):
        nonlocal last_status
        if not force and time.perf_counter() - last_status < 10:
            return
        last_status = time.perf_counter()
        record = dict(status=phase, completed_results=len(completed), expected_results=100,
                      completed_by_method={m: sum(key[1] == m for key in completed) for m in METHODS},
                      extra_retries=extra_retries, active=active, progress=progress,
                      available_workers=len(gpus), elapsed_seconds=time.perf_counter() - started,
                      pid=os.getpid(), updated_at_utc=utc())
        write_json(run_dir / 'status.json', record)
        print(f'[{phase}] {len(completed)}/100 results; retries={extra_retries}; active={len(active)}; elapsed={record["elapsed_seconds"]:.1f}s', flush=True)

    def event():
        while True:
            publish()
            try:
                item = events.get(timeout=5)
            except queue.Empty:
                dead = {g: p.exitcode for g, p in processes.items() if p.exitcode is not None}
                if dead:
                    raise RuntimeError(f'Worker exited unexpectedly; inspect attempt journals: {dead}')
                continue
            item['observed_at_utc'] = utc()
            import json
            event_log.write(json.dumps(item, ensure_ascii=False) + '\n')
            return item

    def run_phase(phase_jobs):
        nonlocal extra_retries
        pending = [j for j in phase_jobs if any((j['row']['question_id'], m) not in completed for m in METHODS)]
        while pending or active:
            for gpu in list(sorted(idle)):
                eligible = next((i for i, job in enumerate(pending) if owners.get(job['row']['question_id'], gpu) == gpu), None)
                if eligible is None:
                    continue
                job = pending.pop(eligible)
                qid = job['row']['question_id']
                owners[qid] = gpu
                write_json(owner_path, owners)
                active[gpu] = qid
                idle.remove(gpu)
                queues[gpu].put(job)
            item = event()
            gpu = item['gpu']
            kind = item['type']
            if kind == 'method_completed':
                completed.add((item['question_id'], item['method']))
            elif kind in ('progress', 'method_started'):
                progress[gpu] = item
            elif kind == 'job_completed':
                active.pop(gpu, None)
                progress.pop(gpu, None)
                idle.add(gpu)
            elif kind == 'job_failed':
                active.pop(gpu, None)
                idle.add(gpu)
                if not item['retryable'] or extra_retries >= 10:
                    raise RuntimeError(f'Run paused after failure: {item}')
                extra_retries += 1
                write_json(retry_file, dict(used=extra_retries, maximum=10))
                pending.insert(0, next(j for j in phase_jobs if j['row']['question_id'] == item['question_id']))
            elif kind == 'worker_failed':
                raise RuntimeError(f'Worker failure: {item}')
            publish()

    final_status = 'paused'
    error = None
    try:
        for process in processes.values():
            process.start()
        while len(idle) < len(gpus):
            item = event()
            if item['type'] != 'ready':
                raise RuntimeError(f'Worker did not initialize: {item}')
            idle.add(item['gpu'])
        phase = 'pilot'
        publish(True)
        pilot_started = time.perf_counter()
        run_phase(jobs[:5])
        for row in rows[:5]:
            pair = {m: read_json(result_path(run_dir, m, row['question_id'])) for m in METHODS}
            validate_pair(pair, row, spec_hash)
        write_json(run_dir / 'pilot_acceptance.json', dict(status='passed', questions=[r['question_id'] for r in rows[:5]],
                   results=10, criterion='input integrity, finite outputs, token counts, and timing reconciliation; not accuracy',
                   wall_seconds=time.perf_counter() - pilot_started, accepted_at_utc=utc()))
        phase = 'full_run'
        publish(True)
        run_phase(sorted(jobs[5:], key=lambda j: -durations[j['row']['video_id']]))
        for row in rows:
            validate_pair({m: read_json(result_path(run_dir, m, row['question_id'])) for m in METHODS}, row, spec_hash)
        final_status = 'completed'
    except BaseException as caught:
        error = f'{type(caught).__name__}: {caught}'
        print(error, flush=True)
    finally:
        stop.set()
        for q in queues.values():
            q.put(None)
        # Let already running inference finish and save; never kill an uncertain generation.
        while any(p.is_alive() for p in processes.values()):
            for p in processes.values():
                if p.pid is not None:
                    p.join(timeout=1)
        completed = check_recovery(run_dir, rows, spec_hash) if final_status == 'completed' else completed
        phase = final_status
        publish(True)
        write_json(run_dir / 'execution.json', dict(status=final_status, error=error, started_at_utc=verification['checked_at_utc'],
                   finished_at_utc=utc(), selected_gpus=gpus, preflight_seconds=verification['seconds'],
                   total_active_wall_seconds=wall_before + time.perf_counter() - started,
                   completed_results=len(completed), extra_retries=extra_retries, resume=resume))
        event_log.close()
    if error:
        raise RuntimeError(error)
