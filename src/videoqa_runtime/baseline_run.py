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

# 数据集注册表：每项定义一次基线运行的题目范围与冻结条件。
# development50 为默认项，行为与历史50题运行逐字段一致；full2700 为全量公开测试集。
# manifest_sha256 在清单生成后冻结（scripts/build_video_mme_full_assets.py audit 阶段打印）。
DATASETS = {
    'development50': dict(
        manifest_path='data/manifests/video_mme_development.json',
        manifest_sha256=DEVELOPMENT_SHA256,
        asset_audit_path='data/manifests/video_mme_asset_validation.json',
        question_count=50, unique_video_count=50,
        protocol_version='videomme50-baselines-v2-budgetcap',
        scope='50-question development set; no application-cache hits; OS cache uncontrolled; resident-model E2E'),
    'full2700': dict(
        manifest_path='data/manifests/video_mme_full_2700.json',
        manifest_sha256='eca56f202afcf3cf7e3e3d608be6f95ec1980a7967c117ff379832ce17f3bdd0',
        asset_audit_path='data/manifests/video_mme_full_asset_audit.json',
        question_count=2700, unique_video_count=900,
        protocol_version='videomme-full-baselines-v2-budgetcap',
        scope='full 2700-question public Video-MME test set; no application-cache hits; OS cache uncontrolled; resident-model E2E'),
    # 2026-09-17 修订后的补跑集：r1 因 16 帧硬性预算暂停的 6 题（2 个短视频）。
    # 行对象与 full2700 清单逐字段一致；预期与其 5388 个已完成结果合并为完整 5400。
    'full2700_completion': dict(
        manifest_path='data/manifests/video_mme_full_completion_6.json',
        manifest_sha256='6ad2b2836ba566420db0c4afdbb91e60cc3cf2849df59386ff841cd343bd7ae8',
        asset_audit_path='data/manifests/video_mme_full_asset_audit.json',
        question_count=6, unique_video_count=2,
        protocol_version='videomme-full-completion6-v1',
        prior_smoke_result='excluded; 12 newly measured answers completing videomme_full_uniform_topk_20260916_r1 (5388/5400)',
        scope='6-question completion of the full 2700-question run under the 2026-09-17 budget-cap amendment; '
              'no application-cache hits; OS cache uncontrolled; resident-model E2E'),
}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def protocol(dataset):
    """冻结本轮运行协议指纹；dataset 决定题目范围，其余条件两种数据集完全一致。"""
    files = sorted((ROOT / 'src/videoqa_runtime').glob('*.py')) + [ROOT / 'scripts/run_baselines.py', ROOT / 'scripts/summarize_baselines.py']
    return dict(version=dataset['protocol_version'], dataset_manifest=dataset['manifest_path'],
                dataset_manifest_sha256=dataset['manifest_sha256'], question_count=dataset['question_count'],
                llava_source_commit=SOURCE_COMMIT, methods=list(METHODS),
                candidate_fps=1, phase_seconds=0.25,
                frame_budget=16,
                frame_budget_semantics='maximum cap (2026-09-17 amendment): videos with fewer distinct '
                                        'candidates use all of them; with >=16 candidates behavior is '
                                        'identical to the original fixed budget',
                visual_tokens_per_frame=210,
                llava_dtype='bfloat16', llava_attention='sdpa', blip_dtype='float32', blip_batch_size=16,
                max_new_tokens=8, seed=2027, cpu_threads=2, video_decode_threads=2,
                cache_mode='no application cache; OS page cache uncontrolled',
                method_order='zero-based even ordinal uniform-first, odd ordinal topk-first',
                timing_scope='models resident; raw video open to parsed answer; no result writes inside interval',
                prior_smoke_result=dataset.get('prior_smoke_result',
                    f'excluded; {dataset["question_count"] * 2} newly measured answers'), max_extra_retries=10,
                scope=dataset['scope'],
                code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in files},
                model_inventory_sha256=sha256(ROOT / 'configs/local_model_inventory.json'),
                environment_lock_sha256=sha256(ROOT / 'requirements-runtime.lock.txt'))


def preflight(rows, dataset=None):
    """运行前冻结条件核验：清单哈希、视频字节哈希、模型身份、包版本与BLIP输入长度。
    步骤1 清单规模与唯一性；步骤2 清单文件哈希；步骤3 逐视频字节哈希；
    步骤4 模型文件哈希；步骤5 包版本与离线配置；步骤6 BLIP问题Token上限。"""
    started = time.perf_counter()
    # 兼容研究方法既有的preflight(rows)调用；默认仍为冻结开发50题。
    dataset = DATASETS['development50'] if dataset is None else dataset
    expected = dataset['question_count']
    if len(rows) != expected or len({r['question_id'] for r in rows}) != expected or len({r['video_id'] for r in rows}) != dataset['unique_video_count']:
        raise ValueError(f'Expected {expected} distinct frozen questions on {dataset["unique_video_count"]} videos')
    if sha256(ROOT / dataset['manifest_path']) != dataset['manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    audit = read_json(ROOT / dataset['asset_audit_path'])
    asset_map = {f['path']: f for f in audit['files']}
    for row in rows:
        path = 'data/' + row['video_relative_path']
        if sha256(ROOT / path) != asset_map[path]['sha256']:
            raise ValueError(f'Video bytes changed: {row["video_id"]}')
    print(f'Preflight: all {len(asset_map)} video hashes match.', flush=True)
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


def execute(run_id, requested_gpus, resume=False, dataset_name='development50'):
    if dataset_name not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('Invalid run ID')
    run_dir = ROOT / 'outputs/baselines' / run_id
    run_dir.mkdir(parents=True, exist_ok=resume)
    with (run_dir / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _execute_locked(run_dir, requested_gpus, resume, DATASETS[dataset_name])


def _execute_locked(run_dir, requested_gpus, resume, dataset):
    started = time.perf_counter()
    for directory in ('logs', 'workers', 'attempts', 'results/uniform', 'results/topk', 'code_snapshot'):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    spec = protocol(dataset)
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
    expected_results = dataset['question_count'] * len(METHODS)
    rows = read_json(ROOT / dataset['manifest_path'])['rows']
    completed = check_recovery(run_dir, rows, spec_hash)
    if len(completed) == expected_results:
        print(f'All {expected_results} results already complete; no generation repeated.', flush=True)
        return
    write_json(run_dir / 'pid.json', dict(pid=os.getpid(), session_id=os.getsid(0), started_at_utc=utc(), executable=sys.executable))
    write_json(run_dir / 'status.json', dict(status='preflight', completed_results=len(completed), pid=os.getpid(), updated_at_utc=utc()))
    verification = preflight(rows, dataset)
    write_json(run_dir / 'preflight.json', verification)
    gpus, gpu_states = available_gpus(requested_gpus)
    owner_path = run_dir / 'owners.json'
    owners = read_json(owner_path) if owner_path.exists() else {}
    for qid, gpu in owners.items():
        if any((qid, m) not in completed for m in METHODS) and gpu not in gpus:
            raise RuntimeError(f'Incomplete pair owns currently unavailable GPU {gpu}: {qid}')
    write_json(run_dir / 'gpu_allocation.json', dict(requested=requested_gpus, selected=gpus, before_launch=gpu_states))
    durations = {v['video_id']: v['duration_seconds'] for v in read_json(ROOT / dataset['asset_audit_path'])['videos']}
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
        record = dict(status=phase, completed_results=len(completed), expected_results=expected_results,
                      completed_by_method={m: sum(key[1] == m for key in completed) for m in METHODS},
                      extra_retries=extra_retries, active=active, progress=progress,
                      available_workers=len(gpus), elapsed_seconds=time.perf_counter() - started,
                      pid=os.getpid(), updated_at_utc=utc())
        write_json(run_dir / 'status.json', record)
        print(f'[{phase}] {len(completed)}/{expected_results} results; retries={extra_retries}; active={len(active)}; elapsed={record["elapsed_seconds"]:.1f}s', flush=True)

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
        # 在途任务退出后，暂停态也重新统计落盘结果；不确定生成仍保持暂停。
        try:
            completed = check_recovery(run_dir, rows, spec_hash)
        except (ValueError, OSError) as recovery_error:
            final_status = 'paused'
            error = f'{error or ""}; final verification: {recovery_error}'
        phase = final_status
        publish(True)
        write_json(run_dir / 'execution.json', dict(status=final_status, error=error, started_at_utc=verification['checked_at_utc'],
                   finished_at_utc=utc(), selected_gpus=gpus, preflight_seconds=verification['seconds'],
                   total_active_wall_seconds=wall_before + time.perf_counter() - started,
                   completed_results=len(completed), extra_retries=extra_retries, resume=resume))
        event_log.close()
    if error:
        raise RuntimeError(error)
