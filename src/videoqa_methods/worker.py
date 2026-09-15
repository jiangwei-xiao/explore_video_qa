import contextlib
import datetime
import os
import time
import traceback
from pathlib import Path
import numpy as np

from videoqa_runtime.common import ROOT, read_json, write_json, sha256, offline_environment
from videoqa_runtime.baseline_selection import ProtocolError
from .model_ops import FeatureScorer, ScopeClassifier
from .pipeline import select_method, select_diagnostic
from .records import result_path, validate_result


def utc():
    """返回带时区的UTC时间，用于可追溯的事件与尝试记录。"""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def order_for(index):
    """按冻结样本序号轮换A/B/C顺序，并保证D紧随C，避免固定执行顺序偏差。"""
    return [('A', 'B', 'C', 'D'), ('B', 'C', 'D', 'A'), ('C', 'D', 'A', 'B')][index % 3]


def worker_main(gpu, tasks, events, stop, run, fingerprint):
    """工作进程接口：独占一张指定GPU，配置离线环境及日志；致命错误通过事件队列通知调度器。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    offline_environment()
    run = Path(run)
    with (run / 'logs' / f'gpu_{gpu}.log').open('a', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                loop(gpu, tasks, events, stop, run, fingerprint)
            except BaseException as e:
                traceback.print_exc()
                events.put(dict(type='worker_failed', gpu=gpu, error=f'{type(e).__name__}: {e}'))


def loop(gpu, tasks, events, stop, run, fingerprint):
    """核心执行循环：模型常驻，逐题依次执行四变体；计时前持久化尝试状态，计时后保存结果与特征，已完成结果直接跳过。"""
    import torch
    from videoqa_runtime.llava_backend import LlavaBackend
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cfg = read_json(run / 'method_config.json')
    # 步骤1：每卡只加载一套常驻模型，合成前向预热不计入开发题调用。
    start = time.perf_counter()
    backend = LlavaBackend()
    scorer = FeatureScorer(read_json(ROOT / 'configs/local_model_inventory.json')['models']['blip']['path'])
    classifier = ScopeClassifier(backend)
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    backend.warmup()
    scorer.warmup()
    startup = dict(gpu=gpu, pid=os.getpid(), load_seconds=load_seconds, warmup_seconds=time.perf_counter() - start,
                   llava_synthetic_forwards=1, blip_synthetic_forwards=2, backend=backend.load_report,
                   resident_allocated_gib=torch.cuda.memory_allocated() / 2**30)
    write_json(run / 'workers' / f'gpu_{gpu}.json', startup)
    events.put(dict(type='ready', gpu=gpu))
    # 步骤2：领取整组视频任务，按预定轮换顺序处理；完成条目直接复用。
    while not stop.is_set():
        job = tasks.get()
        if job is None:
            return
        row = job['row']
        qid = row['question_id']
        for variant in order_for(job['order']):
            if stop.is_set():
                return
            destination = result_path(run, variant, qid)
            if destination.exists():
                validate_result(read_json(destination), row, fingerprint)
                continue
            # 步骤3：D读取C的冻结候选池，该读取位于D新增成本的计时区间之外。
            donor_path = result_path(run, 'C', qid)
            donor = read_json(donor_path) if variant == 'D' else None
            donor_sha = sha256(donor_path) if donor is not None else None
            attempt_dir = run / 'attempts' / qid
            attempt_dir.mkdir(parents=True, exist_ok=True)
            number = len(list(attempt_dir.glob(f'{variant}.*.json'))) + 1
            journal = attempt_dir / f'{variant}.{number:02}.json'
            with journal.open('x') as f:
                f.write('{}\n')
            attempt = dict(question_id=qid, variant=variant, attempt=number, gpu=gpu,
                           started_at_utc=utc(), status='running_may_generate')
            # 步骤4：计时前持久化“可能生成”状态，异常断线时宁可停查也不重复调用。
            write_json(journal, attempt)
            events.put(dict(type='started', gpu=gpu, question_id=qid, variant=variant))
            scope_state, answer_state = {}, {}
            measured = None
            try:
                torch.cuda.synchronize()
                resident = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                # 步骤5：直接墙钟覆盖分类/选帧到答案解析，区间内不写结果文件。
                start = time.perf_counter()
                if variant == 'D':
                    frames, pool, times = select_diagnostic(donor)
                    features = None
                else:
                    frames, pool, features, times = select_method(
                        ROOT / 'data' / row['video_relative_path'], row['question'], variant, scorer, classifier, cfg,
                        scope_state, lambda n: events.put(dict(type='progress', gpu=gpu, question_id=qid, variant=variant, candidates=n)))
                torch.cuda.synchronize()
                selection_end = time.perf_counter()
                selector_peak = torch.cuda.max_memory_allocated()
                before_qa = torch.cuda.memory_allocated()
                qa_start = time.perf_counter()
                answer = backend.answer(frames, [r['timestamp_seconds'] for r in pool['selected_frames']],
                                        pool['video']['duration_seconds'], row['question'], row['options'],
                                        generation_state=answer_state)
                end = time.perf_counter()
                # 步骤6：核对分阶段耗时；D同时保留新增耗时及含C前序的诊断归因值。
                times['selection_total_seconds'] = selection_end - start
                times['qa_total_seconds'] = end - qa_start
                times['generation_seconds'] = answer['generation_seconds']
                times['llava_preprocess_seconds'] = answer['preprocessing_seconds']
                times['end_to_end_seconds'] = end - start
                times['stage_sum_seconds'] = (times['selection_core_seconds'] + times.get('candidate_decode_seconds', 0)
                                              + times.get('fine_decode_seconds', 0) + times['selected_decode_seconds'] + times['qa_total_seconds'])
                times['unattributed_seconds'] = times['end_to_end_seconds'] - times['stage_sum_seconds']
                if variant == 'D':
                    times['donor_pool_acquisition_seconds'] = donor['pool']['pool_acquisition_seconds']
                    times['diagnostic_attributed_e2e_seconds'] = times['donor_pool_acquisition_seconds'] + times['end_to_end_seconds']
                measured = dict(status='completed', variant=variant, question_id=qid, video_id=row['video_id'],
                                stratum=row['stratum'], question=row['question'], options=row['options'],
                                source_video_sha256=job['video_sha256'], protocol_sha256=fingerprint,
                                order_index=job['order'], execution_order=list(order_for(job['order'])), physical_gpu=gpu,
                                pool=pool, answer=answer, timings=times,
                                reference_answer='ABCD'[row['answer_index']], correct=answer['parsed_answer'] == 'ABCD'[row['answer_index']],
                                timing_kind='diagnostic_incremental' if variant == 'D' else 'direct_cold_application',
                                application_cache_hits=0 if variant != 'D' else 1,
                                scope_state=scope_state, answer_state=answer_state, attempt_number=number,
                                started_at_utc=attempt['started_at_utc'], completed_at_utc=utc(),
                                memory=dict(resident_allocated_gib=resident / 2**30, selector_peak_allocated_gib=selector_peak / 2**30,
                                            selector_incremental_peak_gib=max(0, selector_peak - resident) / 2**30,
                                            qa_incremental_peak_gib=max(0, answer['peak_allocated_gib'] - before_qa / 2**30),
                                            worker_peak_allocated_gib=max(selector_peak / 2**30, answer['peak_allocated_gib'])))
                # 步骤7：计时结束后保存特征、结果和尝试记录；标准答案仅用于此时的计分。
                save_start = time.perf_counter()
                if variant != 'D':
                    feature_path = run / 'features' / f'{qid}_{variant}_{number:02}.npy'
                    with feature_path.open('xb') as f:
                        np.save(f, features, allow_pickle=False)
                    measured.update(feature_file=str(feature_path.relative_to(run)), feature_sha256=sha256(feature_path))
                else:
                    measured.update(feature_file=donor['feature_file'], feature_sha256=donor['feature_sha256'],
                                    donor_result_sha256=donor_sha, expected_donor_result_sha256=sha256(donor_path))
                validate_result(measured, row, fingerprint)
                write_json(destination, measured)
                attempt.update(status='completed', scope_state=scope_state, answer_state=answer_state,
                               artifact_write_seconds=time.perf_counter() - save_start,
                               result_file=str(destination.relative_to(run)), attempt_wall_seconds=time.perf_counter() - start)
                write_json(journal, attempt)
                events.put(dict(type='completed', gpu=gpu, question_id=qid, variant=variant,
                                e2e_seconds=times['end_to_end_seconds']))
                del measured, pool, features, frames, answer, donor
                measured = None
            except BaseException as e:
                uncertain = bool(answer_state.get('returned'))
                blocking = isinstance(e, (ValueError, torch.cuda.OutOfMemoryError))
                # 步骤8：区分明确失败与已返回生成，保留分类结果，禁止无记录地再次分类。
                retryable = not uncertain and not scope_state.get('returned') and not blocking and isinstance(e, (OSError, RuntimeError))
                attempt.update(status='uncertain' if uncertain else 'failed', retryable=retryable,
                               scope_state=scope_state, answer_state=answer_state, error=f'{type(e).__name__}: {e}',
                               traceback=traceback.format_exc())
                if measured is not None:
                    attempt['recoverable_result'] = measured
                write_json(journal, attempt)
                events.put(dict(type='job_failed', gpu=gpu, question_id=qid, variant=variant,
                                retryable=retryable, error=attempt['error']))
                break
        else:
            events.put(dict(type='job_completed', gpu=gpu, question_id=qid))
