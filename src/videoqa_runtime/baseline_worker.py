import contextlib
import datetime
import os
import time
import traceback
from pathlib import Path

from .common import ROOT, offline_environment, read_json, write_json
from .baseline_records import METHODS, result_path, validate_result
from .baseline_selection import BlipItmScorer, ProtocolError, prepare_selection


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def worker_main(gpu, task_queue, events, stop, run_dir, protocol_hash):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    offline_environment()
    run_dir = Path(run_dir)
    with (run_dir / 'logs' / f'gpu_{gpu}.log').open('a', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                _worker_loop(gpu, task_queue, events, stop, run_dir, protocol_hash)
            except BaseException as error:
                traceback.print_exc()
                events.put(dict(type='worker_failed', gpu=gpu, error=f'{type(error).__name__}: {error}'))


def _worker_loop(gpu, task_queue, events, stop, run_dir, protocol_hash):
    import torch
    from .llava_backend import LlavaBackend
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    llava = LlavaBackend()
    scorer = BlipItmScorer(read_json(ROOT / 'configs/local_model_inventory.json')['models']['blip']['path'])
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    llava.warmup()
    scorer.warmup()
    warmup_seconds = time.perf_counter() - started
    startup = dict(gpu=gpu, pid=os.getpid(), model_load_seconds=load_seconds,
                   warmup_seconds=warmup_seconds, synthetic_llava_forwards=1, synthetic_blip_forwards=2,
                   resident_allocated_gib=torch.cuda.memory_allocated() / 2**30,
                   resident_reserved_gib=torch.cuda.memory_reserved() / 2**30, llava=llava.load_report,
                   cpu_threads=2, decode_threads=2, utc=utc())
    write_json(run_dir / 'workers' / f'gpu_{gpu}.json', startup)
    events.put(dict(type='ready', gpu=gpu, startup_seconds=load_seconds + warmup_seconds))
    while not stop.is_set():
        job = task_queue.get()
        if job is None:
            break
        row, order = job['row'], job['order']
        qid = row['question_id']
        methods = METHODS if order % 2 == 0 else tuple(reversed(METHODS))
        for method in methods:
            if stop.is_set():
                break
            destination = result_path(run_dir, method, qid)
            if destination.exists():
                validate_result(read_json(destination), row, protocol_hash)
                continue
            attempt_dir = run_dir / 'attempts' / qid
            attempt_dir.mkdir(parents=True, exist_ok=True)
            attempt_number = len(list(attempt_dir.glob(f'{method}.*.json'))) + 1
            journal = attempt_dir / f'{method}.{attempt_number:02}.json'
            attempt = dict(question_id=qid, method=method, physical_gpu=gpu, attempt=attempt_number,
                           status='running_may_generate', started_at_utc=utc(), generation_state={},
                           note='Written before timing; abrupt interruption is conservatively uncertain.')
            with journal.open('x') as reserved:
                reserved.write('{}\n')
            write_json(journal, attempt)
            events.put(dict(type='method_started', gpu=gpu, question_id=qid, method=method))
            generation_state = {}
            measured_result = None
            method_start = time.perf_counter()
            try:
                torch.cuda.synchronize()
                resident = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                frames, selection = prepare_selection(
                    ROOT / 'data' / row['video_relative_path'], row['question'], method, scorer,
                    progress=lambda n: events.put(dict(type='progress', gpu=gpu, question_id=qid, method=method, candidates=n)))
                torch.cuda.synchronize()
                selection_end = time.perf_counter()
                selector_peak = torch.cuda.max_memory_allocated()
                before_qa_allocated = torch.cuda.memory_allocated()
                qa_start = time.perf_counter()
                answer = llava.answer(frames, [r['timestamp_seconds'] for r in selection['selected_frames']],
                                      selection['video']['duration_seconds'], row['question'], row['options'],
                                      generation_state=generation_state)
                end = time.perf_counter()
                times = dict(selection['timings'])
                times['selection_total_seconds'] = selection_end - start
                times['qa_total_seconds'] = end - qa_start
                times['llava_preprocess_seconds'] = answer['preprocessing_seconds']
                times['prompt_tokenize_seconds'] = answer['prompt_tokenize_seconds']
                times['generation_seconds'] = answer['generation_seconds']
                times['answer_parse_seconds'] = answer['answer_parse_seconds']
                times['end_to_end_seconds'] = end - start
                # Disjoint leaves, with the entire externally timed answer call as one leaf.
                times['stage_sum_seconds'] = (times['candidate_decode_seconds'] + times['selection_core_seconds']
                                              + times['selected_decode_seconds'] + times['qa_total_seconds'])
                times['unattributed_seconds'] = times['end_to_end_seconds'] - times['stage_sum_seconds']
                reference = 'ABCD'[row['answer_index']]
                measured_result = dict(
                    status='completed', question_id=qid, video_id=row['video_id'], stratum=row['stratum'],
                    question=row['question'], options=row['options'], method=method, order_index=order,
                    method_order=list(methods), physical_gpu=gpu, worker_pid=os.getpid(),
                    protocol_sha256=protocol_hash, started_at_utc=attempt['started_at_utc'], completed_at_utc=utc(),
                    source_video_sha256=job['video_sha256'],
                    generation_attempts=1, generation_state=generation_state, application_cache_hits=0,
                    selection=selection, answer=answer, reference_answer=reference,
                    correct=answer['parsed_answer'] == reference, timings=times,
                    memory=dict(resident_allocated_gib=resident / 2**30,
                                selector_peak_allocated_gib=selector_peak / 2**30,
                                selector_incremental_peak_gib=max(0, selector_peak - resident) / 2**30,
                                qa_incremental_peak_gib=max(0, answer['peak_allocated_gib'] - before_qa_allocated / 2**30),
                                worker_method_peak_allocated_gib=max(selector_peak / 2**30, answer['peak_allocated_gib'])),
                    attempt_number=attempt_number, resumed_execution=job['resumed'])
                validate_result(measured_result, row, protocol_hash)
                save_start = time.perf_counter()
                write_json(destination, measured_result)
                attempt.update(status='completed', generation_state=generation_state, result_path=str(destination.relative_to(run_dir)),
                               result_write_seconds=time.perf_counter() - save_start,
                               attempt_wall_seconds=time.perf_counter() - method_start)
                write_json(journal, attempt)
                del frames, selection, answer, measured_result
                measured_result = None
                events.put(dict(type='method_completed', gpu=gpu, question_id=qid, method=method,
                                e2e_seconds=times['end_to_end_seconds']))
            except BaseException as error:
                # A returned generation is never automatically repeated if saving/validation failed.
                uncertain = bool(generation_state.get('returned'))
                blocking = isinstance(error, (ProtocolError, ValueError, torch.cuda.OutOfMemoryError))
                retryable = not uncertain and not blocking and isinstance(error, (OSError, RuntimeError))
                attempt.update(status='uncertain' if uncertain else 'failed', generation_state=generation_state,
                               retryable=retryable, error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(),
                               attempt_wall_seconds=time.perf_counter() - method_start)
                if measured_result is not None:
                    attempt['recoverable_result'] = measured_result
                write_json(journal, attempt)
                events.put(dict(type='job_failed', gpu=gpu, question_id=qid, method=method,
                                retryable=retryable, error=attempt['error']))
                break
        else:
            events.put(dict(type='job_completed', gpu=gpu, question_id=qid))
