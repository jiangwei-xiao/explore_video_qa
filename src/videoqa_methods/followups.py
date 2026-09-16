"""首轮改进的持久化调用账本、长期GPU进程和分阶段调度。"""
import contextlib
import datetime
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import shutil
import time
import traceback
import numpy as np

from videoqa_runtime.common import ROOT, read_json, write_json, sha256, offline_environment
from videoqa_runtime.baseline_run import preflight, available_gpus
from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_runtime.protocol import question_text
from .algorithm import segment_scores, compete, membership
from .improvements import reselect_local, select_clocal, rewrite_query, compare_upstream, QUERY_PROMPT, WHITELIST
from .model_ops import ScopeClassifier, FeatureScorer
from .pipeline import scan
from .run import verify_reference_and_baseline

HISTORY = ROOT / 'outputs/methods/method_v1_videomme50_20260914_r1'
AUDIT = ROOT / 'outputs/analysis/evidence_videomme50_20260915_r1'


def utc():
    """返回UTC审计时间；耗时另用单调时钟。"""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def durable(path, value):
    """接口：原子替换JSON并同步文件和父目录，确保生成开始记录先于实际调用持久化。"""
    write_json(path, value)
    with Path(path).open('rb') as handle: os.fsync(handle.fileno())
    fd = os.open(str(Path(path).parent), os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


class CallLedger:
    """接口：每题每类型只允许一个调用ID；已完成返回保存结果，未确认状态禁止再生成。"""
    def __init__(self, run, qid, fingerprint, gpu):
        """初始化单题账本、运行身份和本次检查点计时。"""
        self.run, self.qid, self.fingerprint, self.gpu = Path(run), qid, fingerprint, gpu
        self.checkpoint_seconds = 0.0
        self.reused = []

    def invoke(self, kind, function):
        """步骤1检查已保存状态，步骤2持久化调用开始，步骤3保存完整返回；异常一律暂停。"""
        if kind not in ('scope', 'answer', 'query'): raise ProtocolError('Unknown call type')
        path = self.run / 'calls' / kind / f'{self.qid}.json'
        if path.exists():
            saved = read_json(path)
            if saved['protocol_sha256'] != self.fingerprint or saved['status'] != 'completed':
                raise ProtocolError(f'Uncertain/failed call: {path}; automatic repetition prohibited')
            self.reused.append(kind)
            return saved['result']
        record = dict(call_id=f'{kind}:{self.qid}', kind=kind, question_id=self.qid, status='started_may_generate',
                      protocol_sha256=self.fingerprint, gpu=self.gpu, started_at_utc=utc(), pid=os.getpid())
        t = time.perf_counter()
        durable(path, record)
        self.checkpoint_seconds += time.perf_counter() - t
        state = {}
        try:
            result = function(state)
            record.update(status='completed', result=result, generation_state=state, completed_at_utc=utc())
        except BaseException as exc:
            record.update(status='failed_or_uncertain', generation_state=state, error=repr(exc), traceback=traceback.format_exc())
            durable(path, record)
            raise
        t = time.perf_counter()
        durable(path, record)
        self.checkpoint_seconds += time.perf_counter() - t
        return result


def historical(qid, variant='C'):
    """接口：只读加载历史方法池和哈希核验后的特征；新E扫描不以此作为应用缓存。"""
    record = read_json(HISTORY / 'results' / variant / f'{qid}.json')
    path = HISTORY / record['feature_file']
    if sha256(path) != record['feature_sha256']: raise ProtocolError('Historical feature changed')
    return record, np.load(path, allow_pickle=False)


class CountedScorer(FeatureScorer):
    """记录实际BLIP前向批次和帧数，合成预热与正式任务分别取差值。"""
    def __init__(self, path):
        """加载冻结评分器并初始化计数，不改变原批次精度与大小。"""
        super().__init__(path)
        self.calls = 0
        self.frames = 0

    def score_batch_features(self, encoded, images):
        """调用开始即计入预算成本；异常批次也不会从计数中消失。"""
        self.calls += 1
        self.frames += len(images)
        return super().score_batch_features(encoded, images)


def validate_e(record, row, fingerprint):
    """接口：验证E身份、原提示词、配额、16帧/3360Token和计分；不以准确率决定验收。"""
    from .records import validate_result
    # 步骤1：复用旧C的通用输入/配额验证，但不改写保存记录或旧验证器。
    proxy = dict(record, variant='C', application_cache_hits=0)
    validate_result(proxy, row, fingerprint)
    pool = record['pool']
    if record['variant'] != 'E': raise ProtocolError('Expected E')
    if not record['upstream_validation']['passed']: raise ProtocolError('Upstream mismatch; attribution paused')
    message = question_text(row['question'], row['options'], pool['video']['duration_seconds'],
                            [r['timestamp_seconds'] for r in pool['selected_frames']])
    if message not in record['answer']['prompt']: raise ProtocolError('QA message differs')
    features = np.load(Path(record['run_directory']) / record['feature_file'], allow_pickle=False)
    if sha256(Path(record['run_directory']) / record['feature_file']) != record['feature_sha256']:
        raise ProtocolError('E feature changed')
    selected, trace = reselect_local(pool, features, read_json(ROOT / 'configs/method_v1.json'))
    if selected != pool['selected_indices'] or trace != pool['local_reselection_trace']:
        raise ProtocolError('C-local replay mismatch')


def save_features(run, name, features):
    """接口：保存本轮特征，临时文件替换确保恢复不会读取半个npy文件。"""
    path = run / 'features' / name
    tmp = path.with_suffix('.npy.tmp')
    with tmp.open('wb') as handle:
        np.save(handle, features, allow_pickle=False)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)
    return dict(feature_file=str(path.relative_to(run)), feature_sha256=sha256(path))


def execute_e(run, job, backend, scorer, gpu, fingerprint, progress):
    """核心接口：分类到答案解析直接计时，生成前持久化输入；恢复复用已完成调用并单列成本。"""
    import torch
    row, qid = job['row'], job['row']['question_id']
    destination = run / 'results/E' / f'{qid}.json'
    if destination.exists():
        validate_e(read_json(destination), row, fingerprint)
        return
    old, oldfeatures = historical(qid)
    ledger = CallLedger(run, qid, fingerprint, gpu)
    checkpoint = run / 'selection_checkpoints' / f'{qid}.json'
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    # 步骤1：仅恢复可用已完成选择快照；正常任务独立扫描原片。
    if checkpoint.exists():
        selected_state = read_json(checkpoint)
        if selected_state['protocol_sha256'] != fingerprint: raise ProtocolError('Selection checkpoint fingerprint changed')
        pool = selected_state['pool']
        # 恢复阶段只统计本次实际工作；原扫描成本保留在检查点，不冒充本次直接E2E。
        times = {k: 0.0 for k in selected_state['timings']}
        feature_info = selected_state['feature_info']
        if sha256(run / feature_info['feature_file']) != feature_info['feature_sha256']: raise ProtocolError('Checkpoint feature changed')
        features = np.load(run / feature_info['feature_file'], allow_pickle=False)
        from videoqa_runtime.video import decode_selected
        decode_start = time.perf_counter()
        frames = decode_selected(pool['video'], pool['selected_frames'])
        times['selected_decode_seconds'] = time.perf_counter() - decode_start
        resumed = True
    else:
        scope = ledger.invoke('scope', lambda state: ScopeClassifier(backend).classify(row['question'], state))
        frames, pool, features, times = select_clocal(ROOT / 'data' / row['video_relative_path'], row['question'],
                                                     scorer, scope, read_json(run / 'method_config.json'), progress)
        times['scope_seconds'] = scope['seconds']
        times['selection_core_seconds'] += scope['seconds']
        # 步骤2：问答之前持久化精确选择及特征；此必要检查点开销计入直接E2E。
        t = time.perf_counter()
        feature_info = save_features(run, f'{qid}_E.npy', features)
        durable(checkpoint, dict(protocol_sha256=fingerprint, pool=pool, timings=times, feature_info=feature_info))
        ledger.checkpoint_seconds += time.perf_counter() - t
        resumed = bool(ledger.reused)
    torch.cuda.synchronize()
    selection_end = time.perf_counter()
    selector_peak = torch.cuda.max_memory_allocated()
    before_qa = torch.cuda.memory_allocated()
    # 步骤3：调用账本在生成前落盘；答案返回的即时墙钟由闭包记录，排除返回结果落盘。
    end_holder = []
    qa_start = time.perf_counter()
    qa_checkpoint_before = ledger.checkpoint_seconds
    def answer_once(state):
        """调用原问答入口，不输入重选理由或审计答案；立即记录解析完成时刻。"""
        result = backend.answer(frames, [r['timestamp_seconds'] for r in pool['selected_frames']], pool['video']['duration_seconds'],
                                row['question'], row['options'], generation_state=state)
        end_holder.append(time.perf_counter())
        return result
    answer = ledger.invoke('answer', answer_once)
    end = end_holder[0] if end_holder else time.perf_counter()
    times['selection_total_seconds'] = selection_end - start
    times['qa_total_seconds'] = end - qa_start
    times['end_to_end_seconds'] = end - start
    times['generation_seconds'] = answer['generation_seconds']
    times['llava_preprocess_seconds'] = answer['preprocessing_seconds']
    # 返回记录写盘在E2E外，按直接测得的阶段余量报告必要检查点。
    times['stage_sum_seconds'] = times['selection_core_seconds'] + sum(times.get(k, 0) for k in
        ('candidate_decode_seconds', 'fine_decode_seconds', 'selected_decode_seconds')) + times['qa_total_seconds']
    # 分类检查点与选择快照写入位于选择区间，但分类原函数/扫描子阶段不计其成本。
    times['stage_sum_seconds'] += qa_checkpoint_before
    times['unattributed_seconds'] = times['end_to_end_seconds'] - times['stage_sum_seconds']
    times['runtime_checkpoint_seconds_total'] = ledger.checkpoint_seconds
    times['selection_checkpoint_seconds'] = qa_checkpoint_before
    # 步骤4：保留实际生成后核对历史上游；不一致仍保存结果，随后暂停归因。
    upstream = compare_upstream(pool, features, old['pool'], oldfeatures)
    reference = 'ABCD'[row['answer_index']]
    record = dict(status='completed', variant='E', question_id=qid, video_id=row['video_id'], question=row['question'],
                  options=row['options'], stratum=row['stratum'], protocol_sha256=fingerprint, run_directory=str(run),
                  physical_gpu=gpu, order_index=job['order'], pool=pool, answer=answer, timings=times,
                  correct=answer['parsed_answer'] == reference, reference_answer=reference, source_video_sha256=job['video_sha256'],
                  **feature_info, upstream_validation=upstream, application_cache_hits=int(resumed),
                  timing_kind='resumed_partial_not_primary' if resumed or ledger.reused else 'direct_no_application_cache',
                  calls_reused=ledger.reused, completed_at_utc=utc(),
                  memory=dict(resident_allocated_gib=resident / 2**30, selector_peak_allocated_gib=selector_peak / 2**30,
                              selector_incremental_peak_gib=max(0, selector_peak-resident) / 2**30,
                              qa_incremental_peak_gib=max(0, answer['peak_allocated_gib'] - before_qa / 2**30),
                              worker_peak_allocated_gib=max(selector_peak / 2**30, answer['peak_allocated_gib'])))
    t = time.perf_counter()
    durable(destination, record)
    durable(run / 'writes' / f'{qid}_E.json', dict(final_result_write_seconds=time.perf_counter()-t))
    validate_e(record, row, fingerprint)


def execute_query(run, job, backend, gpu, fingerprint):
    """接口：每题生成一次查询并持久化，不评分、不查询选项或标准答案。"""
    row = job['row']
    result = CallLedger(run, row['question_id'], fingerprint, gpu).invoke(
        'query', lambda state: rewrite_query(row['question'], backend, state))
    durable(run / 'queries' / f'{row["question_id"]}.json', dict(question_id=row['question_id'], question=row['question'],
            protocol_sha256=fingerprint, physical_gpu=gpu, **result))


def execute_score(run, job, scorer, fingerprint, progress):
    """接口：冻结查询后重新完整扫描评分；仅BLIP编码完全相同允许复用历史B，并明确成本来源。"""
    row, qid = job['row'], job['row']['question_id']
    destination = run / 'results/Q' / f'{qid}.json'
    if destination.exists(): return
    frozen = read_json(run / 'query_freeze.json')
    query_path = run / 'queries' / f'{qid}.json'
    if sha256(query_path) != frozen['sha256'][qid]: raise ProtocolError('Frozen query changed')
    query_result = read_json(query_path)
    old, oldfeatures = historical(qid, 'B')
    t = time.perf_counter()
    original = scorer.tokenize(row['question'])['input_ids'].tolist()
    rewritten = scorer.tokenize(query_result['query'])['input_ids'].tolist()
    token_check = time.perf_counter() - t
    cache = original == rewritten
    start = time.perf_counter()
    # 步骤1：变化查询重新读取原片完整候选；未变化查询记录历史缓存身份。
    if cache:
        pool = {k: old['pool'][k] for k in ('candidates', 'scores', 'video', 'blip_question_tokens')}
        features = oldfeatures
        times = dict(candidate_decode_seconds=0., rgb_seconds=0., blip_preprocess_seconds=0., blip_forward_features_seconds=0.)
    else:
        pool, features, times, _ = scan(ROOT / 'data' / row['video_relative_path'], query_result['query'], scorer, progress)
    if pool['candidates'] != old['pool']['candidates']: raise ProtocolError('Query initial candidates changed')
    # 步骤2：只复用历史B的λ，新分数重新分段、竞争；无补查、无范围分类或最终QA。
    cfg = read_json(run / 'method_config.json')
    t = time.perf_counter()
    segments, wavelet = segment_scores(pool['scores'], pool['candidates'], pool['video'], cfg)
    times['segmentation_seconds'] = time.perf_counter()-t
    t = time.perf_counter()
    selected, quotas, trace = compete(pool['candidates'], pool['scores'], features, segments, old['pool']['lambda_value'], cfg)
    times['competition_seconds'] = time.perf_counter()-t
    times.update(offline_selection_seconds=time.perf_counter()-start, token_comparison_seconds=token_check,
                 query_generation_seconds=query_result['seconds'])
    pool.update(initial_count=len(pool['candidates']), lambda_value=old['pool']['lambda_value'], segments=segments, wavelet=wavelet,
                quotas=quotas, competition_trace=trace, selected_indices=selected, selected_frames=[pool['candidates'][i] for i in selected])
    info = save_features(run, f'{qid}_Q.npy', features)
    durable(destination, dict(status='completed', variant='Q', question_id=qid, query=query_result['query'],
            protocol_sha256=fingerprint, pool=pool, timings=times, **info, application_cache_hits=int(cache),
            cache_source=str(HISTORY / 'results/B' / f'{qid}.json') if cache else None,
            historical_b_sha256=sha256(HISTORY / 'results/B' / f'{qid}.json'),
            query_sha256=sha256(query_path), original_blip_token_ids=original, rewritten_blip_token_ids=rewritten,
            timing_kind='offline_selection_not_qa_e2e'))


def worker(gpu, tasks, events, stop, run, fingerprint):
    """接口：每卡长期驻留模型，按照阶段领取任务；任何异常保留日志并暂停，不重试生成。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    offline_environment()
    run = Path(run)
    with (run / 'logs' / f'gpu_{gpu}.log').open('a', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            import torch
            from videoqa_runtime.llava_backend import LlavaBackend
            torch.set_num_threads(2); torch.set_num_interop_threads(1)
            t = time.perf_counter()
            backend = LlavaBackend()
            scorer = CountedScorer(read_json(ROOT / 'configs/local_model_inventory.json')['models']['blip']['path'])
            load = time.perf_counter()-t
            t = time.perf_counter(); backend.warmup(); scorer.warmup()
            durable(run / 'workers' / f'gpu_{gpu}_{os.getpid()}.json', dict(gpu=gpu, pid=os.getpid(), load_seconds=load,
                    warmup_seconds=time.perf_counter()-t, llava_synthetic_forwards=1, blip_synthetic_forwards=2,
                    backend=backend.load_report, resident_allocated_gib=torch.cuda.memory_allocated()/2**30))
            events.put(dict(type='ready', gpu=gpu))
            while not stop.is_set():
                job = tasks.get()
                if job is None: return
                phase, qid = job['phase'], job['row']['question_id']
                events.put(dict(type='started', gpu=gpu, phase=phase, question_id=qid))
                progress = lambda n: events.put(dict(type='progress', gpu=gpu, phase=phase, question_id=qid, candidates=n))
                before_calls, before_frames = scorer.calls, scorer.frames
                if phase == 'E': execute_e(run, job, backend, scorer, gpu, fingerprint, progress)
                elif phase == 'query': execute_query(run, job, backend, gpu, fingerprint)
                elif phase == 'Q': execute_score(run, job, scorer, fingerprint, progress)
                else: raise ProtocolError('Unknown phase')
                durable(run / 'blip_calls' / f'{phase}_{qid}.json',dict(phase=phase,question_id=qid,
                        forward_calls=scorer.calls-before_calls,scored_frames=scorer.frames-before_frames))
                events.put(dict(type='completed', gpu=gpu, phase=phase, question_id=qid))
        except BaseException as exc:
            traceback.print_exc()
            durable(run / 'failures' / f'gpu_{gpu}_{time.time_ns()}.json', dict(error=repr(exc), traceback=traceback.format_exc(), utc=utc()))
            stop.set()
            events.put(dict(type='failed', gpu=gpu, error=repr(exc)))


def freeze_proxy(rows):
    """接口：在模型调用前冻结42题75事实及独立哨兵候选，不用新输出扩充评价分母。"""
    facts = []
    for row in rows:
        qid = row['question_id']
        if qid in ('491-2', '321-2'): continue
        old = read_json(HISTORY / 'results/B' / f'{qid}.json')['pool']
        pts = {r['source_pts'] for r in old['candidates']}
        for fact in read_json(AUDIT / 'full_review/witnesses_r6' / f'{qid}.json')['facts']:
            witnesses = sorted({w['source_pts'] for w in fact['witnesses']} & pts)
            if fact['role'] in ('required', 'partial') and witnesses:
                facts.append(dict(question_id=qid, fact_id=fact['id'], role=fact['role'], description=fact['description'], witness_pts=witnesses))
    if len(facts) != 75 or len({f['question_id'] for f in facts}) != 42: raise ProtocolError('Proxy denominator changed')
    sentinels = []
    for qid, indices in {'103-1':[5], '306-2':[88,89], '409-3':[77,80]}.items():
        pool = read_json(HISTORY / 'results/B' / f'{qid}.json')['pool']
        for i in indices: sentinels.append(dict(question_id=qid, candidate_index=i, source_pts=pool['candidates'][i]['source_pts']))
    return dict(questions=42, facts=75, entries=facts, sentinels=sentinels, ambiguity_excluded=['491-2', '321-2'])


def protocol():
    """接口：冻结新旧执行代码、依赖、全部历史结果与代理事实身份，不改写历史快照。"""
    files = sorted((ROOT / 'src/videoqa_runtime').glob('*.py'))
    files += [ROOT / 'src/videoqa_methods' / name for name in
              ('algorithm.py','refinement.py','model_ops.py','pipeline.py','records.py','run.py','worker.py','improvements.py','followups.py')]
    files += [ROOT / 'scripts/run_method_followups.py', ROOT / 'configs/method_v1.json',
              ROOT / 'requirements-runtime.lock.txt', ROOT / 'requirements-method.lock.txt',
              ROOT / 'configs/local_model_inventory.json', ROOT / 'configs/llava_source_manifest.json']
    history_files = sorted((HISTORY / 'results').glob('*/*.json')) + sorted((HISTORY / 'baseline_reference/results').glob('*/*.json'))
    history_files += sorted((HISTORY / 'features').glob('*.npy')) + sorted((AUDIT / 'full_review/witnesses_r6').glob('*.json'))
    history_files += [AUDIT / 'final_review/acceptance.json', HISTORY / 'protocol.json']
    return dict(version='followups-clocal-query-v1', config=read_json(ROOT / 'configs/method_v1.json'),
                query_prompt=QUERY_PROMPT, query_whitelist=sorted(WHITELIST), max_calls=dict(answer=50, scope=50, query=50),
                code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in files},
                historical_sha256={str(p.relative_to(ROOT)):sha256(p) for p in history_files},
                query_max_new_tokens=128, final_qa_max_new_tokens=8, scope_max_new_tokens=8,
                primary_proxy_questions=42, primary_proxy_facts=75, query_no_final_qa=True,
                runtime_checkpoints_in_direct_e2e=True, no_automatic_generation_retry=True)


def replay(run, rows, cfg):
    """接口：无模型重放50个冻结C池，保存固定帧和替换追踪；正式E不读取重放作为选帧缓存。"""
    for row in rows:
        qid = row['question_id']
        record, features = historical(qid)
        selected, trace = reselect_local(record['pool'], features, cfg)
        durable(run / 'replay' / f'{qid}.json', dict(question_id=qid, selected_indices=selected, trace=trace,
                historical_selected=record['pool']['selected_indices'], historical_sha256=sha256(HISTORY/'results/C'/f'{qid}.json')))


def execute(run_id, gpus, phase='all', resume=False, replay_only=False):
    """入口：取得排他锁，按E试跑/全量→查询冻结→离线评分执行，支持安全恢复与纯CPU重放。"""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id): raise ValueError('Invalid run id')
    run = ROOT / 'outputs/methods' / run_id
    run.mkdir(parents=True, exist_ok=resume)
    with (run / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute_locked(run, gpus, phase, resume, replay_only)
    return run


def execute_locked(run, requested, phase, resume, replay_only):
    """核心调度：冻结身份、核查调用预算，模型常驻跨阶段；停止只阻止后续任务，不杀正在生成的进程。"""
    start = time.perf_counter()
    for d in ('logs','workers','features','calls/scope','calls/query','calls/answer','queries','results/E','results/Q',
              'selection_checkpoints','failures','writes','replay','code_snapshot','blip_calls'): (run/d).mkdir(parents=True, exist_ok=True)
    rows = read_json(ROOT / 'data/manifests/video_mme_development.json')['rows']
    spec = protocol()
    if resume:
        if read_json(run/'protocol.json') != spec: raise ProtocolError('Run fingerprint changed; resume refused')
    else:
        durable(run/'protocol.json', spec); durable(run/'method_config.json', spec['config'])
        durable(run/'proxy_manifest.json', freeze_proxy(rows))
        for name in spec['code_sha256']:
            target = run/'code_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, target)
    fingerprint = sha256(run/'protocol.json')
    for kind in ('scope','answer','query'):
        paths = list((run/'calls'/kind).glob('*.json'))
        if len(paths)>50: raise ProtocolError('Call budget exceeded')
        for p in paths:
            c = read_json(p)
            if c['question_id'] not in {r['question_id'] for r in rows} or c['protocol_sha256'] != fingerprint or c['status'] != 'completed':
                raise ProtocolError(f'Incomplete or invalid call requires manual inspection: {p}')
    replay(run, rows, spec['config'])
    if replay_only:
        durable(run/'status.json', dict(status='cpu_replay_completed', replay_questions=50, generation_calls=0)); return
    # 步骤1补充：完整恢复只做核验，不再加载模型或生成。Q同时校验冻结查询和特征身份。
    e_done = len(list((run/'results/E').glob('*.json'))) == 50
    q_done = len(list((run/'results/Q').glob('*.json'))) == 50
    if e_done:
        for row in rows: validate_e(read_json(run/'results/E'/f'{row["question_id"]}.json'),row,fingerprint)
    if q_done:
        frozen=read_json(run/'query_freeze.json')['sha256']
        for row in rows:
            qid=row['question_id']; rec=read_json(run/'results/Q'/f'{qid}.json')
            if rec['protocol_sha256']!=fingerprint or sha256(run/rec['feature_file'])!=rec['feature_sha256']:
                raise ProtocolError('Q result fingerprint or features changed')
            if sha256(run/'queries'/f'{qid}.json')!=frozen[qid] or rec['query_sha256']!=frozen[qid]:
                raise ProtocolError('Q frozen query changed')
    if (phase=='all' and e_done and q_done) or (phase=='clocal' and e_done) or (phase=='query' and q_done):
        print('Requested phases complete; no model load or generation repeated.',flush=True); return
    verify_reference_and_baseline()
    durable(run/'status.json', dict(status='preflight', pid=os.getpid(), utc=utc()))
    evidence = preflight(rows); durable(run/'preflight.json', evidence)
    selected_gpus, states = available_gpus(requested)
    durable(run/'gpu_allocation.json', dict(requested=requested, selected=selected_gpus, before_launch=states))
    durable(run/'pid.json', dict(pid=os.getpid(), session_id=os.getsid(0), utc=utc()))
    # 步骤2：每卡长期进程跨阶段复用；前5题门槛不检查正确率。
    ctx = mp.get_context('spawn'); events = ctx.Queue(); stop = ctx.Event()
    queues = {g:ctx.Queue() for g in selected_gpus}
    processes = {g:ctx.Process(target=worker, args=(g,queues[g],events,stop,str(run),fingerprint)) for g in selected_gpus}
    idle, active = set(), {}
    current, last = 'startup', 0.
    durations = {x['video_id']:x['duration_seconds'] for x in read_json(ROOT/'data/manifests/video_mme_asset_validation.json')['videos']}
    jobs = [dict(row=r, order=i, video_sha256=evidence['video_hashes'][r['video_id']]) for i,r in enumerate(rows)]
    log = (run/'events.jsonl').open('a', buffering=1)
    def publish(force=False):
        """定期发布可观察状态；SSH观察断开不影响调度。"""
        nonlocal last
        if not force and time.perf_counter()-last<10: return
        last=time.perf_counter()
        counts={kind:len(list((run/'calls'/kind).glob('*.json'))) for kind in ('scope','answer','query')}
        result_counts={kind:len(list((run/'results'/kind).glob('*.json'))) for kind in ('E','Q')}
        durable(run/'status.json', dict(status=current, calls=counts, results=result_counts, active=active,
                elapsed_seconds=last-start, pid=os.getpid(), utc=utc()))
        print(f'{current}: {result_counts}, calls={counts}, active={active}, elapsed={last-start:.1f}s', flush=True)
    def event():
        """等待队列同时检查工作进程存活；超时仅更新观察，不重启任务。"""
        while True:
            publish()
            try: item=events.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode is not None for p in processes.values()): raise RuntimeError('Worker exited; inspect durable calls')
                continue
            log.write(json.dumps(dict(item, observed_at=utc()),ensure_ascii=False)+'\n')
            if item['type']=='failed': raise RuntimeError(str(item))
            return item
    def run_phase(work, kind):
        """共享队列按空闲GPU分发；已完成文件跳过，不在断点恢复时重复调用。"""
        pending=[]
        for job in work:
            qid=job['row']['question_id']
            path=run/'queries'/f'{qid}.json' if kind=='query' else run/'results'/kind/f'{qid}.json'
            if path.exists():
                if kind=='E': validate_e(read_json(path),job['row'],fingerprint)
                continue
            pending.append(dict(job,phase=kind))
        while pending or active:
            for gpu in sorted(idle):
                if not pending: break
                job=pending.pop(0); active[gpu]=job['row']['question_id']; idle.remove(gpu); queues[gpu].put(job)
            item=event()
            if item['type']=='completed': active.pop(item['gpu']); idle.add(item['gpu'])
    error=None
    try:
        for p in processes.values(): p.start()
        while len(idle)<len(processes):
            item=event()
            if item['type']=='ready': idle.add(item['gpu'])
        if phase in ('all','clocal'):
            current='E_pilot'; publish(True); pilot=time.perf_counter()
            run_phase(jobs[:5],'E')
            for job in jobs[:5]: validate_e(read_json(run/'results/E'/f'{job["row"]["question_id"]}.json'),job['row'],fingerprint)
            durable(run/'pilot_acceptance.json',dict(status='passed',questions=[j['row']['question_id'] for j in jobs[:5]],
                    wall_seconds=time.perf_counter()-pilot, criterion='protocol and upstream, not accuracy'))
            current='E_full'; publish(True)
            run_phase(sorted(jobs[5:],key=lambda j:-durations[j['row']['video_id']]),'E')
        if phase in ('all','query'):
            current='query_generation'; publish(True); run_phase(jobs,'query')
            hashes={r['question_id']:sha256(run/'queries'/f'{r["question_id"]}.json') for r in rows}
            if (run/'query_freeze.json').exists():
                if read_json(run/'query_freeze.json')['sha256']!=hashes: raise ProtocolError('Query freeze mismatch')
            else: durable(run/'query_freeze.json',dict(utc=utc(),sha256=hashes,questions=50,before_any_scoring=True))
            current='Q_offline_scoring'; publish(True)
            run_phase(sorted(jobs,key=lambda j:-durations[j['row']['video_id']]),'Q')
        current='model_phases_completed'
    except BaseException as exc:
        error=repr(exc); current='paused'; print(traceback.format_exc(),flush=True)
    finally:
        stop.set()
        for q in queues.values(): q.put(None)
        while any(p.is_alive() for p in processes.values()):
            for p in processes.values():
                if p.pid is not None: p.join(timeout=1)
        publish(True); log.close()
        durable(run/f'execution_{time.time_ns()}.json',dict(status=current,error=error,phase=phase,wall_seconds=time.perf_counter()-start,
                gpus=selected_gpus,resume=resume,utc=utc(),preflight_seconds=evidence['seconds']))
    if error: raise RuntimeError(error)
