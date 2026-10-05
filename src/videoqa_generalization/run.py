"""独立全量验证协调器：三阶段调用隔离、持久检查点与后台恢复。"""
import contextlib
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import time
import traceback
from videoqa_runtime.common import ROOT, read_json, sha256, offline_environment
from videoqa_full.state import durable, exclusive, rows, pilot_ids, invoke_once, utc
from videoqa_full.prepare import environment
from videoqa_budget32.frames import FrameProvider, validate_frames
from videoqa_lmms_bridge.common import framework_identity, load_task, task_doc, POST_PROMPT
from .inputs import BASE, OLD_QWEN, WFS_DEV, DEV, LLAVA_BASE, LLAVA_UNIFORM, frozen

QPY = '/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'
LPY = '/App/conda/envs/explore_video_qa_lmms/bin/python'
STAGES = {'qwen16': ('qwen', 'RD-1.2-exp12', 16, 2700),
          'llava16': ('llava', 'RD-1.2-exp12', 16, 2700),
          'wfs32': ('qwen', 'REF-WFS-SB', 32, 2500)}


def code_sources():
    """接口：只绑定实际执行代码与冻结配置；文档更新不改变实验指纹。"""
    names = ['scripts/run_exp12_generalization_full.py']
    names += [str(p.relative_to(ROOT)) for p in (ROOT/'src/videoqa_generalization').glob('*.py')]
    for folder in ['videoqa_budget32', 'videoqa_lmms_bridge', 'videoqa_methods']:
        names += [str(p.relative_to(ROOT)) for p in (ROOT/'src'/folder).glob('*.py')]
    names += ['src/videoqa_full/state.py', 'src/videoqa_full/prepare.py',
              'src/videoqa_full/qwen_backend.py', 'src/videoqa_full/qwen_fa2_backend.py',
              'src/videoqa_runtime/llava_backend.py', 'src/videoqa_runtime/common.py',
              'configs/retrieval_density_v1.json', 'configs/llava_source_manifest.json']
    return {n: sha256(ROOT/n) for n in sorted(set(names))}


def prepare_stage(root, name, gpus):
    """接口：冻结单组2700输入，WFS严格导入200来源，剩余2500调用独立计数。"""
    root = Path(root); run = root/name; run.mkdir(parents=True, exist_ok=True)
    kind, method, budget, maximum = STAGES[name]
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists(): return verify(run)
        pre = read_json(root/'preflight.json'); summary = read_json(root/'input_summary.json')
        if summary['status'] != 'passed' or summary['m32_control_exact'] != 2700:
            raise ValueError('Input control not accepted')
        baseline = read_json((BASE if kind == 'qwen' else LLAVA_UNIFORM)/'protocol.json')
        folder = 'wfs' if name == 'wfs32' else 'm16'; exports = {}; sources = {}
        for row in rows():
            q = row['question_id']; path = root/'inputs'/folder/(q+'.json')
            item = read_json(path); validate_frames(item['selection'], budget=budget)
            rel = f'selections/{method}/{q}.json'; dest = run/rel
            if dest.exists():
                if sha256(dest) != sha256(path): raise ValueError('Stage input differs')
            else:
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
            exports[rel] = sha256(dest); sources[str(path)] = sha256(path)
        code = code_sources()
        for rel in code:
            dest = run/'code_snapshot'/rel; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/rel, dest)
        spec = dict(model=baseline['model'], model_kind=kind, methods=[method], stage=name,
            profile=dict(baseline.get('profile', baseline.get('reference_profile', {})), frame_budget=budget),
            source_frame_budget=budget, configuration='M16/K16' if name != 'wfs32' else 'K32 published WFS-SB',
            max_calls=maximum, max_calls_per_method=maximum, question_count=2700, allowed_gpus=gpus,
            model_files=pre['model_files'], video_stats=pre['video_stats'], video_hashes=pre['video_hashes'],
            framework=framework_identity(), environment=pre['environments'][kind],
            code_sha256=code, selection_exports=exports, input_sources=sources,
            preflight_sha256=sha256(root/'preflight.json'), input_summary_sha256=sha256(root/'input_summary.json'),
            pilot_question_ids=pilot_ids(rows()), imported_count=200 if name == 'wfs32' else 0,
            max_new_tokens=16, seed=2027, utc=utc())
        frozen(run/'protocol.json', spec)
        if name == 'wfs32': import_wfs(run, spec)
        return spec


def verify(run):
    """接口：恢复前复核环境、代码、实际框架及上游完整性，不改冻结协议。"""
    run = Path(run); spec = read_json(run/'protocol.json'); root = run.parent
    if framework_identity() != spec['framework']: raise ValueError('Framework changed')
    py = QPY if spec['model_kind'] == 'qwen' else LPY
    if environment(py) != spec['environment']: raise ValueError('Environment changed')
    for n, h in spec['code_sha256'].items():
        if sha256(ROOT/n) != h: raise ValueError('Code changed: '+n)
    if sha256(root/'preflight.json') != spec['preflight_sha256'] or sha256(root/'input_summary.json') != spec['input_summary_sha256']:
        raise ValueError('Prepared inputs changed')
    for p, st in spec['model_files'].items():
        now = Path(p).stat()
        if (now.st_size, now.st_mtime_ns) != (st['bytes'], st['mtime_ns']): raise ValueError('Model stat changed')
    return spec


def import_wfs(run, spec):
    """接口：只导入冻结200条；校验原调用、实际Token与输入RGB，保存三份来源哈希。"""
    from videoqa_budget32.run import validate as validate_old
    old_spec = read_json(WFS_DEV/'protocol.json')
    if old_spec['environment'] != spec['environment'] or old_spec['framework'] != spec['framework']:
        raise ValueError('Reference QA protocol differs')
    ids = [r['question_id'] for r in read_json(DEV)['rows']]; data = {r['question_id']: r for r in rows()}
    records = {}
    for q in ids:
        old = validate_old(WFS_DEV, data[q], 'REF-WFS-SB', old_spec)
        source = WFS_DEV/'results/REF-WFS-SB'/(q+'.json')
        item = read_json(run/'selections/REF-WFS-SB'/(q+'.json'))
        if old['selection'] != item['selection'] or old['rgb_sha256'] != item['rgb_sha256']:
            raise ValueError('Reference imported pixels differ')
        origins = {str(WFS_DEV/folder/'REF-WFS-SB'/(q+'.json')): sha256(WFS_DEV/folder/'REF-WFS-SB'/(q+'.json'))
                   for folder in ['results', 'calls', 'raw_generations']}
        value = dict(old, protocol_sha256=sha256(run/'protocol.json'), imported=True,
            source_result=str(source), import_sources=origins, historical_protocol_sha256=old['protocol_sha256'])
        frozen(run/'results/REF-WFS-SB'/(q+'.json'), value); records[q] = origins
    frozen(run/'import_manifest.json', dict(count=200, question_ids=ids, sources=records,
        max_new_calls=2500, imported_calls_not_new=True))


def verify_wfs_inputs(root):
    """接口：所有正式模型加载前核验WFS200导入条件，避免到第三阶段才发现不可复用。"""
    from videoqa_budget32.run import validate as validate_old
    root=Path(root); old_spec=read_json(WFS_DEV/'protocol.json'); base=read_json(BASE/'protocol.json')
    # 步骤1：完整问答协议和依赖必须与本轮Qwen源配置相同，逐题原账本已有返回。
    for field in ['model','environment','framework','profile']:
        if old_spec[field]!=base[field]: raise ValueError('Reference import protocol differs: '+field)
    data={r['question_id']:r for r in rows()}; records={}
    for row in read_json(DEV)['rows']:
        q=row['question_id']; old=validate_old(WFS_DEV,data[q],'REF-WFS-SB',old_spec)
        item=read_json(root/'inputs/wfs'/(q+'.json'))
        if old['selection']!=item['selection'] or old['rgb_sha256']!=item['rgb_sha256']:
            raise ValueError('Reference import input differs')
        for folder in ['results','calls','raw_generations']:
            path=WFS_DEV/folder/'REF-WFS-SB'/(q+'.json'); records[str(path)]=sha256(path)
    # 步骤2：冻结600份来源，恢复时内容必须相同；尚不在正式账本伪造新调用。
    frozen(root/'wfs_import_preflight.json',dict(status='passed',importable_questions=200,
        new_generations=0,sources=records,source_protocol_sha256=sha256(WFS_DEV/'protocol.json')))


def validate(run, row, spec):
    """接口：逐题输入、原任务计分、生成账本核验；导入答案与新调用分别审核。"""
    run = Path(run); method = spec['methods'][0]; q = row['question_id']
    r = read_json(run/'results'/method/(q+'.json'))
    p = run/'selections'/method/(q+'.json'); item = read_json(p)
    if sha256(p) != spec['selection_exports'][str(p.relative_to(run))]: raise ValueError('Input changed')
    validate_frames(item['selection'], budget=spec['source_frame_budget'])
    if r['selection'] != item['selection'] or r['rgb_sha256'] != item['rgb_sha256']:
        raise ValueError('Cross-backend input identity mismatch')
    if r['protocol_sha256'] != sha256(run/'protocol.json'): raise ValueError('Result protocol mismatch')
    for k in ['question_id', 'video_id', 'question', 'options', 'stratum']:
        if r[k] != row[k]: raise ValueError('Question identity changed')
    a = r['answer']; o = a['observed']
    metric = load_task().videomme_process_results(task_doc(row), [a['raw_output']])['videomme_perception_score']
    if metric != a['framework_metric'] or r['correct'] != (metric['answer'] == metric['pred_answer']):
        raise ValueError('Task score mismatch')
    expected_context = load_task().videomme_doc_to_text(task_doc(row), {'post_prompt': POST_PROMPT})
    if o['context'] != expected_context or o['generate_calls'] != 1 or o['prefill_calls'] != 1 or o['checked_logit_steps'] < 1:
        raise ValueError('Generation observation differs')
    if not 1 <= len(o['generated_token_ids']) <= 16: raise ValueError('Token generation cap differs')
    n = len(item['selection']['selected_frames'])
    if spec['model_kind'] == 'llava':
        if o['visual_tokens'] != n*210 or o['pixel_sha256'] != o['expected_pixel_sha256'] or o['max_new_tokens'] != 16:
            raise ValueError('LLaVA visual protocol differs')
    else:
        if o['source_frames'] != n or o['processor_pixel_sha256'] != o['model_pixel_sha256']:
            raise ValueError('Qwen visual protocol differs')
        from videoqa_budget32.frames import padded_count
        h, w = o['processed_size']
        if o['encoded_frames'] != padded_count(n) or h%28 or w%28 or not 100352 <= h*w <= 602112:
            raise ValueError('Qwen grid or padding differs')
        if o['generation_kwargs']['max_new_tokens'] != 16 or o['second_per_grid_ts'] != [1.0]:
            raise ValueError('Qwen temporal or generation protocol differs')
    if r.get('imported'):
        if spec['stage'] != 'wfs32': raise ValueError('Unexpected imported answer')
        for path, digest in r['import_sources'].items():
            if sha256(path) != digest: raise ValueError('Imported source changed')
    else:
        call = read_json(run/'calls'/method/(q+'.json')); raw = read_json(run/'raw_generations'/method/(q+'.json'))
        if call['status'] != 'completed' or call['answer'] != a or call['gpu'] != r['gpu']:
            raise ValueError('Answer ledger mismatch')
        identity = dict(question_id=q, method=method, protocol_sha256=r['protocol_sha256'], gpu=r['gpu'])
        if raw['identity'] != identity or raw['generation_state'] != {'started': True, 'returned': True}:
            raise ValueError('Unknown raw generation')
    return r


def recover(run, row, spec):
    """接口：只有原始生成已确认返回才恢复后处理，绝不因缺结果重新答题。"""
    method = spec['methods'][0]; q = row['question_id']; p = run/'calls'/method/(q+'.json')
    if not p.exists(): return
    call = read_json(p); raw_path = run/'raw_generations'/method/(q+'.json')
    fp = sha256(run/'protocol.json')
    if call['protocol_sha256'] != fp: raise ValueError('Call fingerprint changed')
    if call['status'] != 'completed':
        raw = read_json(raw_path)
        if raw['identity'] != dict(question_id=q, method=method, protocol_sha256=fp, gpu=call['gpu']):
            raise ValueError('Raw identity mismatch')
        if spec['model_kind'] == 'qwen':
            from videoqa_budget32.observe import result_from_raw
            answer = result_from_raw(raw, row)
        else:
            from .llava_observe import result_from_raw
            restored = result_from_raw(raw, row); answer = dict(restored['answer'], observed=restored['observed'])
        call.update(status='completed', answer=answer, generation_state=raw['generation_state'], recovered_postprocessing=True)
        durable(p, call)
    dest = run/'results'/method/(q+'.json')
    if not dest.exists():
        item = read_json(run/'selections'/method/(q+'.json')); a = call['answer']; metric = a['framework_metric']
        durable(dest, dict(status='completed', model=spec['model'], method=method, protocol_sha256=fp,
            **{k:row[k] for k in ['question_id','video_id','question','options','stratum']},
            selection=item['selection'], rgb_sha256=item['rgb_sha256'], answer=a, gpu=call['gpu'],
            correct=metric['answer']==metric['pred_answer'], reused_answer=True,
            timings={}, timing_kind='resumed_not_primary', utc=utc()))


def process(run, row, backend, adapter, gpu, spec):
    """接口：共享精确RGB至一次生成；结果落盘后立刻校验，异常停止派发。"""
    method = spec['methods'][0]; q = row['question_id']
    if (run/'results'/method/(q+'.json')).exists(): validate(run, row, spec); return
    if spec['model_kind'] == 'qwen':
        from videoqa_budget32.run import process as qprocess
        qprocess(run, row, method, backend, adapter, gpu, spec)
        validate(run, row, spec); return
    from .llava_observe import observe
    with exclusive(run/'locks'/method/(q+'.lock')):
        start = time.perf_counter(); item = read_json(run/'selections'/method/(q+'.json'))
        images, hashes = FrameProvider(item['selection']).decode()
        if hashes != item['rgb_sha256']: raise ValueError('Prepared RGB differs')
        decoded = time.perf_counter(); fp = sha256(run/'protocol.json')
        identity = dict(question_id=q, method=method, protocol_sha256=fp, gpu=gpu)
        def invoke(state):
            result = observe(backend, adapter, images, item['selection'], row, state,
                run/'raw_generations'/method/(q+'.json'), identity)
            return dict(result['answer'], observed=result['observed'])
        try: answer, reused, seconds = invoke_once(run, q, method, fp, gpu, invoke)
        finally:
            for image in images: image.close()
        metric = answer['framework_metric']
        durable(run/'results'/method/(q+'.json'), dict(status='completed', model=spec['model'], method=method,
            protocol_sha256=fp, **{k:row[k] for k in ['question_id','video_id','question','options','stratum']},
            selection=item['selection'], rgb_sha256=hashes, answer=answer, gpu=gpu, reused_answer=reused,
            correct=metric['answer']==metric['pred_answer'], timings=dict(final_decode_seconds=decoded-start,
                cached_input_to_persisted_answer_seconds=time.perf_counter()-start, call_start_checkpoint_seconds=seconds), utc=utc()))
        validate(run, row, spec)


def worker(run, gpu, jobs, events, stop):
    """接口：一张物理卡常驻一个模型；合成预热单列，任何运行异常终止派发。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    offline_environment(); run = Path(run); active = None
    with (run/'logs'/f'gpu_{gpu}.log').open('a', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            import torch
            from PIL import Image
            from videoqa_lmms_bridge.bridge import synthetic_row
            torch.set_num_threads(2); torch.set_num_interop_threads(1)
            spec = read_json(run/'protocol.json'); t = time.perf_counter()
            if spec['model_kind'] == 'qwen':
                from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
                from videoqa_budget32.qwen_adapter import FrozenInputQwen
                from videoqa_budget32.observe import observe
                backend = QwenFA2VideoBackend(); adapter = FrozenInputQwen(backend)
            else:
                from videoqa_runtime.llava_backend import LlavaBackend
                from videoqa_lmms_bridge.llava_adapter import FrozenInputLlavaVid
                backend = LlavaBackend(); adapter = FrozenInputLlavaVid(backend)
            loading = time.perf_counter()-t; t = time.perf_counter(); n = spec['source_frame_budget']
            images = [Image.new('RGB',(896,672),(40,i*7,150)) for i in range(n)]
            if spec['model_kind'] == 'qwen': syn = observe(backend, adapter, images, synthetic_row(n), {})
            else:
                from .llava_observe import observe
                s = dict(video=dict(path='synthetic.mp4', duration_seconds=float(n)),
                    selected_frames=[dict(timestamp_seconds=float(i)) for i in range(n)])
                syn = observe(backend, adapter, images, s, synthetic_row(n), {})
            for image in images: image.close()
            durable(run/'workers'/f'{gpu}_{os.getpid()}.json',dict(gpu=gpu,pid=os.getpid(),load_seconds=loading,
                warmup_seconds=time.perf_counter()-t,synthetic_generations=1,warmup=syn,load_report=backend.load_report))
            events.put(dict(type='ready', gpu=gpu))
            # 步骤1：调用ID由阶段、方法、题号决定；失败后不能领取新任务。
            while not stop.is_set():
                row = jobs.get()
                if row is None: break
                active = row['question_id']; events.put(dict(type='active', gpu=gpu, question_id=active))
                process(run, row, backend, adapter, gpu, spec)
                events.put(dict(type='completed', gpu=gpu, question_id=active)); active = None
        except BaseException as exc:
            stop.set(); traceback.print_exc()
            durable(run/'failures'/f'{gpu}_{time.time_ns()}.json',dict(error=repr(exc),question_id=active,traceback=traceback.format_exc(),utc=utc()))
            events.put(dict(type='failed',gpu=gpu,error=repr(exc)))


def execute(run):
    """接口：后台独立先行验收、按时长派发和安全恢复；已完成阶段不加载模型。"""
    from videoqa_runtime.baseline_run import available_gpus
    run = Path(run)
    with exclusive(run/'coordinator.lock'):
        spec = verify(run); data = rows(); byid = {r['question_id']:r for r in data}; method = spec['methods'][0]
        if spec['stage']=='wfs32': import_wfs(run, spec)
        done = set(); pinned = {}; start = time.perf_counter(); phase = 'checking'; active = {}; gpus = []
        def publish():
            elapsed = time.perf_counter()-start; fresh = len(done)-spec['imported_count']
            durable(run/'status.json',dict(status=phase,completed=len(done),total=2700,new_completed=fresh,
                imported=spec['imported_count'],pid=os.getpid(),sid=os.getsid(0),active=active,gpus=gpus,
                elapsed_seconds=elapsed,eta_seconds=(2700-len(done))*elapsed/fresh if fresh>0 else None,utc=utc()))
        # 步骤1：调用预算按已开始文件计；不确定状态先核查，不以缺结果触发重答。
        calls = list((run/'calls'/method).glob('*.json'))
        if len(calls)>spec['max_calls']: raise ValueError('Call budget exceeded')
        for p in calls:
            if p.stem not in byid: raise ValueError('Unknown question call')
            recover(run, byid[p.stem], spec); call = read_json(p)
            if call['gpu'] not in spec['allowed_gpus']: raise ValueError('Physical GPU differs')
            pinned[p.stem]=call['gpu']
        for p in (run/'results'/method).glob('*.json'):
            if p.stem not in byid: raise ValueError('Unknown result')
            validate(run, byid[p.stem], spec); done.add(p.stem)
        if len(done)==2700: return
        required={gpu for q,gpu in pinned.items() if q not in done}
        while True:
            try: gpus, states = available_gpus(spec['allowed_gpus'])
            except RuntimeError: gpus=[]; states={}
            if gpus and required<=set(gpus): break
            phase='waiting_resources'; publish(); time.sleep(15)
        for folder in ['logs','workers','failures']: (run/folder).mkdir(exist_ok=True)
        durable(run/'gpu_preflight.json',dict(actual=gpus,states=states,utc=utc()))
        ctx=mp.get_context('spawn'); jobs={g:ctx.Queue() for g in gpus}; events=ctx.Queue(); stop=ctx.Event()
        children={}; ready=set(); busy=set(); pending=[]
        def schedule():
            nonlocal phase,pending
            pilot=spec['pilot_question_ids']
            if any(q not in done for q in pilot): phase='pilot'; pending=[q for q in pilot if q not in done]
            else:
                for q in pilot: validate(run,byid[q],spec)
                durable(run/'pilot_acceptance.json',dict(status='passed',question_ids=pilot,accuracy_not_gate=True))
                phase='running'; pending=sorted([q for q in byid if q not in done],key=lambda q:(-spec['video_stats'][byid[q]['video_id']].get('duration_seconds',
                    read_json(run/'selections'/method/(q+'.json'))['selection']['video']['duration_seconds']),q))
        def refill():
            for gpu in gpus:
                if gpu in busy: continue
                q=next((q for q in pending if q not in pinned or pinned[q]==gpu),None)
                if q is not None: pending.remove(q); busy.add(gpu); jobs[gpu].put(byid[q])
        # 步骤2：后台放行只基于协议校验，派发期间不读取分数决策。
        try:
            phase='loading'
            for gpu in gpus:
                child=ctx.Process(target=worker,args=(str(run),gpu,jobs[gpu],events,stop)); child.start(); children[gpu]=child
            publish()
            while len(done)<2700:
                try: event=events.get(timeout=5)
                except queue.Empty: event=None
                if event:
                    if event['type']=='failed': raise RuntimeError(str(event))
                    if event['type']=='ready':
                        ready.add(event['gpu'])
                        if len(ready)==len(gpus): schedule(); refill()
                    elif event['type']=='active': active[event['gpu']]=event['question_id']
                    elif event['type']=='completed':
                        done.add(event['question_id']); busy.remove(event['gpu']); active.pop(event['gpu'],None)
                        if phase=='pilot' and not pending and not busy: schedule()
                        refill()
                for gpu, child in children.items():
                    if child.exitcode is not None: raise RuntimeError(f'Worker {gpu} exited {child.exitcode}')
                publish()
        except BaseException:
            phase='paused'; stop.set(); publish(); raise
        finally:
            for job in jobs.values(): job.put(None)
            while any(child.is_alive() for child in children.values()):
                for child in children.values(): child.join(timeout=1)
                publish()
        # 步骤3：全部闭合后核验恢复不增加生成，再允许后台切换下一个模型。
        phase='completed'; publish()
        durable(run/'execution.json',dict(status='completed',wall_seconds=time.perf_counter()-start,
            new_answer_calls=len(list((run/'calls'/method).glob('*.json'))),actual_gpus=gpus,utc=utc()))


def audit(run):
    """接口：全2700题检查与实际完成恢复；导入200不伪造成本或新调用。"""
    run=Path(run); spec=verify(run); method=spec['methods'][0]
    ids={r['question_id'] for r in rows()}; paths=list((run/'results'/method).glob('*.json'))
    if {p.stem for p in paths} != ids: raise ValueError('Incomplete result population')
    for row in rows(): validate(run,row,spec)
    calls=list((run/'calls'/method).glob('*.json'))
    if len(calls)!=spec['max_calls']: raise ValueError('Expected new call count differs')
    before={str(p):sha256(p) for folder in ['results','calls','raw_generations'] for p in (run/folder).glob('*/*.json')}
    execute(run)
    after={str(p):sha256(p) for folder in ['results','calls','raw_generations'] for p in (run/folder).glob('*/*.json')}
    if before!=after: raise ValueError('Completed recovery changed answers')
    durable(run/'completion_audit.json',dict(status='passed',questions=2700,new_calls=len(calls),
        imported=spec['imported_count'],unchanged_files=len(before),extra_generations=0,utc=utc()))
