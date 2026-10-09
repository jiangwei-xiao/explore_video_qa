"""022-1真实计时图诊断：固定两个自动输入，不把局部结果当作方法成绩。"""
import os
from pathlib import Path
import subprocess
import time

from videoqa_runtime.common import ROOT, read_json, sha256, offline_environment
from videoqa_full.state import durable, exclusive, rows, utc, invoke_once
from videoqa_full.prepare import environment
from videoqa_budget32.frames import FrameProvider, validate_frames
from videoqa_audit.framework_contraction import RUN8
from videoqa_audit.region_scope_runtime import (
    PYTHONS, QA_FILES, parent_result, parent_call_sources, cpu_identity, source_order,
)
from videoqa_lmms_bridge.common import framework_identity, load_task, task_doc

QUESTION = '022-1'
PROFILE = 'llava8'
SOURCE = ROOT/'outputs/methods/rd12_exp30_confirmed_text_state_20261009_r1'
CONDITIONS = ('original', 'candidate')
FILES = ('src/videoqa_audit/text_state_diagnostic.py', 'scripts/run_text_state_diagnostic.py') + QA_FILES
PYTHON = PYTHONS[PROFILE]


def row():
    """接口：仅读取冻结题面，答案只在返回后计分；不通过新答案选择诊断题。"""
    return next(r for r in rows() if r['question_id'] == QUESTION)


def prepare(root):
    """接口：在CPU冻结原输入与exp30实际输入、像素/Token及两个真实调用额度。"""
    root = Path(root); root.mkdir(parents=True, exist_ok=True); offline_environment()
    with exclusive(root/'prepare.lock'):
        if (root/'protocol.json').exists():
            return verify(root)
        import torch
        from transformers import AutoTokenizer
        from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        old_spec = read_json(RUN8/'protocol.json')
        if framework_identity() != old_spec['framework'] or environment(PYTHON) != old_spec['environment']:
            raise ValueError('Historical framework/environment changed')
        # 步骤1：父模型与推理接口只读验收，新选择源及旧物理调用都绑定指纹。
        for path, value in old_spec['model_files'].items():
            if sha256(path) != value['sha256']:
                raise ValueError('Frozen model file changed: '+path)
        for name in QA_FILES:
            if name in old_spec['code_sha256'] and sha256(ROOT/name) != old_spec['code_sha256'][name]:
                raise ValueError('Historical inference code changed: '+name)
        old_path = parent_result(PROFILE, QUESTION); old = read_json(old_path)
        candidate_path = SOURCE/'offline'/PROFILE/(QUESTION+'.json'); candidate = read_json(candidate_path)
        data = row(); video = Path(old['selection']['video']['path'])
        expected_video = old_spec['video_hashes'][data['video_id']]
        if sha256(video) != expected_video:
            raise ValueError('Frozen video bytes changed')
        sources = parent_call_sources(PROFILE, old_path, old)
        for path in (candidate_path, SOURCE/'protocol.json', SOURCE/candidate['trace_file'], RUN8/'protocol.json'):
            sources[str(path)] = sha256(path)
        sources.update(candidate['sources'])
        for path, h in sources.items():
            if sha256(path) != h:
                raise ValueError('Frozen selection source changed: '+path)
        tools = (SigLipImageProcessor(), AutoTokenizer.from_pretrained(
            '/home/models/lmms-lab/LLaVA-Video-7B-Qwen2', local_files_only=True))
        selections = {'original': old['selection'], 'candidate': candidate['selection']}
        items = {}; exports = {}
        # 步骤2：逐输入精确解码并使用官方CPU处理器，原输入必须与历史真实输入一致。
        for condition, selection in selections.items():
            validate_frames(selection, 8)
            images, rgb = FrameProvider(selection).decode()
            try:
                identity = cpu_identity(PROFILE, tools, images, data)
                if condition == 'original':
                    obs = old['answer']['observed']
                    if (rgb != old['rgb_sha256'] or identity['pixel_sha256'] != obs['pixel_sha256']
                            or identity['prompt'] != obs['prompt'] or identity['input_token_ids'] != obs['input_token_ids']):
                        raise ValueError('Historical input pixels/prompt not reproduced')
                # 实际384输入视图仅用于可见性审核，不作为新的候选或提示文本。
                from PIL import Image
                pixels = tools[0].preprocess(images, return_tensors='pt')['pixel_values']
                for index, (im, frame) in enumerate(zip(images, selection['selected_frames'])):
                    out = root/'review'/condition/(str(frame['source_pts'])+'.png')
                    out.parent.mkdir(parents=True, exist_ok=True); im.save(out)
                    visible = ((pixels[index].permute(1, 2, 0).numpy()*0.5+0.5)*255).clip(0, 255).astype('uint8')
                    Image.fromarray(visible).save(out.with_name(out.stem+'_384.png'))
            finally:
                for im in images: im.close()
            item = dict(question_id=QUESTION, condition=condition, selection=selection,
                        rgb_sha256=rgb, input_identity=identity)
            dest = root/'inputs'/(condition+'.json'); durable(dest, item)
            items[condition] = item; exports[str(dest)] = sha256(dest)
        old_keys = set(map(tuple, source_order(selections['original'])['frames']))
        new_keys = set(map(tuple, source_order(selections['candidate'])['frames']))
        if not new_keys-old_keys or len(old_keys) != len(new_keys) or len(old_keys) != 8:
            raise ValueError('Diagnostic input change or budget invalid')
        # 步骤3：在新回答前冻结执行顺序、输入变化及预算，不按结果追加有利版本。
        ordinal = next(i for i, r in enumerate(rows()) if r['question_id'] == QUESTION)
        order = list(CONDITIONS if ordinal % 2 == 0 else CONDITIONS[::-1])
        stat = video.stat()
        spec = dict(question_id=QUESTION, profile=PROFILE, method='RD-1.2-exp30',
            scope='Local automatic-input diagnostic, not algorithm accuracy or exp30 development gate',
            call_order=order, max_real_calls=2, max_synthetic_calls=1, frame_budget=8,
            historical_prediction=old['answer']['parsed_answer'], sources=sources, input_exports=exports,
            added_source_frames=sorted(new_keys-old_keys), removed_source_frames=sorted(old_keys-new_keys),
            framework=old_spec['framework'], environment=old_spec['environment'], model_files=old_spec['model_files'],
            video=dict(path=str(video), sha256=expected_video, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
            code_sha256={n:sha256(ROOT/n) for n in FILES}, allowed_gpus=list(map(str, range(8))), utc=utc())
        for name in FILES:
            import shutil
            out = root/'code_snapshot'/name; out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, out)
        durable(root/'protocol.json', spec)
        return spec


def verify(root):
    """接口：恢复时检查冻结输入、代码和环境；已有生成不因观察超时而重复。"""
    root = Path(root); spec = read_json(root/'protocol.json')
    if spec['question_id'] != QUESTION or spec['max_real_calls'] != 2 or set(spec['call_order']) != set(CONDITIONS):
        raise ValueError('Diagnostic population/budget changed')
    if framework_identity() != spec['framework'] or environment(PYTHON) != spec['environment']:
        raise ValueError('Diagnostic environment changed')
    for path, h in {**spec['sources'], **spec['input_exports']}.items():
        if sha256(path) != h: raise ValueError('Diagnostic frozen input changed: '+path)
    for name, h in spec['code_sha256'].items():
        if sha256(ROOT/name) != h: raise ValueError('Diagnostic frozen code changed: '+name)
    for path, value in spec['model_files'].items():
        st = Path(path).stat()
        if st.st_size != value['bytes'] or st.st_mtime_ns != value['mtime_ns']:
            raise ValueError('Diagnostic model identity changed')
    v = spec['video']; st = Path(v['path']).stat()
    if st.st_size != v['bytes'] or st.st_mtime_ns != v['mtime_ns']:
        raise ValueError('Diagnostic video identity changed')
    return spec


def recover_return(root, condition, qid):
    """接口：只有完整原始返回可以恢复后处理；未知状态禁止再次调用。"""
    root = Path(root); call_path = root/'calls'/condition/(qid+'.json')
    if not call_path.exists(): return
    c = read_json(call_path)
    if c['protocol_sha256'] != sha256(root/'protocol.json'):
        raise ValueError('Recovery call protocol differs')
    if c['status'] == 'completed': return
    raw_path = root/'raw_generations'/condition/(qid+'.json')
    if not raw_path.exists(): raise RuntimeError('Uncertain generation, no retry: '+str(call_path))
    raw = read_json(raw_path)
    if raw['identity'] != dict(question_id=qid, method=condition, protocol_sha256=c['protocol_sha256'], gpu=c['gpu']):
        raise ValueError('Recovery raw identity differs')
    from videoqa_generalization.llava_observe import result_from_raw
    from videoqa_lmms_bridge.bridge import synthetic_row
    value = result_from_raw(raw, row() if qid == QUESTION else synthetic_row(8))
    c.update(status='completed', answer=dict(value['answer'], observed=value['observed']),
             generation_state={'started':True, 'returned':True}, recovered_postprocessing=True)
    durable(call_path, c)


def synthetic():
    """接口：固定8张合成输入独立预热，不使用研究题的答案或原片。"""
    from PIL import Image
    from videoqa_lmms_bridge.bridge import synthetic_row
    images = [Image.new('RGB', (896, 672), (30, 10+i*3, 150)) for i in range(8)]
    selection = dict(video=dict(path='synthetic.mp4', duration_seconds=8.0),
                     selected_frames=[dict(timestamp_seconds=float(i)) for i in range(8)])
    return images, selection, synthetic_row(8)


def validate_answer(item, answer, data):
    """接口：检查模型实际像素/Token、1680视觉Token、16新Token及参考计分。"""
    expected = item['input_identity']; o = answer['observed']
    if any(o[k] != expected[k] for k in ('pixel_sha256', 'prompt', 'input_token_ids')):
        raise ValueError('Actual diagnostic input differs')
    if o['visual_tokens'] != 1680 or o['max_new_tokens'] != 16:
        raise ValueError('Diagnostic visual/generation protocol differs')
    if o['generate_calls'] != 1 or o['prefill_calls'] != 1 or not 1 <= len(o['generated_token_ids']) <= 16:
        raise ValueError('Diagnostic call/token count invalid')
    metric = load_task().videomme_process_results(task_doc(data), [answer['raw_output']])['videomme_perception_score']
    if metric != answer['framework_metric'] or answer['parsed_answer'] != (metric['pred_answer'] or None):
        raise ValueError('Diagnostic reference scoring differs')


def worker(root, gpu):
    """接口：同一卡同一长期LLaVA进程执行两个冻结条件，失败不自动重试。"""
    root = Path(root); offline_environment(); spec = verify(root)
    import torch
    from videoqa_runtime.llava_backend import LlavaBackend
    from videoqa_generalization.llava_budget8 import adapter_for
    from videoqa_generalization.llava_observe import observe
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(2027)
    started = time.perf_counter(); backend = LlavaBackend(); adapter = adapter_for(backend)
    durable(root/'load_report.json', dict(gpu=gpu, seconds=time.perf_counter()-started, report=backend.load_report))
    fp = sha256(root/'protocol.json')
    # 步骤1：合成仅一次，有完整返回才恢复；正式账本与合成账本严格隔离。
    def invoke(images, selection, data, condition, qid):
        raw_path = root/'raw_generations'/condition/(qid+'.json')
        identity = dict(question_id=qid, method=condition, protocol_sha256=fp, gpu=gpu)
        def generate(state):
            v = observe(backend, adapter, images, selection, data, state, raw_path, identity)
            return dict(v['answer'], observed=v['observed'])
        return invoke_once(root, qid, condition, fp, gpu, generate)[0]
    recover_return(root, 'synthetic', 'warmup')
    images, selection, data = synthetic()
    try: invoke(images, selection, data, 'synthetic', 'warmup')
    finally:
        for im in images: im.close()
    # 步骤2：同题条件顺序事前交替；前向前再次核验输入，回答正确性不控制重跑。
    for condition in spec['call_order']:
        durable(root/'status.json', dict(status='running', condition=condition, gpu=gpu, pid=os.getpid(), sid=os.getsid(0), utc=utc()))
        item = read_json(root/'inputs'/(condition+'.json')); recover_return(root, condition, QUESTION)
        call_path = root/'calls'/condition/(QUESTION+'.json')
        images = []
        if call_path.exists():
            answer = read_json(call_path)['answer']
        else:
            images, rgb = FrameProvider(item['selection']).decode()
            if rgb != item['rgb_sha256']:
                raise ValueError('Diagnostic RGB changed before generation')
            if cpu_identity(PROFILE, (backend.processor, backend.tokenizer), images, row()) != item['input_identity']:
                raise ValueError('Diagnostic pixels/prompt changed before generation')
            try: answer = invoke(images, item['selection'], row(), condition, QUESTION)
            finally:
                for im in images: im.close()
        if read_json(call_path)['gpu'] != gpu: raise ValueError('Partial diagnostic moved physical GPU')
        validate_answer(item, answer, row())
        persist_result(root, condition, answer, gpu)
    return audit(root)


def persist_result(root, condition, answer, gpu):
    """接口：生成结果与只恢复后处理共用保存边界，已有结果字节保持不变。"""
    root = Path(root); item_path = root/'inputs'/(condition+'.json')
    validate_answer(read_json(item_path), answer, row())
    dest = root/'results'/(condition+'.json')
    if not dest.exists():
        durable(dest, dict(condition=condition, answer=answer,
            correct=answer['parsed_answer'] == 'ABCD'[row()['answer_index']], gpu=gpu,
            input_sha256=sha256(item_path), utc=utc()))


def audit(root):
    """接口：两条结果独立复算，复测不一致时暂停归因，不伪装成方法精度。"""
    root = Path(root); spec = verify(root); results = {}
    for condition in CONDITIONS:
        item = read_json(root/'inputs'/(condition+'.json')); r = read_json(root/'results'/(condition+'.json'))
        call = read_json(root/'calls'/condition/(QUESTION+'.json')); raw = read_json(root/'raw_generations'/condition/(QUESTION+'.json'))
        if (call['status'] != 'completed' or call['answer'] != r['answer'] or raw['generation_state'] != {'started':True,'returned':True}
                or raw['identity'] != dict(question_id=QUESTION, method=condition, protocol_sha256=sha256(root/'protocol.json'), gpu=r['gpu'])
                or r['input_sha256'] != sha256(root/'inputs'/(condition+'.json'))):
            raise ValueError('Diagnostic result/call/raw identity differs')
        validate_answer(item, r['answer'], row())
        if r['correct'] != (r['answer']['parsed_answer'] == 'ABCD'[row()['answer_index']]):
            raise ValueError('Diagnostic correctness changed')
        results[condition] = r
    calls = list((root/'calls').glob('*/*.json'))
    if len(calls) != 3 or {p.parent.name for p in calls} != {'original', 'candidate', 'synthetic'}:
        raise ValueError('Diagnostic real/synthetic budget exceeded or incomplete')
    ledgers = [read_json(p) for p in calls]
    real = [c for c in ledgers if c['method'] in CONDITIONS]
    if (len({c['pid'] for c in real}) != 1 or len({c['gpu'] for c in ledgers}) != 1
            or any(c['status'] != 'completed' for c in ledgers)):
        raise ValueError('Paired diagnostic was not one process/GPU or synthetic uncertain')
    warm = read_json(root/'raw_generations/synthetic/warmup.json')
    if (warm['generation_state'] != {'started':True, 'returned':True}
            or warm['identity'] != dict(question_id='warmup', method='synthetic', protocol_sha256=sha256(root/'protocol.json'), gpu=ledgers[0]['gpu'])):
        raise ValueError('Synthetic raw return uncertain')
    matched = results['original']['answer']['parsed_answer'] == spec['historical_prediction']
    files = list((root/'results').glob('*.json')) + calls + list((root/'raw_generations').glob('*/*.json'))
    fingerprints = {str(p):sha256(p) for p in files}
    previous = root/'completion_audit.json'
    if previous.exists() and read_json(previous)['artifacts'] != fingerprints:
        raise ValueError('Completed diagnostic artifacts changed')
    summary = dict(status='completed', question_id=QUESTION, model_profile=PROFILE, real_calls=2, synthetic_calls=1,
        historical_prediction=spec['historical_prediction'], original_prediction=results['original']['answer']['parsed_answer'],
        candidate_prediction=results['candidate']['answer']['parsed_answer'], historical_replay_matches=matched,
        causal_attribution_allowed=matched, original_correct=results['original']['correct'], candidate_correct=results['candidate']['correct'],
        scope=spec['scope'], artifacts=fingerprints, additional_generation_on_recovery=0, utc=utc())
    durable(previous, summary); durable(root/'status.json', summary)
    return summary


def execute(root):
    """接口：排他后台等待真实空闲卡；完成恢复不加载模型，未知调用禁止重复。"""
    from videoqa_runtime.baseline_run import available_gpus
    root = Path(root); spec = verify(root)
    with exclusive(root/'coordinator.lock'):
        for condition in (*CONDITIONS, 'synthetic'):
            recover_return(root, condition, 'warmup' if condition == 'synthetic' else QUESTION)
        completed = []
        for condition in CONDITIONS:
            path = root/'calls'/condition/(QUESTION+'.json')
            if path.exists():
                call = read_json(path); completed.append(condition)
                persist_result(root, condition, call['answer'], call['gpu'])
        if all((root/'results'/(c+'.json')).exists() for c in CONDITIONS): return audit(root)
        # 同题配对要求同一个加载进程；原进程退出后不能把另一进程的答案强作配对。
        if completed:
            raise RuntimeError('Pair process ended before both returns; no replacement generation')
        assigned = root/'gpu_assignment.json'
        required = read_json(assigned)['gpu'] if assigned.exists() else None
        # 步骤1：他人占卡是等待状态，不是失败；不能以低瞬时利用率抢占常驻任务。
        while True:
            try: free, states = available_gpus(spec['allowed_gpus'])
            except RuntimeError: free, states = [], {}
            if free and (required is None or required in free): break
            durable(root/'status.json', dict(status='waiting_resources', real_calls_started=len(list((root/'calls').glob('original/*.json')))+len(list((root/'calls').glob('candidate/*.json'))),
                max_real_calls=2, required_gpu=required, pid=os.getpid(), sid=os.getsid(0), utc=utc()))
            time.sleep(15)
        gpu = required or free[0]
        uuid = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        assignment = dict(gpu=gpu, uuid=uuid)
        if assigned.exists() and read_json(assigned) != assignment: raise ValueError('Assigned physical GPU identity changed')
        if not assigned.exists(): durable(assigned, assignment)
        # 步骤2：独立子进程限定一张物理卡，后台等待/执行不依赖SSH或Codex连接。
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        with (root/'worker.log').open('a', buffering=1) as log:
            child = subprocess.Popen([PYTHON, str(ROOT/'scripts/run_text_state_diagnostic.py'), '--run-id', root.name, '--stage', 'worker', '--gpu', gpu],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                durable(root/'heartbeat.json', dict(pid=os.getpid(), sid=os.getsid(0), worker_pid=child.pid, gpu=gpu, utc=utc()))
                time.sleep(5)
            if child.returncode != 0:
                durable(root/'status.json', dict(status='paused', reason='Worker exited; inspect raw ledger before recovery', returncode=child.returncode, utc=utc()))
                raise RuntimeError('Diagnostic worker failed, no automatic retry')
        return audit(root)
