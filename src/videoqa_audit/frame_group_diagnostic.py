"""固定五组单槽位邻接上下文诊断；不修改选择器，不计作算法成绩。"""
import copy
import html
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from videoqa_runtime.common import ROOT, read_json, sha256, offline_environment
from videoqa_full.state import durable, digest, exclusive, invoke_once, rows, utc
from videoqa_budget32.frames import FrameProvider, validate_frames
from videoqa_audit.region_scope_runtime import PYTHONS, QA_FILES

REVIEW = ROOT/'outputs/analysis/dual_dataset_selection_review_20261015_r2/records'
MLVU = ROOT/'outputs/full_videoqa/mlvu_dev__S1__llava8_qwen32__four_methods__20261014_r1'
FILES = tuple(dict.fromkeys((
    'src/videoqa_audit/frame_group_diagnostic.py', 'scripts/run_frame_group_diagnostic.py',
    'src/videoqa_mlvu/backends.py', 'src/videoqa_mlvu/data.py',
    'src/videoqa_scorer_comparison/runtime.py', 'src/videoqa_budget32/frames.py',
    'src/videoqa_audit/cross_dataset_review.py',
) + QA_FILES))


def cases():
    """接口：固定题目、父输入、物理GPU和精确PTS；不根据新答案换题或换帧。"""
    specs = (
        ('D1', 'mlvu', 'mlvu-dev-Q8', 'qwen32', 'RD-1.2-exp12', '2', 2256384, 1942528, 2243584, 1936384, 'B'),
        ('D2', 'videomme', '113-3', 'qwen32', 'RD-1.2-exp07', '0', 1463963, 878378, 1554053, 833333, 'D'),
        ('D3', 'videomme', '386-1', 'llava8', 'RD-1.2-exp12', '1', 2384384, 1468928, 2371584, 1462784, 'C'),
        ('P1', 'mlvu', 'mlvu-dev-Q8', 'llava8', 'RD-1.2-exp12', '6', 2307584, 1942528, 2320384, 1936384, 'D'),
        ('P2', 'videomme', '386-1', 'qwen32', 'RD-1.2-exp12', '7', 2077184, 1468928, 2064384, 1462784, 'D'),
    )
    keys = ('id', 'dataset', 'question_id', 'profile', 'method', 'gpu', 'removed_pts', 'target_pts', 'control_pts', 'clue_pts', 'historical_option')
    return [dict(zip(keys, x), order=['O']+(['T', 'C'] if i % 2 == 0 else ['C', 'T'])) for i, x in enumerate(specs)]


def build_conditions(case, selection, pool):
    """接口：只替换关键帧后一槽；有理PTS顺序、源唯一性和全部未改位置必须成立。"""
    # 步骤1：按精确PTS匹配原合法池；不把相邻秒数近似当作源身份。
    frames = selection['selected_frames']; pts = [f['source_pts'] for f in frames]
    if len(set(pts)) != len(pts) or pts.count(case['removed_pts']) != 1:
        raise ValueError('Original source identity is not unique')
    slot = pts.index(case['removed_pts'])
    if slot == 0 or slot == len(pts)-1 or pts[slot-1] != case['clue_pts']:
        raise ValueError('Changed slot does not immediately follow the protected clue')
    out = {'O': copy.deepcopy(selection)}
    for condition, key in (('T', 'target_pts'), ('C', 'control_pts')):
        matches = [f for f in pool['candidates'] if f['source_pts'] == case[key]]
        if len(matches) != 1 or case[key] in pts or not pts[slot-1] < case[key] < pts[slot+1]:
            raise ValueError('Replacement absent, repeated, or outside adjacent fixed slots')
        current = copy.deepcopy(selection); current['selected_frames'][slot] = copy.deepcopy(matches[0])
        # 步骤2：没有重排或顺便改变其他槽位；候选编号仅作为来源追踪。
        if [i for i, (a, b) in enumerate(zip(frames, current['selected_frames'])) if a != b] != [slot]:
            raise ValueError('More than one slot changed')
        validate_frames(current, 8 if case['profile'] == 'llava8' else 32)
        out[condition] = current
    return out, slot


def temporal_pairs(selection):
    """接口：记录Qwen相邻两帧联合编码的源PTS；不解释为真实均匀采样时间。"""
    points = [f['source_pts'] for f in selection['selected_frames']]
    if len(points) % 2: points.append(points[-1])
    return [points[i:i+2] for i in range(0, len(points), 2)]


def canonical_option(case, answer):
    """接口：MLVU仅增加已确认的单字母句点归一化，Video-MME保持参考解析。"""
    native = answer['parsed_answer']
    if case['dataset'] == 'mlvu':
        from videoqa_audit.cross_dataset_review import normalize_option_period
        return normalize_option_period(answer['raw_output'], native)
    return native


def task_row(case):
    """接口：固定历史题面；答案只用于返回后计分，不参与条件构造。"""
    if case['dataset'] == 'mlvu':
        from videoqa_mlvu.data import rows as mlvu_rows
        data = mlvu_rows(MLVU)
    else: data = rows()
    return next(r for r in data if r['question_id'] == case['question_id'])


def cpu_input(case, tools, images, row):
    """接口：两个任务使用各自原生上下文，避免MLVU与Video-MME模板互相污染。"""
    if case['dataset'] == 'mlvu':
        from videoqa_mlvu.backends import cpu_identity
    else:
        from videoqa_audit.region_scope_runtime import cpu_identity
    return cpu_identity(case['profile'], tools, images, row)


def observed_input(profile, answer):
    """接口：CPU冻结与GPU真实观测使用同一像素、文本和视频网格定义。"""
    from videoqa_scorer_comparison.runtime import observed_identity
    return observed_identity(profile, answer)


def history_sources(case, source, old):
    """接口：追溯主调用及完整原始返回；同帧历史引用不充作重新生成。"""
    base = next(p for p in source.parents if (p/'protocol.json').exists())
    representative = old.get('representative', case['method'])
    candidates = [base/folder/'calls'/representative/(case['question_id']+'.json') for folder in ('', 'imported')]
    existing = [p for p in candidates if p.exists()]
    if len(existing) != 1: raise ValueError('Historical physical call is ambiguous')
    callpath = existing[0]; rawpath = callpath.parent.parent.parent/'raw_generations'/representative/(case['question_id']+'.json')
    call, raw = read_json(callpath), read_json(rawpath)
    if call['status'] != 'completed' or call['answer'] != old['answer'] or raw['generation_state'] != {'started': True, 'returned': True}:
        raise ValueError('Historical physical return is incomplete')
    if str(old['gpu']) != case['gpu'] or str(call['gpu']) != case['gpu']:
        raise ValueError('Historical physical GPU differs')
    return {str(p): sha256(p) for p in (source, base/'protocol.json', callpath, rawpath)}


def prepare_case(root, case):
    """接口：CPU冻结三条件及历史身份；GPU生成开始前所有条件必须已发布。"""
    from videoqa_full.prepare import environment
    from videoqa_lmms_bridge.common import framework_identity
    from videoqa_scorer_comparison.runtime import make_tools
    from videoqa_audit.framework_contraction import RUN8, RUN32
    offline_environment(); root = Path(root)
    with exclusive(root/'prepare.lock'):
        if (root/'protocol.json').exists(): return verify(root)
        # 步骤1：绑定父答案、合法候选、模型字节与环境，不接触历史输出。
        recordpath = REVIEW/case['dataset']/case['profile']/(case['question_id']+'.json')
        record = read_json(recordpath); source = Path(record['answer_sources'][case['method']]); old = read_json(source)
        poolpath = Path(record['pool_source']); pool = read_json(poolpath)
        parent = (RUN32 if case['profile'] == 'qwen32' else RUN8)/'protocol.json'
        parent_spec = read_json(parent); current_env = environment(PYTHONS[case['profile']])
        if framework_identity() != parent_spec['framework'] or current_env != parent_spec['environment']:
            raise ValueError('Historical model/framework environment changed')
        sources = history_sources(case, source, old)
        sources.update({str(p): sha256(p) for p in (recordpath, poolpath, parent)})
        if case['dataset'] == 'mlvu':
            ml = read_json(MLVU/'protocol.json')
            # 两任务的文字/解析哈希本应不同；模型和框架提交相同，MLVU任务文件单独验收。
            if current_env != ml['environments'][case['profile']] or any(
                    ml['framework'][k] != parent_spec['framework'][k] for k in ('commit', 'model_sha256')):
                raise ValueError('MLVU historical inference environment differs')
            for path, value in ml['task_files'].items():
                if sha256(path) != value: raise ValueError('Frozen MLVU task changed')
                sources[path] = value
            sources[str(MLVU/'protocol.json')] = sha256(MLVU/'protocol.json')
        for path, value in parent_spec['model_files'].items():
            if sha256(path) != value['sha256']: raise ValueError('Frozen model file changed: '+path)
        for name in QA_FILES:
            if name in parent_spec['code_sha256'] and sha256(ROOT/name) != parent_spec['code_sha256'][name]:
                raise ValueError('Historical inference code changed: '+name)
        conditions, slot = build_conditions(case, old['selection'], pool)
        data = task_row(case); video = Path(old['selection']['video']['path']); video_hash = sha256(video)
        if case['dataset'] == 'videomme': expected_video = parent_spec['video_hashes'][data['video_id']]
        else: expected_video = read_json(MLVU/'manifest.json')['video_hashes'][str(video)]
        if video_hash != expected_video: raise ValueError('Historical video bytes changed')
        if canonical_option(case, old['answer']) != case['historical_option']:
            raise ValueError('Fixed historical option differs')
        tools = make_tools(case['profile']); exports = {}; items = {}
        # 步骤2：精确解码官方预处理，O必须复现历史RGB、文本Token与处理器像素。
        for condition in ('O', 'T', 'C'):
            selection = conditions[condition]; images, rgb = FrameProvider(selection).decode()
            try:
                ident = cpu_input(case, tools, images, data)
                if condition == 'O' and (rgb != old['rgb_sha256'] or ident != observed_input(case['profile'], old['answer'])):
                    raise ValueError('Original actual input did not reproduce history')
                if condition != 'O':
                    unchanged = [i for i in range(len(rgb)) if i != slot]
                    if any(rgb[i] != items['O']['rgb_sha256'][i] for i in unchanged):
                        raise ValueError('An unchanged RGB slot differs')
                    for key in ('prompt', 'input_token_ids', 'video_grid_thw'):
                        if ident.get(key) != items['O']['input_identity'].get(key): raise ValueError('Text/grid changed')
                for i in (slot-1, slot, slot+1):
                    dest = root/'review'/condition/(str(i)+'_'+str(selection['selected_frames'][i]['source_pts'])+'.jpg')
                    dest.parent.mkdir(parents=True, exist_ok=True); images[i].save(dest, quality=92)
                    exports[str(dest)] = sha256(dest)
            finally:
                for im in images: im.close()
            item = dict(condition=condition, selection=selection, rgb_sha256=rgb, input_identity=ident,
                qwen_temporal_pairs=temporal_pairs(selection) if case['profile'] == 'qwen32' else [],
                no_effective_input_difference=condition != 'O' and ident == items['O']['input_identity'])
            path = root/'inputs'/(condition+'.json'); durable(path, item); exports[str(path)] = sha256(path); items[condition] = item
        # 步骤3：冻结调用顺序与预算，候选没有实际输入差异时只跳过、不临时替换。
        stat = video.stat(); spec = dict(case=case, row=data, changed_slot=slot, frame_count=len(conditions['O']['selected_frames']),
            sources=sources, exports=exports, framework=parent_spec['framework'], environment=current_env,
            model=parent_spec['model'], model_files=parent_spec['model_files'], max_real_calls=3, max_synthetic_calls=1,
            video=dict(path=str(video), sha256=video_hash, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
            code_sha256={n: sha256(ROOT/n) for n in FILES}, scope='Fixed targeted frame-group diagnostic, not algorithm accuracy', utc=utc())
        for name in FILES:
            dest = root/'code_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, dest)
        durable(root/'protocol.json', spec)
        durable(root/'status.json', dict(status='prepared', utc=utc()))
        return spec


def verify(root):
    """接口：冻结输入、来源、代码和环境一致才可恢复；检查不产生任何模型前向。"""
    from videoqa_full.prepare import environment
    from videoqa_lmms_bridge.common import framework_identity
    root = Path(root); spec = read_json(root/'protocol.json'); case = spec['case']
    if case not in cases() or spec['max_real_calls'] != 3 or spec['max_synthetic_calls'] != 1:
        raise ValueError('Fixed diagnostic population/budget changed')
    if environment(PYTHONS[case['profile']]) != spec['environment'] or framework_identity() != spec['framework']:
        raise ValueError('Diagnostic inference environment changed')
    for path, value in {**spec['sources'], **spec['exports']}.items():
        if sha256(path) != value: raise ValueError('Frozen input/source changed: '+path)
    for name, value in spec['code_sha256'].items():
        if sha256(ROOT/name) != value: raise ValueError('Frozen code changed: '+name)
    for path, value in {**spec['model_files'], spec['video']['path']: spec['video']}.items():
        st = Path(path).stat()
        if (st.st_size, st.st_mtime_ns) != (value['bytes'], value['mtime_ns']): raise ValueError('Model/video identity changed')
    return spec


def task_backend(case, backend):
    """接口：每组独立进程构建各自任务适配器，冻结观测器不改字节。"""
    if case['dataset'] == 'mlvu':
        from videoqa_mlvu.backends import make_adapter, observer_module
        return make_adapter(case['profile'], backend), observer_module(case['profile'])
    if case['profile'] == 'qwen32':
        from videoqa_budget32.qwen_adapter import FrozenInputQwen
        from videoqa_budget32 import observe
        return FrozenInputQwen(backend), observe
    from videoqa_generalization.llava_budget8 import adapter_for
    from videoqa_generalization import llava_observe
    return adapter_for(backend), llava_observe


def observe_answer(case, observer, backend, adapter, images, selection, data, state, rawpath, identity):
    """接口：正式生成真正经过固定框架，返回立即落盘的原文可用于仅恢复后处理。"""
    if case['dataset'] == 'mlvu':
        from videoqa_mlvu.backends import observe
        return observe(case['profile'], observer, backend, adapter, images, selection, data, state, rawpath, identity)
    if case['profile'] == 'qwen32': return observer.observe(backend, adapter, images, data, state, rawpath, identity)
    value = observer.observe(backend, adapter, images, selection, data, state, rawpath, identity)
    return dict(value['answer'], observed=value['observed'])


def restore_answer(case, raw, data):
    """接口：完整原始返回只重放停止清理和任务计分，禁止再次生成。"""
    if case['dataset'] == 'mlvu':
        from videoqa_mlvu.backends import answer_from_raw
        return answer_from_raw(case['profile'], raw, data)
    if case['profile'] == 'qwen32':
        from videoqa_budget32.observe import result_from_raw
        return result_from_raw(raw, data)
    from videoqa_generalization.llava_observe import result_from_raw
    value = result_from_raw(raw, data)
    return dict(value['answer'], observed=value['observed'])


def synthetic_input(case):
    """接口：每组仅一次8/32帧合成验收，独立记账，不读取研究题标准答案。"""
    from PIL import Image
    from videoqa_lmms_bridge.bridge import synthetic_row
    n = 32 if case['profile'] == 'qwen32' else 8; data = synthetic_row(n)
    if case['dataset'] == 'mlvu':
        data.update(qa_question='Which color is visible?\n(A) Red\n(B) Blue\n(C) Green\n(D) Gray\n', task_type='plotQA')
    images = [Image.new('RGB', (896, 672), (20, 40+i*3, 150)) for i in range(n)]
    selection = dict(video=dict(path='synthetic.mp4', duration_seconds=float(n)),
        selected_frames=[dict(timestamp_seconds=float(i)) for i in range(n)])
    return images, selection, data


def validate_answer(spec, item, answer, data):
    """接口：复算任务计分并核验真实像素/Token、视频配对、生成配置和视觉预算。"""
    case = spec['case']; o = answer['observed']
    if observed_input(case['profile'], answer) != item['input_identity']:
        raise ValueError('Actual model input differs from CPU freeze')
    if o['generate_calls'] != 1 or o['prefill_calls'] != 1 or not 1 <= len(o['generated_token_ids']) <= 16:
        raise ValueError('Actual generate/token count differs')
    if case['profile'] == 'qwen32':
        g = o['generation_kwargs']
        if g['max_new_tokens'] != 16 or g['do_sample'] is not False or o['encoding_fps'] != 2.0 or o['source_frames'] != spec['frame_count']:
            raise ValueError('Qwen generation/video protocol differs')
        if o['encoded_frames'] != spec['frame_count'] or o['internal_padding_frames'] != 0:
            raise ValueError('Unexpected temporal padding')
    elif o['visual_tokens'] != 210*spec['frame_count'] or o['max_new_tokens'] != 16:
        raise ValueError('LLaVA visual/generation protocol differs')
    if case['dataset'] == 'mlvu':
        from videoqa_mlvu.backends import metric
        score = metric(data, [answer['raw_output']])
    else:
        from videoqa_lmms_bridge.common import load_task, task_doc
        score = load_task().videomme_process_results(task_doc(data), [answer['raw_output']])['videomme_perception_score']
    if score != answer['framework_metric'] or (score['pred_answer'] or None) != (answer['parsed_answer'] or None):
        raise ValueError('Native framework scoring differs')


def persist(root, condition, answer):
    """接口：只保存新诊断结果，保留原生与句点归一化计分以及原始输入引用。"""
    root = Path(root); spec = read_json(root/'protocol.json'); itempath = root/'inputs'/(condition+'.json')
    validate_answer(spec, read_json(itempath), answer, spec['row'])
    option = canonical_option(spec['case'], answer); gold = 'ABCD'[spec['row']['answer_index']]
    dest = root/'results'/(condition+'.json')
    if not dest.exists(): durable(dest, dict(condition=condition, answer=answer, option=option,
        native_correct=answer['parsed_answer'] == gold, correct=option == gold,
        input_sha256=sha256(itempath), utc=utc()))


def recover_returns(root):
    """接口：恢复完成原文及其后处理；未返回生成拒绝重发，部分组拒绝跨进程补跑。"""
    root = Path(root); spec = verify(root); case = spec['case']; fp = sha256(root/'protocol.json')
    for path in (root/'calls').glob('*/*.json'):
        call = read_json(path); condition = call['method']; qid = path.stem
        if condition not in ('O', 'T', 'C', 'synthetic') or call['protocol_sha256'] != fp or call['gpu'] != case['gpu']:
            raise ValueError('Call identity changed')
        rawpath = root/'raw_generations'/condition/(qid+'.json')
        if not rawpath.exists(): raise RuntimeError('Uncertain generation, no automatic retry')
        raw = read_json(rawpath)
        if raw['identity'] != dict(question_id=qid, method=condition, protocol_sha256=fp, gpu=case['gpu']) or raw['generation_state'] != {'started': True, 'returned': True}:
            raise RuntimeError('Raw return identity incomplete')
        if call['status'] != 'completed':
            data = spec['row']
            if condition == 'synthetic':
                images, _, data = synthetic_input(case)
                for im in images: im.close()
            answer = restore_answer(case, raw, data)
            call.update(status='completed', answer=answer, generation_state=raw['generation_state'], recovered_postprocessing=True)
            durable(path, call)
        if condition != 'synthetic': persist(root, condition, call['answer'])
    return finish_case(root, allow_incomplete=True)


def finish_case(root, allow_incomplete=False):
    """接口：O复现门槛、跳过及真实/合成调用闭合，同组必须是同一PID和物理卡。"""
    root = Path(root); spec = read_json(root/'protocol.json'); case = spec['case']; results = {}
    calls = [read_json(p) for p in (root/'calls').glob('*/*.json')]
    if len(calls) > 4 or sum(c['method'] != 'synthetic' for c in calls) > 3:
        raise ValueError('Per-case call budget exceeded')
    if calls and (len({c['pid'] for c in calls}) != 1 or {c['gpu'] for c in calls} != {case['gpu']}):
        raise ValueError('Diagnostic crossed process/GPU boundaries')
    # 步骤1：包括合成在内的每次返回，都必须有同身份原文；未返回调用不能当作跳过。
    for call in calls:
        qid, condition = call['question_id'], call['method']
        if qid != ('warmup' if condition == 'synthetic' else case['question_id']):
            raise ValueError('Call task identity changed')
        rawpath = root/'raw_generations'/condition/(qid+'.json')
        if not rawpath.exists():
            if allow_incomplete: return None
            raise ValueError('Raw generation missing')
        raw = read_json(rawpath)
        expected_identity = dict(question_id=qid, method=condition, protocol_sha256=sha256(root/'protocol.json'), gpu=case['gpu'])
        if call['protocol_sha256'] != expected_identity['protocol_sha256'] or raw['identity'] != expected_identity:
            raise ValueError('Call/raw protocol identity changed')
        if raw['generation_state'] != {'started': True, 'returned': True} or call['status'] != 'completed':
            if allow_incomplete: return None
            raise RuntimeError('Uncertain generation, no retry')
    for condition in ('O', 'T', 'C'):
        path = root/'results'/(condition+'.json')
        if not path.exists(): continue
        r = read_json(path); itempath = root/'inputs'/(condition+'.json'); item = read_json(itempath)
        call = read_json(root/'calls'/condition/(case['question_id']+'.json'))
        if r['input_sha256'] != sha256(itempath) or call['status'] != 'completed' or call['answer'] != r['answer']:
            raise ValueError('Result/call/input differs')
        validate_answer(spec, item, r['answer'], spec['row']); results[condition] = r
    if 'O' not in results:
        if allow_incomplete: return None
        raise ValueError('Original replay missing')
    matched = results['O']['option'] == case['historical_option']; skipped = {}
    for condition in ('T', 'C'):
        if not matched: skipped[condition] = 'original_did_not_reproduce_history'
        elif read_json(root/'inputs'/(condition+'.json'))['no_effective_input_difference']:
            skipped[condition] = 'no_effective_model_input_difference'
    if any(c in results for c in skipped): raise ValueError('Skipped condition was generated')
    expected = {'O', 'T', 'C'}-set(skipped)
    if set(results) != expected:
        if allow_incomplete: return None
        raise ValueError('Diagnostic condition incomplete')
    expected_calls = expected | {'synthetic'}
    if {c['method'] for c in calls} != expected_calls or any(c['status'] != 'completed' for c in calls):
        raise ValueError('Physical calls do not close with skips')
    files = {str(p.relative_to(root)): sha256(p) for folder in ('calls', 'results', 'raw_generations') for p in (root/folder).rglob('*.json')}
    previous = root/'completion_audit.json'
    if previous.exists() and read_json(previous)['artifacts'] != files: raise ValueError('Completion recovery changed generated artifacts')
    value = dict(status='completed', case=case, historical_replay_matches=matched, skipped=skipped,
        results={k: dict(option=r['option'], correct=r['correct'], native_correct=r['native_correct'], raw_output=r['answer']['raw_output']) for k, r in results.items()},
        real_calls=len(expected), synthetic_calls=1, artifacts=files, additional_generation_on_recovery=0, utc=utc())
    durable(previous, value); durable(root/'status.json', value)
    return value


def worker(root):
    """接口：独立组同进程先O，再按冻结次序T/C；错误或格式失败不重试。"""
    root = Path(root); offline_environment(); spec = verify(root); case = spec['case']
    with exclusive(root/'worker.lock'):
        recovered = recover_returns(root)
        if recovered: return recovered
        if list((root/'calls').glob('*/*.json')):
            raise RuntimeError('Partial group process ended; cross-process continuation prohibited')
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(2027)
        start = time.perf_counter()
        if case['profile'] == 'qwen32':
            from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
            backend = QwenFA2VideoBackend()
        else:
            from videoqa_runtime.llava_backend import LlavaBackend
            backend = LlavaBackend()
        adapter, observer = task_backend(case, backend)
        durable(root/'load_report.json', dict(pid=os.getpid(), gpu=case['gpu'], seconds=time.perf_counter()-start, report=backend.load_report))
        fp = sha256(root/'protocol.json')
        def invoke(condition, images, selection, data, qid):
            identity = dict(question_id=qid, method=condition, protocol_sha256=fp, gpu=case['gpu'])
            def generate(state):
                return observe_answer(case, observer, backend, adapter, images, selection, data, state,
                    root/'raw_generations'/condition/(qid+'.json'), identity)
            return invoke_once(root, qid, condition, fp, case['gpu'], generate)[0]
        # 步骤1：合成与真实调用各自独立ID，每组只执行一次同预算合成输入。
        images, selection, data = synthetic_input(case)
        try: invoke('synthetic', images, selection, data, 'warmup')
        finally:
            for im in images: im.close()
        # 步骤2：运行前重复核验实际RGB和CPU输入；不允许接口重新采样。
        for condition in case['order']:
            if condition != 'O':
                original = read_json(root/'results/O.json')
                if original['option'] != case['historical_option']: break
            item = read_json(root/'inputs'/(condition+'.json'))
            if item['no_effective_input_difference']: continue
            durable(root/'status.json', dict(status='running', condition=condition, pid=os.getpid(), gpu=case['gpu'], utc=utc()))
            images, rgb = FrameProvider(item['selection']).decode()
            try:
                tools = (backend.processor, backend.tokenizer if case['profile'] == 'llava8' else None)
                if rgb != item['rgb_sha256'] or cpu_input(case, tools, images, spec['row']) != item['input_identity']:
                    raise ValueError('Frozen actual input changed before generate')
                answer = invoke(condition, images, item['selection'], spec['row'], case['question_id'])
            finally:
                for im in images: im.close()
            persist(root, condition, answer)
        # 步骤3：结果/原文/账本在退出前闭合；O不一致仅停止本组，不影响其他独立组。
        return finish_case(root)


def summarize(root):
    """接口：诊断按组报告，不计算算法精度，不自动启动统一候选或额外问答。"""
    root = Path(root); summaries = [finish_case(root/c['id']) for c in cases()]
    fixes = [s['case']['id'] for s in summaries if s['case']['id'].startswith('D') and s['historical_replay_matches']
        and set(s['results']) == {'O', 'T', 'C'} and not s['results']['O']['correct']
        and s['results']['T']['correct'] and not s['results']['C']['correct']]
    controls = [s for s in summaries if s['case']['id'].startswith('P')]
    valid_controls = all(s['historical_replay_matches'] and 'T' in s['results'] for s in controls)
    no_harm = valid_controls and all(s['results']['T']['correct'] for s in controls)
    value = dict(status='completed', groups=summaries, target_only_fixes=fixes,
        both_controls_valid=valid_controls, controls_without_target_regression=no_harm,
        automatic_design_resource_gate_passed=len(fixes) >= 2 and no_harm,
        real_calls=sum(s['real_calls'] for s in summaries), synthetic_calls=sum(s['synthetic_calls'] for s in summaries),
        additional_generation_on_recovery=0, next_step='Review diagnostic evidence; no automatic new experiment', utc=utc())
    if value['real_calls'] > 15 or value['synthetic_calls'] > 5: raise ValueError('Global budget exceeded')
    durable(root/'summary.json', value)
    lines = ['# 单帧邻接上下文诊断', '', '固定定向诊断，不计作算法正确率。O/T/C分别为原输入、同事件邻接替换、原场景近邻替换。', '',
        '| 组 | 配置／题目 | O | T | C | O复现历史 |', '|---|---|---|---|---|---|']
    for s in summaries:
        def cell(k):
            r = s['results'].get(k)
            return (str(r['option'])+(' ✓' if r['correct'] else ' ✗')) if r else '跳过：'+s['skipped'].get(k, '')
        c = s['case']; lines.append(f"| {c['id']} | {c['profile']}／{c['question_id']} | {cell('O')} | {cell('T')} | {cell('C')} | {s['historical_replay_matches']} |")
    lines += ['', f"真实生成 {value['real_calls']}/15；合成 {value['synthetic_calls']}/5。T独有纠正：{fixes}。", '',
        f"两个正确对照有效且T无退化：{no_harm}；自动候选设计投入门槛：{value['automatic_design_resource_gate_passed']}。", '',
        'Qwen同事件替换同时改变关键帧的时间patch编码伙伴，不能单独归因于语义干扰删除，也不能据此推广为通用配对。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    # 步骤2：可点击页面只呈现冻结输入和结果，不写回候选、不添加模型提示。
    content = ['<!doctype html><meta charset="utf-8"><title>单帧邻接上下文诊断</title>',
        '<style>body{font:16px system-ui;max-width:1250px;margin:30px auto}img{width:30%;vertical-align:top}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}summary{cursor:pointer}pre{white-space:pre-wrap}</style>',
        '<h1>单帧邻接上下文诊断</h1><p>O原输入／T同事件邻接／C原场景近邻。此为定向诊断，不是算法精度。</p>',
        '<p><a href="summary.json">完整审计</a> · <a href="report.md">结果表</a></p>']
    for s in summaries:
        c = s['case']; run = root/c['id']; spec = read_json(run/'protocol.json')
        content += [f"<h2>{c['id']} · {c['profile']} · {c['question_id']}</h2>",
            '<pre>'+html.escape(spec['row']['question']+'\n'+'\n'.join(spec['row']['options']))+'</pre>',
            f"<p>槽位（从0计）：{spec['changed_slot']}；历史选项：{c['historical_option']}；复现：{s['historical_replay_matches']}</p>"]
        for condition in ('O', 'T', 'C'):
            result = s['results'].get(condition)
            label = str(result) if result else s['skipped'].get(condition, '未执行')
            content += [f'<h3>{condition}：{html.escape(label)}</h3>']
            for image in sorted((run/'review'/condition).glob('*.jpg'), key=lambda p:int(p.stem.split('_')[0])):
                rel = image.relative_to(root); content.append(f'<a href="{rel}"><img src="{rel}"></a>')
            item = read_json(run/'inputs'/(condition+'.json'))
            content += [f'<p><a href="{c["id"]}/inputs/{condition}.json">冻结实际输入</a></p>', '<details><summary>全部源帧及Qwen配对</summary>',
                '<pre>'+html.escape(str([(i,f['source_pts'],f['timestamp_seconds']) for i,f in enumerate(item['selection']['selected_frames'])]))+'</pre>',
                '<pre>'+html.escape(str(item['qwen_temporal_pairs']))+'</pre></details>']
    content.append('<p>Qwen变化包含时间patch伙伴变化；不能单独归因为删除语义干扰。后续实验不自动启动。</p>')
    (root/'index.html').write_text('\n'.join(content))
    return value


def coordinate(root):
    """接口：后台最多两组并行，等待固定历史物理卡；失败暂停，不重启不确定生成。"""
    from videoqa_runtime.baseline_run import available_gpus
    root = Path(root)
    if not (root/'prepared.json').exists(): raise ValueError('Freeze all 15 CPU inputs before GPU execution')
    prepared = read_json(root/'prepared.json')
    if prepared['protocols'] != {c['id']:sha256(root/c['id']/'protocol.json') for c in cases()}:
        raise ValueError('Published CPU protocols changed')
    with exclusive(root/'coordinator.lock'):
        active = {}; pending = list(cases())
        durable(root/'launch.json', dict(pid=os.getpid(), sid=os.getsid(0), max_concurrent_groups=2, utc=utc()))
        # 步骤1：已完成只审计；部分组只恢复返回后的处理，不能换进程继续配对生成。
        while pending or active:
            for key, (process, log, case) in list(active.items()):
                if process.poll() is None: continue
                log.close(); del active[key]
                if process.returncode:
                    durable(root/'status.json', dict(status='paused', case=key, returncode=process.returncode,
                        reason='Worker failed; inspect ledger, no automatic retry', utc=utc()))
                    # 已进入生成的其他独立组允许完成；不终止正在生成的调用，不启动后续组。
                    for child, handle, _ in active.values(): child.wait(); handle.close()
                    raise RuntimeError('Diagnostic pipeline paused at '+key)
            free, _ = available_gpus([c['gpu'] for c in pending]) if pending else ([], {})
            for case in list(pending):
                if len(active) >= 2: break
                if case['gpu'] not in free: continue
                run = root/case['id']; pending.remove(case)
                uuid = subprocess.check_output(['nvidia-smi', '-i', case['gpu'], '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
                assigned = run/'gpu_assignment.json'; identity = dict(gpu=case['gpu'], uuid=uuid)
                if assigned.exists() and read_json(assigned) != identity: raise ValueError('Physical GPU identity changed')
                if not assigned.exists(): durable(assigned, identity)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=case['gpu'], OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                log = (run/'worker.log').open('a', buffering=1)
                process = subprocess.Popen([PYTHONS[case['profile']], str(ROOT/'scripts/run_frame_group_diagnostic.py'),
                    '--run-id', root.name, '--stage', 'worker', '--case', case['id']],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                active[case['id']] = (process, log, case)
            # 步骤2：心跳反映真实子进程和调用数，不依赖外部会话保持存活。
            durable(root/'heartbeat.json', dict(pid=os.getpid(), sid=os.getsid(0), active={k:p.pid for k,(p,_,_) in active.items()},
                waiting=[c['id'] for c in pending], real_calls=len([p for p in root.glob('*/calls/*/*.json') if p.parent.name != 'synthetic']), utc=utc()))
            if pending or active: time.sleep(5)
        return summarize(root)
