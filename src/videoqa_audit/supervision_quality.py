"""训练监督的画面敏感性诊断：固定120视频、五条件、至多600次教师生成。"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
from PIL import Image

from videoqa_runtime.common import ROOT, read_json, sha256, offline_environment
from videoqa_full.state import durable, digest, exclusive, invoke_once, utc
from videoqa_set_value.data import METHODS
from videoqa_set_value.data_acquisition import stable_key

SOURCE = ROOT/'outputs/set_value/set_value_learning__nextqa800__llava8__20261019_r2'
QA_PYTHON = '/App/conda/envs/explore_video_qa_lmms/bin/python'
CONDITIONS = ('O', 'G', 'W', 'L', 'R')
PUBLIC = ('question_id', 'split', 'video_id', 'video_sha256', 'video_path', 'index_path',
          'question', 'options', 'task_type')
_TOOLS = None


def public_training_rows(rows):
    """接口：只投影原600训练题；答案、历史预测及官方源行不会进入输入构造器。"""
    selected = [{k: r[k] for k in PUBLIC} for r in rows if r['split'] == 'train']
    if len(selected) != 600 or len({r['video_id'] for r in selected}) != 300:
        raise ValueError('Expected original 600 training questions / 300 videos')
    return selected


def select_panel(rows, durations, size=120):
    """接口：公开视频身份和时长→题目及供体；哈希选样不接收任何监督标签。"""
    groups = defaultdict(list)
    for row in rows:
        if row['split'] != 'train' or set(row)-set(PUBLIC):
            raise ValueError('Only public training fields are allowed')
        groups[row['video_id']].append(row)
    if len(groups) < size: raise ValueError('Not enough distinct training videos')
    # 步骤1：先按视频哈希取样，再在该视频内按题号哈希取一道题。
    videos = sorted(groups, key=lambda v: stable_key('supervision_quality_v1/video', v))[:size]
    representative = lambda v: min(groups[v], key=lambda r: stable_key('supervision_quality_v1/question', r['question_id']))
    result = []
    for ordinal, vid in enumerate(videos):
        row = representative(vid)
        eligible = [v for v in groups if v != vid and representative(v)['video_sha256'] != row['video_sha256']]
        if not eligible: raise ValueError('No distinct-content donor')
        # 步骤2：时长最近的另一内容作为供体；不读取供体问题的对错。
        donor = min(eligible, key=lambda v: (abs(durations[v]-durations[vid]),
                                            stable_key('supervision_quality_v1/donor', v)))
        result.append(dict(ordinal=ordinal, row=row, donor=representative(donor),
                           duration_seconds=durations[vid], donor_duration_seconds=durations[donor]))
    return result


def candidate_groups(candidates, scores, question_id):
    """接口：原粗候选与旧分数→固定局部8帧和哈希分散8帧；不生成新采样目标。"""
    if len(candidates) < 8 or len(candidates) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid frozen coarse pool')
    order = sorted(range(len(candidates)), key=lambda i: (candidates[i]['source_pts'], candidates[i]['candidate_index']))
    identities = [(candidates[i]['source_pts'], candidates[i]['source_frame_index']) for i in order]
    if len({p for p, _ in identities}) != len(order) or len({n for _, n in identities}) != len(order):
        raise ValueError('Duplicate coarse sources')
    # 步骤1：最高分稳定排序；八帧偶数块固定为前3/后4，首尾仅裁剪起点。
    best = min(order, key=lambda i: (-scores[i], candidates[i]['source_pts'], candidates[i]['candidate_index']))
    start = max(0, min(order.index(best)-3, len(order)-8))
    local = order[start:start+8]
    # 步骤2：题号及源身份固定随机顺序，取八张后恢复真实时间顺序。
    random = sorted(order, key=lambda i: stable_key('supervision_quality_v1/random',
                    question_id+'/'+str(candidates[i]['source_pts'])+'/'+str(candidates[i]['source_frame_index'])))[:8]
    random.sort(key=lambda i: (candidates[i]['source_pts'], candidates[i]['candidate_index']))
    return dict(L=[copy.deepcopy(candidates[i]) for i in local],
                R=[copy.deepcopy(candidates[i]) for i in random])


def source_order(record):
    """接口：实际源身份用于同题分组；灰图显式标为人工槽位，不能伪装为视频PTS。"""
    if record.get('artificial'): return [['artificial_gray', i] for i in range(8)]
    return [[f['source_pts'], f['source_frame_index']] for f in record['frames']]


def condition_key(record, identity):
    """接口：源顺序、RGB和完整官方输入相同才能共享本轮返回，不跨题共享。"""
    return digest(dict(source_order=source_order(record), rgb=[f['rgb_sha256'] for f in record['frames']], input=identity))


def group_conditions(entries):
    """接口：固定O/G/W/L/R优先级建立一次生成到多个条件的映射。"""
    groups = []
    for condition in CONDITIONS:
        key = entries[condition]['input_key']
        existing = next((g for g in groups if g['input_key'] == key), None)
        if existing is None: groups.append(dict(input_key=key, representative=condition, conditions=[condition]))
        else: existing['conditions'].append(condition)
    return groups


def ordered_groups(groups, ordinal):
    """接口：O必先复测；其他条件循环轮换，以首次出现确定执行组次序。"""
    rest = CONDITIONS[1:]; shift = ordinal % len(rest); order = ['O', *rest[shift:], *rest[:shift]]
    output = []
    for condition in order:
        group = next(g for g in groups if condition in g['conditions'])
        if group not in output: output.append(group)
    return output


def historical_reproduced(raw, historical_option):
    """接口：按固定完整选项解析判断复现，解析失败不算复现。"""
    from videoqa_set_value.task import strict_option
    parsed = strict_option(raw)
    return parsed is not None and parsed == historical_option


def _init_cpu():
    """接口：spawn工作进程仅加载CPU Tokenizer和官方处理器，不加载任何模型权重。"""
    global _TOOLS
    import torch
    from videoqa_set_value.task import cpu_tools
    offline_environment(); torch.set_num_threads(2); torch.set_num_interop_threads(1)
    _TOOLS = cpu_tools()


def _load_images(record):
    """接口：核对PNG字节及RGB；真实源映射在准备时已由顺序解码器双验。"""
    images = []
    try:
        for frame in record['frames']:
            if sha256(frame['image']) != frame['image_sha256']: raise ValueError('Frozen PNG changed')
            image = Image.open(frame['image']).convert('RGB'); images.append(image)
            if hashlib.sha256(np.ascontiguousarray(np.asarray(image)).tobytes()).hexdigest() != frame['rgb_sha256']:
                raise ValueError('Frozen source RGB changed')
        return images
    except BaseException:
        for image in images: image.close()
        raise


def _with_image_hashes(record):
    """接口：保留原源帧属性，旁路绑定实际图片文件，不改写历史记录。"""
    value = copy.deepcopy(record)
    for f in value['frames']: f['image_sha256'] = sha256(f['image'])
    return value


def prepare_case(job):
    """接口：公开面板项→五份CPU冻结输入；函数不接收标准答案或历史预测。"""
    from videoqa_set_value.media import SequentialFrameProvider
    from videoqa_set_value.task import cpu_identity
    from videoqa_budget32.frames import validate_frames
    root, case = Path(job[0]), job[1]; row = case['row']; q = row['question_id']
    path = root/'inputs'/(q+'.json')
    if path.exists(): return read_json(path)
    # 步骤1：O和W均读取原均匀源帧；新L/R只使用原始粗候选前缀。
    original = read_json(SOURCE/'selections'/METHODS[0]/(q+'.json'))
    donor = read_json(SOURCE/'selections'/METHODS[0]/(case['donor']['question_id']+'.json'))
    pool = read_json(SOURCE/'pools'/(q+'.json')); coarse = pool['candidates'][:pool['initial_count']]
    chosen = candidate_groups(coarse, pool['initial_scores'], q)
    cached = {}
    for method in METHODS:
        record = read_json(SOURCE/'selections'/method/(q+'.json'))
        for f in record['frames']: cached[(f['source_pts'], f['source_frame_index'])] = f
    missing = {(f['source_pts'], f['source_frame_index']):f for frames in chosen.values() for f in frames
               if (f['source_pts'], f['source_frame_index']) not in cached}
    began = time.monotonic()
    if missing:
        frames = [missing[k] for k in sorted(missing)]
        images, hashes = SequentialFrameProvider(dict(video=pool['video'], selected_frames=frames)).decode()
        try:
            for f, image, rgb in zip(frames, images, hashes):
                target = root/'rgb'/q/(str(f['source_pts'])+'_'+str(f['source_frame_index'])+'.png')
                target.parent.mkdir(parents=True, exist_ok=True); image.save(target)
                cached[(f['source_pts'], f['source_frame_index'])] = dict(f, image=str(target), rgb_sha256=rgb)
        finally:
            for image in images: image.close()
    records = dict(O=_with_image_hashes(original), W=_with_image_hashes(donor))
    gray = root/'gray.png'; gray_sha = sha256(gray)
    with Image.open(gray) as image:
        gray_rgb = hashlib.sha256(np.ascontiguousarray(np.asarray(image.convert('RGB'))).tobytes()).hexdigest()
    records['G'] = dict(artificial=True, selection=dict(video=dict(path='diagnostic-gray.mp4', duration_seconds=8.),
                    selected_frames=[dict(timestamp_seconds=float(i)) for i in range(8)]),
                    frames=[dict(image=str(gray), image_sha256=gray_sha, rgb_sha256=gray_rgb, artificial_slot=i) for i in range(8)])
    for condition, frames in chosen.items():
        selection = dict(video=pool['video'], selected_frames=frames); validate_frames(selection, 8)
        records[condition] = _with_image_hashes(dict(selection=selection,
                            frames=[dict(f, image=cached[(f['source_pts'], f['source_frame_index'])]['image'],
                                         rgb_sha256=cached[(f['source_pts'], f['source_frame_index'])]['rgb_sha256']) for f in frames]))
    # 步骤2：逐条件执行官方CPU处理，文字来自本题；W从不使用供体的任务文字。
    entries = {}
    for condition in CONDITIONS:
        images = _load_images(records[condition])
        try: identity = cpu_identity(_TOOLS, images, row)
        finally:
            for image in images: image.close()
        if identity['visual_tokens'] != 1680 or identity['source_frames'] != 8: raise ValueError('Expected K8 official input')
        entries[condition] = dict(record=records[condition], identity=identity,
                                  source_order=source_order(records[condition]), input_key=condition_key(records[condition], identity))
    old = read_json(SOURCE/'inputs'/(q+'.json'))['methods'][METHODS[0]]
    if (entries['O']['identity'] != old['identity'] or entries['O']['source_order'] != old['source_order']
            or entries['O']['input_key'] != old['input_key']):
        raise ValueError('Original input does not reproduce frozen CPU identity')
    if len({e['identity']['prompt'] for e in entries.values()}) != 1: raise ValueError('Condition text differs')
    value = dict(question_id=q, ordinal=case['ordinal'], entries=entries, groups=group_conditions(entries),
                 input_builder_receives_labels=False, preparation_seconds=time.monotonic()-began)
    durable(path, value); return value


def verify(root, check_inputs=False):
    """接口：正式运行/恢复核验源码、依赖、模型统计及已冻结来源，完成审计另核验全哈希。"""
    root = Path(root); p = read_json(root/'protocol.json')
    for path, expected in p['code'].items():
        if sha256(ROOT/path) != expected: raise ValueError('Diagnostic code changed: '+path)
    for path, expected in p['sources'].items():
        if sha256(path) != expected: raise ValueError('Diagnostic source changed: '+path)
    for path, value in p['models'].items():
        st = Path(path).stat()
        if (st.st_size, st.st_mtime_ns) != (value['bytes'], value['mtime_ns']): raise ValueError('Frozen model changed')
    if check_inputs:
        ready = read_json(root/'inputs_complete.json')
        stage = read_json(root/'qa/protocol.json')
        if stage['parent_protocol_sha256'] != sha256(root/'protocol.json') or stage['inputs_sha256'] != sha256(root/'inputs_complete.json'):
            raise ValueError('Question-answer phase fingerprint changed')
        for q, expected in ready['exports'].items():
            if sha256(root/'inputs'/(q+'.json')) != expected: raise ValueError('Frozen condition input changed')
            plan = read_json(root/'inputs'/(q+'.json'))
            for e in plan['entries'].values():
                for f in e['record']['frames']:
                    if sha256(f['image']) != f['image_sha256']: raise ValueError('Input PNG changed')
        if sha256(root/'panel.json') != ready['panel_sha256'] or sha256(root/'evaluation.json') != ready['evaluation_sha256']:
            raise ValueError('Frozen case/evaluation identity changed')
    return p


def prepare(root):
    """接口：零模型前向冻结120题/供体/实际输入/额度；与原200题及训练器隔离。"""
    from videoqa_set_value.runtime import verify as verify_parent
    from videoqa_full.prepare import environment
    from videoqa_audit.set_value_trainability import sealed_artifacts
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with exclusive(root/'prepare.lock'):
        if (root/'inputs_complete.json').exists(): verify(root, True); return read_json(root/'inputs_complete.json')
        began = time.monotonic(); parent = verify_parent(SOURCE); protection = sealed_artifacts(SOURCE)
        if environment(QA_PYTHON) != parent['environments']['llava']: raise ValueError('QA environment differs from parent')
        all_rows = read_json(SOURCE/'manifest.json')['rows']; public = public_training_rows(all_rows)
        durations = {r['video_id']:read_json(r['index_path'])['video']['duration_seconds'] for r in public}
        cases = select_panel(public, durations)
        # 步骤1：无答案选样和供体映射先落盘；评价标签仅在独立文件中后置关联。
        panel = dict(cases=cases, sample_uses_outcomes=False, question_count=120, video_count=120)
        panel_path = root/'panel.json'
        if panel_path.exists() and read_json(panel_path) != panel: raise ValueError('Panel selection changed')
        if not panel_path.exists(): durable(panel_path, panel)
        rows = {r['question_id']:r for r in all_rows if r['split']=='train'}
        evaluations = {}; states = Counter(); sources = {str(SOURCE/'protocol.json'):sha256(SOURCE/'protocol.json'),
                   str(SOURCE/'manifest.json'):sha256(SOURCE/'manifest.json')}
        raw_audit = read_json(SOURCE/'qa/train/completion_audit.json')
        for c in cases:
            q = c['row']['question_id']; historical = {}
            for method in METHODS:
                path = SOURCE/'qa/train/results'/method/(q+'.json')
                expected = raw_audit['artifacts']['results/'+method+'/'+q+'.json']
                if sha256(path) != expected: raise ValueError('Original training result changed')
                value = read_json(path); sources[str(path)] = expected
                historical[method] = dict(correct=bool(value['correct']), parsed_answer=value['answer']['parsed_answer'],
                    raw_output=value['answer']['raw_output'], input_key=value['input_key'], source_path=str(path), source_sha256=expected)
            labels = [v['correct'] for v in historical.values()]
            state = 'all_correct' if all(labels) else 'all_wrong' if not any(labels) else 'mixed'; states[state]+=1
            original = read_json(SOURCE/'qa/train/results'/METHODS[0]/(q+'.json'))
            evaluations[q] = dict(answer_index=rows[q]['answer_index'], historical=historical,
                                  historical_state=state, gpu=original['gpu'])
            for item in (c['row'], c['donor']):
                for path in (SOURCE/'selections'/METHODS[0]/(item['question_id']+'.json'),Path(item['index_path'])):
                    sources[str(path)] = sha256(path)
            for path in [SOURCE/'inputs'/(q+'.json'), SOURCE/'pools'/(q+'.json'),
                         *(SOURCE/'selections'/m/(q+'.json') for m in METHODS)]: sources[str(path)] = sha256(path)
        if states != Counter(all_correct=99, all_wrong=12, mixed=9): raise ValueError('Fixed panel descriptive counts differ')
        evaluation_path = root/'evaluation.json'
        if evaluation_path.exists() and read_json(evaluation_path) != evaluations: raise ValueError('Frozen evaluation changed')
        if not evaluation_path.exists(): durable(evaluation_path, evaluations)
        code = dict(parent['code'])
        for name in ('src/videoqa_audit/supervision_quality.py','src/videoqa_audit/supervision_quality_report.py',
                     'scripts/run_supervision_quality_diagnostic.py'): code[name] = sha256(ROOT/name)
        for name in code:
            target = root/'code_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name,target)
        value = dict(task='supervision_quality_v1', source_run=str(SOURCE), code=code, sources=sources,
                     models=parent['models'], environments=parent['environments'], original_protection=protection,
                     maximum_real_calls=600, maximum_synthetic_calls=8, teacher=parent['teacher'],
                     conditions=CONDITIONS, training_updates=0,new_blip_calls=0,new_refinement_frames=0,new_feature_calls=0,
                     original_validation_sealed=True, utc=utc())
        if (root/'protocol.json').exists():
            old=read_json(root/'protocol.json'); value['utc']=old['utc']
            if old != dict(value, conditions=list(CONDITIONS)): raise ValueError('Preparation protocol changed')
        else: durable(root/'protocol.json',value)
        gray = root/'gray.png'
        if not gray.exists(): Image.new('RGB',(384,384),(128,128,128)).save(gray)
        # 步骤2：spawn CPU进程完成精确取帧及官方像素/提示身份，未加载教师。
        exports = {}; budget = 0
        with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'), initializer=_init_cpu) as pool:
            for item in pool.map(prepare_case, [(str(root), c) for c in cases]):
                q=item['question_id']; exports[q]=sha256(root/'inputs'/(q+'.json')); budget+=len(item['groups'])
                durable(root/'preparation_progress.json',dict(completed=len(exports),total=120,utc=utc()))
        if len(exports)!=120 or not 120<=budget<=600: raise ValueError('Frozen call plan does not close')
        gpu_uuids={g:subprocess.check_output(['nvidia-smi','-i',g,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
                   for g in sorted({e['gpu'] for e in evaluations.values()})}
        ready=dict(questions=120,condition_results=600,exports=exports,maximum_real_calls=600,
                   unique_input_call_ceiling=budget,maximum_synthetic_calls=8,historical_states=dict(states),
                   panel_sha256=sha256(panel_path),evaluation_sha256=sha256(evaluation_path),gpu_uuids=gpu_uuids,
                   preparation_seconds=time.monotonic()-began,utc=utc())
        durable(root/'inputs_complete.json',ready)
        durable(root/'qa/protocol.json',dict(parent_protocol_sha256=sha256(root/'protocol.json'),
                                           inputs_sha256=sha256(root/'inputs_complete.json'),maximum_real_calls=budget))
        durable(root/'synthetic/protocol.json',dict(parent_protocol_sha256=sha256(root/'protocol.json'),maximum_calls=8))
        verify(root,True); return ready


def _check_answer(answer, expected):
    """接口：实际生成观测必须与CPU冻结输入逐字段一致，严格使用原16Token生成协议。"""
    observed=answer['observed']
    for name in ('prompt','input_token_ids','pixel_sha256','source_frames','visual_tokens'):
        if observed[name]!=expected[name]: raise ValueError('Actual input mismatch: '+name)
    if observed['max_new_tokens']!=16 or observed['generate_calls']!=1 or observed['prefill_calls']!=1:
        raise ValueError('Actual generation protocol differs')


def _generate(root,row,condition,entry,backend,adapter,gpu,synthetic=False):
    """接口：每题条件唯一调用；已有完整原始返回仅恢复后处理，不重复生成。"""
    from videoqa_set_value.runtime import restore_raw
    from videoqa_set_value.task import observe, cpu_identity
    root=Path(root);stage=root/('synthetic' if synthetic else 'qa');q=row['question_id']
    answer=restore_raw(stage,q,condition,row,gpu)
    if answer is None:
        images=_load_images(entry['record'])
        try:
            # 步骤1：生成前再核对官方实际像素与提示，防止冻结后素材漂移。
            if cpu_identity((backend.processor,backend.tokenizer),images,row)!=entry['identity']:
                raise ValueError('Pre-generation identity differs')
            identity=dict(question_id=q,method=condition,gpu=gpu,protocol_sha256=sha256(stage/'protocol.json'))
            def operation(state):
                result=observe(backend,adapter,images,entry['record']['selection'],row,state,
                               stage/'raw_generations'/condition/(q+'.json'),identity)
                _check_answer(result,entry['identity']);return result
            answer,_,_=invoke_once(stage,q,condition,identity['protocol_sha256'],gpu,operation)
        finally:
            for image in images:image.close()
    _check_answer(answer,entry['identity']);return answer


def publish_result(root,q,condition,entry,answer,representative,gpu):
    """接口：共享只发布引用和同一原始调用身份，不伪造另一份物理生成或耗时。"""
    from videoqa_set_value.task import metric
    root=Path(root);evaluation=read_json(root/'evaluation.json')[q]
    score=metric(dict(answer_index=evaluation['answer_index']),answer['raw_output'])
    value=dict(question_id=q,condition=condition,status='completed',answer=answer,correct=score['correct'],
               parsed_answer=score['pred_answer'],parse_failed=score['parse_failed'],input_key=entry['input_key'],
               representative=representative,shared=(condition!=representative),gpu=gpu,
               call_path=str(root/'qa/calls'/representative/(q+'.json')))
    path=root/'results'/condition/(q+'.json')
    if path.exists() and read_json(path)!=value:raise ValueError('Completed result changed')
    if not path.exists():durable(path,value)


def process_question(root,case,backend,adapter,gpu):
    """接口：O复测→按固定轮换完成余下条件；未复现仅跳过该题，不补样或追求一致。"""
    root=Path(root);row=case['row'];q=row['question_id'];evaluation=read_json(root/'evaluation.json')[q]
    if evaluation['gpu']!=gpu:raise ValueError('Historical physical GPU differs')
    path=root/'inputs'/(q+'.json')
    if sha256(path)!=read_json(root/'inputs_complete.json')['exports'][q]:raise ValueError('Condition plan changed')
    plan=read_json(path);scoring_row=dict(row,answer_index=evaluation['answer_index'])
    # 步骤1：O始终强制一次本轮新调用，不引用历史教师答案。
    original=_generate(root,scoring_row,'O',plan['entries']['O'],backend,adapter,gpu)
    publish_result(root,q,'O',plan['entries']['O'],original,'O',gpu)
    if not historical_reproduced(original['raw_output'],evaluation['historical'][METHODS[0]]['parsed_answer']):
        for condition in CONDITIONS[1:]:
            target=root/'results'/condition/(q+'.json')
            value=dict(question_id=q,condition=condition,status='skipped_original_not_reproduced',
                       historical_option=evaluation['historical'][METHODS[0]]['parsed_answer'],
                       repeated_option=original['parsed_answer'])
            if target.exists() and read_json(target)!=value:raise ValueError('Skip status differs')
            if not target.exists():durable(target,value)
        return
    # 步骤2：按当轮输入组共享，只接收相同实际源身份/像素/提示的返回。
    for group in ordered_groups(plan['groups'],case['ordinal']):
        if (root/'PAUSE.json').exists():return
        representative=group['representative']
        answer=original if representative=='O' else _generate(root,scoring_row,representative,
                    plan['entries'][representative],backend,adapter,gpu)
        for condition in group['conditions']:
            if plan['entries'][condition]['input_key']!=group['input_key']:raise ValueError('Shared group identity differs')
            publish_result(root,q,condition,plan['entries'][condition],answer,representative,gpu)


def worker(root,gpu):
    """接口：原物理GPU的一个长期教师进程；同题五条件始终同进程，合成只执行一次。"""
    import torch
    from videoqa_runtime.llava_backend import LlavaBackend
    from videoqa_set_value.task import make_adapter,cpu_identity
    from videoqa_full.prepare import environment
    root=Path(root);protocol=verify(root);offline_environment()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu:raise ValueError('Expose exactly the assigned physical GPU')
    if environment(QA_PYTHON)!=protocol['environments']['llava']:raise ValueError('Teacher dependencies changed')
    uuid=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if uuid!=read_json(root/'inputs_complete.json')['gpu_uuids'][gpu]:raise ValueError('Physical GPU identity changed')
    session=root/'workers'/(gpu+'.json')
    if session.exists():raise RuntimeError('Existing worker session; inspect raw checkpoints, no automatic restart')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    durable(session,dict(status='loading',pid=os.getpid(),sid=os.getsid(0),gpu=gpu,utc=utc()))
    began=time.monotonic();backend=LlavaBackend();adapter=make_adapter(backend)
    durable(root/'load_reports'/(gpu+'.json'),dict(backend.load_report,worker_pid=os.getpid(),gpu=gpu))
    try:
        # 步骤1：合成通过同一任务、同一官方视频入口；灰图检查不计正式600次额度。
        gray=root/'gray.png';image=Image.open(gray).convert('RGB')
        row=dict(question_id='synthetic-'+gpu,question='Which synthetic color is visible?',
                 options=['Gray','Red','Blue','Green','Black'],answer_index=0)
        try:identity=cpu_identity((backend.processor,backend.tokenizer),[image]*8,row)
        finally:image.close()
        first=read_json(root/'panel.json')['cases'][0]['row']['question_id']
        record=read_json(root/'inputs'/(first+'.json'))['entries']['G']['record']
        _generate(root,row,'synthetic',dict(record=record,identity=identity),backend,adapter,gpu,True)
        # 步骤2：前5题按冻结顺序验收；余题按时长降序，仅影响耗时，不按得分调度。
        evaluations=read_json(root/'evaluation.json')
        cases=[c for c in read_json(root/'panel.json')['cases'] if evaluations[c['row']['question_id']]['gpu']==gpu]
        first_cases=cases[:5];remaining=sorted(cases[5:],key=lambda c:(-c['duration_seconds'],c['ordinal']))
        for ordinal,case in enumerate(first_cases+remaining):
            if (root/'PAUSE.json').exists():return
            durable(session,dict(status='generating',pid=os.getpid(),sid=os.getsid(0),gpu=gpu,
                                  question_id=case['row']['question_id'],completed=ordinal,total=len(cases),utc=utc()))
            process_question(root,case,backend,adapter,gpu)
            if ordinal==len(first_cases)-1:durable(root/'pilot_acceptance'/(gpu+'.json'),dict(
                questions=[c['row']['question_id'] for c in first_cases],input_and_execution_passed=True,utc=utc()))
        durable(session,dict(status='completed',pid=os.getpid(),sid=os.getsid(0),gpu=gpu,questions=len(cases),
                              worker_wall_seconds=time.monotonic()-began,utc=utc()))
    except BaseException as error:
        durable(root/'PAUSE.json',dict(gpu=gpu,error=repr(error),traceback=traceback.format_exc(),utc=utc()));raise


def audit(root):
    """接口：核对全部返回/共享/跳过与原输入；完成恢复只读哈希，不加载教师或评分头。"""
    from videoqa_set_value.task import strict_option
    from videoqa_audit.set_value_trainability import sealed_artifacts
    root=Path(root);protocol=verify(root,True)
    if (root/'completion.json').exists():
        old=read_json(root/'completion.json')
        for rel,value in old['artifacts'].items():
            if sha256(root/rel)!=value:raise ValueError('Completed artifact changed: '+rel)
    evaluations=read_json(root/'evaluation.json');cases=read_json(root/'panel.json')['cases']
    calls=set();skips=0;shared=0;counts=Counter();reproduced=0
    for case in cases:
        q=case['row']['question_id'];plan=read_json(root/'inputs'/(q+'.json'));ev=evaluations[q]
        o=read_json(root/'results/O'/(q+'.json'))
        same=historical_reproduced(o['answer']['raw_output'],ev['historical'][METHODS[0]]['parsed_answer']);reproduced+=same
        for condition in CONDITIONS:
            result=read_json(root/'results'/condition/(q+'.json'))
            if result['status'].startswith('skipped'):
                if same or condition=='O':raise ValueError('Invalid original-repeat skip')
                skips+=1;continue
            if not same and condition!='O':raise ValueError('Condition executed after failed original repeat')
            call=Path(result['call_path']);record=read_json(call)
            raw=read_json(root/'qa/raw_generations'/result['representative']/(q+'.json'))
            if record['status']!='completed' or raw['generation_state']!={'started':True,'returned':True}:
                raise ValueError('Incomplete raw generation')
            if record['protocol_sha256']!=sha256(root/'qa/protocol.json') or record['gpu']!=ev['gpu']:
                raise ValueError('Call protocol/GPU differs')
            if raw['identity']!=dict(question_id=q,method=result['representative'],gpu=ev['gpu'],protocol_sha256=record['protocol_sha256']):
                raise ValueError('Raw return identity differs')
            if call!=root/'qa/calls'/result['representative']/(q+'.json'):raise ValueError('Call source outside this run')
            answer=result['answer'];_check_answer(answer,plan['entries'][condition]['identity'])
            if answer!=record['answer'] or raw['raw_generated_text'].strip()!=answer['raw_output']:
                raise ValueError('Raw return/result differs')
            parsed=strict_option(answer['raw_output']);correct=parsed=='ABCDE'[ev['answer_index']]
            if result['parsed_answer']!=parsed or result['correct']!=correct or result['parse_failed']!=(parsed is None):
                raise ValueError('Independent correctness differs')
            if result['input_key']!=plan['entries'][condition]['input_key']:raise ValueError('Result input key differs')
            if plan['entries'][result['representative']]['input_key']!=result['input_key']:raise ValueError('Shared input differs')
            observed=raw['observed']
            if observed['max_new_tokens']!=16 or observed['visual_tokens']!=1680 or not 1<=len(observed['generated_token_ids'])<=16:
                raise ValueError('Observed generation limits differ')
            shared+=result['shared'];counts[condition]+=1;calls.add(str(call))
    physical=list((root/'qa/calls').glob('*/*.json'));synthetic=list((root/'synthetic/calls').glob('*/*.json'))
    if set(map(str,physical))!=calls or len(physical)>read_json(root/'inputs_complete.json')['unique_input_call_ceiling']:
        raise ValueError('Real calls do not close')
    if len(synthetic)>8 or any(read_json(p)['status']!='completed' for p in synthetic):raise ValueError('Synthetic budget/status differs')
    if len({read_json(p)['call_id'] for p in physical})!=len(physical):raise ValueError('Duplicate call IDs')
    if len(physical)+shared+skips!=600 or len(cases)!=120:raise ValueError('Result population does not close')
    protection=sealed_artifacts(SOURCE)
    if protection!=protocol['original_protection']:raise ValueError('Original experiment changed')
    return dict(status='passed',questions=120,condition_results=600,real_calls=len(physical),shared_results=shared,
                skipped_results=skips,synthetic_calls=len(synthetic),reproduced_originals=reproduced,
                complete_condition_counts=dict(counts),original_protection=protection,new_training_updates=0,
                new_blip_calls=0,new_refinement_frames=0,new_feature_calls=0,original_validation_evaluated=False)


def run(root):
    """接口：独立后台协调器等待原卡、运行、统计和审计；连接断开不改变任务生命周期。"""
    from videoqa_runtime.baseline_run import available_gpus
    from videoqa_audit.supervision_quality_report import summarize
    root=Path(root);children={}
    with exclusive(root/'coordinator.lock'):
        if (root/'completion.json').exists():return audit(root)
        try:
            if (root/'PAUSE.json').exists():raise RuntimeError('Paused run requires state inspection')
            began=time.monotonic();prepare(root)
            evaluations=read_json(root/'evaluation.json');needed=sorted({e['gpu'] for e in evaluations.values()})
            launched=set();completed=set()
            # 步骤1：只启动已冻结原物理卡；忙卡保持等待，不挪到其他空闲卡。
            while len(completed)<len(needed):
                try:free,states=available_gpus(needed)
                except RuntimeError as error:
                    if str(error)!='No requested GPU is currently idle':raise
                    free=[];states={}
                for gpu in needed:
                    if gpu in launched or gpu not in free:continue
                    if (root/'workers'/(gpu+'.json')).exists():raise RuntimeError('Prior worker session exists; no automatic replay')
                    log=(root/('worker_'+gpu+'.log')).open('a',buffering=1)
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                    process=subprocess.Popen([QA_PYTHON,str(ROOT/'scripts/run_supervision_quality_diagnostic.py'),
                        '--run-id',root.name,'--stage','worker','--gpu',gpu],cwd=ROOT,env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    children[gpu]=(process,log);launched.add(gpu)
                for gpu,(process,log) in children.items():
                    status=process.poll()
                    if status not in (None,0):raise RuntimeError('Teacher worker failed: GPU '+gpu)
                    if status==0:
                        if read_json(root/'workers'/(gpu+'.json'))['status']!='completed':raise RuntimeError('Worker ended before completion')
                        completed.add(gpu)
                if (root/'PAUSE.json').exists():raise RuntimeError('Worker paused the pipeline')
                count=sum(len(list((root/'results'/c).glob('*.json'))) for c in CONDITIONS)
                durable(root/'heartbeat.json',dict(stage='teacher' if launched-completed else 'waiting_for_original_gpu',
                    pid=os.getpid(),sid=os.getsid(0),completed_results=count,total_results=600,
                    waiting_gpus=[g for g in needed if g not in launched],gpu_states=states,
                    workers={g:p.pid for g,(p,_) in children.items()},completed_gpus=sorted(completed),
                    elapsed_seconds=time.monotonic()-began,utc=utc()))
                if len(completed)<len(needed):time.sleep(3)
            # 步骤2：所有实际返回闭合后才汇总，不以中途准确率决定调度。
            checked=audit(root);report=summarize(root,checked)
            durable(root/'audit.json',checked)
            artifacts={str(p.relative_to(root)):sha256(p) for directory in ('inputs','results','qa','synthetic','load_reports','pilot_acceptance')
                       for p in (root/directory).rglob('*') if p.is_file()}
            for name in ('panel.json','evaluation.json','inputs_complete.json','protocol.json','report.json','case_panel.json','index.html','audit.json'):
                artifacts[name]=sha256(root/name)
            durable(root/'completion.json',dict(status='completed',artifacts=artifacts,
                wall_seconds=time.monotonic()-began,original_validation_evaluated=False,utc=utc()))
            durable(root/'heartbeat.json',dict(stage='completed',pid=os.getpid(),completed_results=600,utc=utc()))
            return report
        except BaseException as error:
            durable(root/'PAUSE.json',dict(error=repr(error),traceback=traceback.format_exc(),utc=utc()));raise
        finally:
            for process,log in children.values():
                if process.poll() is None:process.wait()
                log.close()
