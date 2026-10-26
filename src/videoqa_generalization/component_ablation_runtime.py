"""组件消融的输入冻结、双控制共享问答与后台调度；生成严格记账并可恢复后处理。"""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
import contextlib
import multiprocessing as mp
import os
import queue
import shutil
import time
import traceback

import numpy as np

from videoqa_runtime.common import ROOT, read_json, sha256, offline_environment
from videoqa_full.state import durable, exclusive, rows, pilot_ids, utc, digest, invoke_once
from videoqa_full.prepare import environment
from videoqa_lmms_bridge.common import framework_identity, load_task, task_doc
from videoqa_budget32.frames import FrameProvider, validate_frames
from . import component_ablations as study

QA_FILES = ('src/videoqa_generalization/component_ablation_runtime.py',
    'src/videoqa_generalization/llava_observe.py', 'src/videoqa_generalization/llava_budget8.py',
    'src/videoqa_budget32/observe.py', 'src/videoqa_budget32/qwen_adapter.py',
    'src/videoqa_budget32/frames.py', 'src/videoqa_full/qwen_fa2_backend.py',
    'src/videoqa_full/qwen_backend.py', 'src/videoqa_runtime/llava_backend.py',
    'src/videoqa_lmms_bridge/common.py', 'src/videoqa_lmms_bridge/llava_adapter.py',
    'src/videoqa_full/state.py', 'src/videoqa_audit/region_scope_runtime.py')
_PREPARE = None


def processed_matches(profile, expected, observed):
    """接口：核对真实CPU像素与提示，不把源帧相同直接当作模型输入相同。"""
    pixel = observed.get('processor_pixel_sha256', observed.get('pixel_sha256'))
    if (pixel != expected['pixel_sha256'] or observed['prompt'] != expected['prompt']
            or observed['input_token_ids'] != expected['input_token_ids']):
        return False
    return profile != 'qwen32' or observed['video_grid_thw'] == expected['video_grid_thw']


def input_groups(items):
    """接口：仅同题内按完整输入身份分组，A优先作为代表，不读取正确性。"""
    groups = []
    for control in study.CONTROLS:
        item = items[control]
        match = next((g for g in groups if g['identity'] == item['identity']), None)
        if match is None:
            match = dict(representative=control, controls=[], identity=item['identity'])
            groups.append(match)
        match['controls'].append(control)
    return groups


def historical_result(profile, q, control=None):
    """接口：父版与对应旧开发控制的固定来源；不按答案择优，歧义路径立即拒绝。"""
    base = study.PARENTS[profile] if control is None else study.HISTORY[control]
    method = 'RD-1.2-exp12' if control is None else study.CONTROLS[control]
    paths = [base/'results'/method/(q+'.json'), base/'imported/results'/method/(q+'.json')]
    found = [p for p in paths if p.exists()]
    if not found and control is not None:
        return None
    if len(found) != 1:
        raise ValueError('Historical result missing or ambiguous: '+str(base)+'/'+q)
    return found[0]


def historical_bundle(path, value):
    """接口：由实际result路径定位物理主调用，正确处理父版导入目录及LLaVA同帧共享。"""
    path = Path(path); primary = value.get('representative', value['method'])
    base = path.parents[2]
    q = value['question_id']
    cp = base/'calls'/primary/(q+'.json'); rp = base/'raw_generations'/primary/(q+'.json')
    call, raw = read_json(cp), read_json(rp)
    if call['status'] != 'completed' or call['answer'] != value['answer']:
        raise ValueError('Historical physical call is not a confirmed matching return')
    expected = dict(question_id=q, method=primary, protocol_sha256=value['protocol_sha256'], gpu=value['gpu'])
    if (call['protocol_sha256'] != value['protocol_sha256'] or raw['identity'] != expected
            or raw['generation_state'] != dict(started=True, returned=True)):
        raise ValueError('Historical raw generation identity differs')
    return {str(p): sha256(p) for p in (path, cp, rp)}


def cpu_tools(profile):
    """接口：仅官方CPU处理器和Tokenizer，不加载视觉骨干或语言模型。"""
    import torch
    torch.set_num_threads(2)
    if profile == 'qwen32':
        from transformers import AutoProcessor
        from videoqa_full.state import QWEN_PATH
        processor = AutoProcessor.from_pretrained(QWEN_PATH, local_files_only=True, use_fast=False,
            min_pixels=100352, max_pixels=602112, size=dict(shortest_edge=100352,longest_edge=602112))
        processor.apply_chat_template([dict(role='user',content='synthetic')],tokenize=False,add_generation_prompt=True)
        return processor, None
    from transformers import AutoTokenizer
    from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
    inventory = read_json(ROOT/'configs/local_model_inventory.json')
    return SigLipImageProcessor(), AutoTokenizer.from_pretrained(inventory['models']['llava']['path'],local_files_only=True)


def cpu_identity(profile, tools, images, row):
    """接口：复用已验收的只读CPU身份函数，不调用旧exp27调度器或开发名单。"""
    from videoqa_audit.region_scope_runtime import cpu_identity as identity
    return identity(profile, tools, images, row)


def initialize_prepare(root,profile,binding_sha):
    """接口：每CPU进程独立初始化官方处理器，隔离GIL/懒导入，禁用GPU。"""
    global _PREPARE
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    offline_environment();root=Path(root)
    _PREPARE=dict(root=root,run=root/'qa'/profile,profile=profile,binding_sha=binding_sha,
        offline=read_json(root/'offline_summary.json'),source_spec=read_json(study.PARENTS[profile]/'protocol.json'),
        tools=cpu_tools(profile),dev={r['question_id'] for r in read_json(study.DEV)['rows']})
    load_task()


def prepare_one(row):
    """接口：逐题两控制的同一官方CPU路径；仅改变执行隔离，不改变像素或选帧。"""
    if _PREPARE is None:raise ValueError('CPU processor initializer missing')
    root,run,profile,binding_sha,offline,source_spec,tools,dev=(
        _PREPARE[k] for k in ('root','run','profile','binding_sha','offline','source_spec','tools','dev'))
    q=row['question_id'];items={};bank={};dest=run/'inputs'/(q+'.json')
    # 步骤1完整且来源一致的条目直接复用，不重复解码和预处理。
    if dest.exists():
        saved=read_json(dest)
        if saved['prepare_binding_sha256']!=binding_sha:raise ValueError('Partial input binding differs')
        for item in saved['items'].values():
            if sha256(item['offline_source'])!=item['offline_source_sha256']:
                raise ValueError('Partial offline selection source changed')
        for g in saved['groups']:
            if g['reference']:
                for path,h in g['reference']['sources'].items():
                    if sha256(path)!=h:raise ValueError('Partial historical source changed')
        return q,sha256(dest),saved['groups'],saved['duration_seconds']
    for control,method in study.CONTROLS.items():
        path=root/'selections'/profile/method/(q+'.json')
        if sha256(path)!=offline['exports'][str(path.relative_to(root))]:raise ValueError('Offline selection changed')
        item=read_json(path);selection=item['selection'];key=digest(study.source_identity(selection))
        if key not in bank:
            images,rgb=FrameProvider(selection).decode()
            try:processed=cpu_identity(profile,tools,images,row)
            finally:
                for im in images:im.close()
            bank[key]=(rgb,processed)
        rgb,processed=bank[key]
        identity=dict(source=study.source_identity(selection),rgb_sha256=rgb,processed=processed,
            model_files_sha256=digest(source_spec['model_files']),generation=dict(seed=2027,max_new_tokens=16,do_sample=False))
        items[control]=dict(control=control,method=method,selection=selection,rgb_sha256=rgb,
            processed=processed,identity=identity,offline_source=str(path),offline_source_sha256=sha256(path))
    groups=input_groups(items)
    # 步骤2仍固定父版→旧对应控制优先级，正确性不参与输入或引用选择。
    candidates=[historical_result(profile,q)]
    if profile=='qwen32' and q in dev:candidates += [historical_result(profile,q,c) for c in study.CONTROLS]
    histories=[(p,read_json(p)) for p in candidates if p is not None]
    for group in groups:
        example=items[group['representative']];reference=None
        for path,old in histories:
            if any(old[key]!=row[key] for key in ('question_id','video_id','question','options','stratum')):
                raise ValueError('Historical question/options differ')
            if study.source_identity(old['selection'])!=group['identity']['source']:continue
            if old['rgb_sha256']!=example['rgb_sha256'] or not processed_matches(profile,example['processed'],old['answer']['observed']):
                raise ValueError('Same-source historical model pixels/tokens differ')
            reference=dict(path=str(path),sources=historical_bundle(path,old));break
        group.pop('identity');group['reference']=reference
    # 步骤3原子提交，与四线程版保持同一输入数据结构及实际模型预处理。
    value=dict(question_id=q,profile=profile,items=items,groups=groups,prepare_binding_sha256=binding_sha,
        duration_seconds=float(Fraction(items['A']['selection']['video']['duration_fraction'])))
    study.freeze(dest,value)
    return q,sha256(dest),groups,value['duration_seconds']


def prepare(root, profile, allowed_gpus):
    """接口：2700题两控制→5400方法结果的输入与物理调用划分，先冻结再生成。"""
    root = Path(root); run = root/'qa'/profile; run.mkdir(parents=True,exist_ok=True)
    offline_environment(); k = study.budget(profile); parent = study.PARENTS[profile]
    source_spec = read_json(parent/'protocol.json'); data = rows()
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():
            return verify(run)
        offline = read_json(root/'offline_summary.json')
        if offline['status'] != 'passed' or offline['parent_replays'] != 5400:
            raise ValueError('All offline controls must pass before input freeze')
        if environment(study.PYTHONS[profile]) != source_spec['environment'] or framework_identity() != source_spec['framework']:
            raise ValueError('Historical QA environment/framework changed')
        code = {p: sha256(ROOT/p) for p in set((*study.CORE_FILES,*QA_FILES))}
        for p in QA_FILES:
            if p in source_spec['code_sha256'] and code[p] != source_spec['code_sha256'][p]:
                raise ValueError('Frozen inference implementation changed: '+p)
        # 步骤1权重与视频先检查状态，变化时重算完整摘要，不静默更换资产。
        model_files = copy_file_stats(source_spec['model_files'])
        video_stats = {}
        for vid, old in source_spec['video_stats'].items():
            path = Path(old['path']); actual = path.stat()
            if actual.st_size != old['bytes'] or actual.st_mtime_ns != old['mtime_ns']:
                if sha256(path) != source_spec['video_hashes'][vid]:
                    raise ValueError('Original source video changed')
            video_stats[vid] = dict(old,bytes=actual.st_size,mtime_ns=actual.st_mtime_ns)
        binding=dict(profile=profile,offline_protocol_sha256=sha256(root/'offline_protocol.json'),
                     offline_summary_sha256=sha256(root/'offline_summary.json'),code_sha256=code,
                     model_files_sha256=digest(model_files),video_stats_sha256=digest(video_stats),
                     parent_protocol_sha256=sha256(parent/'protocol.json'))
        study.freeze(run/'prepare_binding.json',binding)
        binding_sha=sha256(run/'prepare_binding.json')
        exports = {}; groups_by_q = {}; began = time.perf_counter()
        with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),
                initializer=initialize_prepare,initargs=(str(root),profile,binding_sha)) as executor:
            for i,(q,h,groups,seconds) in enumerate(executor.map(prepare_one,data),1):
                exports['inputs/'+q+'.json'] = h; groups_by_q[q] = groups
                vid=next(r['video_id'] for r in data if r['question_id']==q)
                if 'duration_seconds' in video_stats[vid] and video_stats[vid]['duration_seconds']!=seconds:
                    raise ValueError('Shared video duration differs between questions')
                video_stats[vid]['duration_seconds']=seconds
                if i%10==0:durable(run/'prepare_status.json',dict(status='running',completed=i,total=2700,
                    seconds=time.perf_counter()-began,utc=utc()))
        new = {q:[g['representative'] for g in gs if g['reference'] is None] for q,gs in groups_by_q.items()}
        maximum = sum(map(len,new.values()))
        if maximum > 5400:
            raise ValueError('Formal call cap exceeded')
        spec = dict(profile=profile,model=source_spec['model'],frame_budget=k,methods=study.CONTROLS,
            question_ids=[r['question_id'] for r in data],max_new_calls=maximum,new_groups=new,
            input_exports=exports,allowed_gpus=allowed_gpus,model_files=model_files,video_stats=video_stats,
            environment=source_spec['environment'],framework=source_spec['framework'],code_sha256=code,
            parent_protocol=str(parent/'protocol.json'),parent_protocol_sha256=sha256(parent/'protocol.json'),
            offline_protocol_sha256=sha256(root/'offline_protocol.json'),offline_summary_sha256=sha256(root/'offline_summary.json'),
            historical_generation_config=read_json(historical_result(profile,data[0]['question_id']))['answer']['observed']['model_generation_config'],
            pilot_question_ids=pilot_ids(data),max_new_tokens=16,seed=2027,utc=utc())
        for name in code:
            target=run/'code_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        study.freeze(run/'protocol.json',spec)
        materialize_imports(run,spec)
        durable(run/'prepare_status.json',dict(status='completed',completed=2700,total=2700,
            method_results=5400,new_calls=maximum,seconds=time.perf_counter()-began,utc=utc()))
        return spec


def copy_file_stats(files):
    """接口：共享模型只读使用；若时间戳变化但字节相同，记录当前状态与原摘要。"""
    result = {}
    for path, old in files.items():
        stat=Path(path).stat()
        if stat.st_size!=old['bytes'] or stat.st_mtime_ns!=old['mtime_ns']:
            if sha256(path)!=old['sha256']:raise ValueError('Frozen model bytes changed')
        result[path]=dict(old,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    return result


def verify(run):
    """接口：环境、代码、来源和输入指纹异常暂停；不执行生成。"""
    run=Path(run); spec=read_json(run/'protocol.json'); root=run.parent.parent
    if spec['max_new_calls']!=sum(map(len,spec['new_groups'].values())) or spec['max_new_calls']>5400:
        raise ValueError('Frozen call partition differs')
    if framework_identity()!=spec['framework'] or environment(study.PYTHONS[spec['profile']])!=spec['environment']:
        raise ValueError('Frozen environment changed')
    for name,h in spec['code_sha256'].items():
        if sha256(ROOT/name)!=h:raise ValueError('Frozen inference code changed: '+name)
    for path,item in spec['model_files'].items():
        current=Path(path).stat()
        if current.st_size!=item['bytes'] or current.st_mtime_ns!=item['mtime_ns']:
            raise ValueError('Model file identity changed')
    if sha256(spec['parent_protocol'])!=spec['parent_protocol_sha256']:
        raise ValueError('Historical parent protocol changed')
    if sha256(root/'offline_protocol.json')!=spec['offline_protocol_sha256'] or sha256(root/'offline_summary.json')!=spec['offline_summary_sha256']:
        raise ValueError('Offline protocol/input manifest changed')
    for relative,h in spec['input_exports'].items():
        if sha256(run/relative)!=h:raise ValueError('Frozen actual input changed')
    return spec


def result_from_answer(run,row,control,answer,gpu,kind,origins=None,representative=None,timings=None):
    """接口：标准答案仅在原始返回之后计分；方法级引用与物理调用分开保存。"""
    run=Path(run); plan=read_json(run/'inputs'/(row['question_id']+'.json')); item=plan['items'][control]
    metric=load_task().videomme_process_results(task_doc(row),[answer['raw_output']])['videomme_perception_score']
    if metric!=answer['framework_metric']:
        raise ValueError('Framework scoring differs from preserved raw output')
    model='LLaVA-Video-7B-Qwen2' if plan['profile']=='llava8' else 'Qwen2.5-VL-7B-Instruct'
    return dict(status='completed',profile=plan['profile'],model=model,method=study.CONTROLS[control],control=control,
        **{k:row[k] for k in ('question_id','video_id','question','options','stratum')},
        protocol_sha256=sha256(run/'protocol.json'),selection=item['selection'],rgb_sha256=item['rgb_sha256'],
        answer=answer,correct=metric['answer']==metric['pred_answer'],gpu=gpu,timing_kind=kind,
        representative=representative or control,reference_sources=origins or {},timings=timings or {},utc=utc())


def materialize_imports(run,spec):
    """接口：只导入已冻结同输入组的历史返回；不生成新账本或伪造QA计时。"""
    bank={r['question_id']:r for r in rows()}; run=Path(run)
    for q in spec['question_ids']:
        plan=read_json(run/'inputs'/(q+'.json'))
        for group in plan['groups']:
            ref=group['reference']
            if ref is None:continue
            for path,h in ref['sources'].items():
                if sha256(path)!=h:raise ValueError('Historical reference changed')
            old=read_json(ref['path'])
            for c in group['controls']:
                target=run/'results'/study.CONTROLS[c]/(q+'.json')
                if not target.exists():durable(target,result_from_answer(run,bank[q],c,old['answer'],old['gpu'],
                    'historical_input_verified_import',ref['sources'],group['representative']))


def validate_result(run,row,control,spec):
    """接口：审计方法结果与物理调用/引用；解析失败计错，不重新回答。"""
    run=Path(run);q=row['question_id'];method=study.CONTROLS[control]
    result=read_json(run/'results'/method/(q+'.json')); plan=read_json(run/'inputs'/(q+'.json'))
    item=plan['items'][control]; answer=result['answer']; observed=answer['observed']
    if result['protocol_sha256']!=sha256(run/'protocol.json') or result['method']!=method or result['model']!=spec['model']:
        raise ValueError('Method result/protocol identity differs')
    if result['selection']!=item['selection'] or result['rgb_sha256']!=item['rgb_sha256']:
        raise ValueError('Method source identity differs')
    if not processed_matches(spec['profile'],item['processed'],observed):
        raise ValueError('Actual model pixels/prompt/token identity differs')
    for key in ('question_id','video_id','question','options','stratum'):
        if result[key]!=row[key]:raise ValueError('Result question/options differ')
    n=len(item['selection']['selected_frames']);validate_frames(item['selection'],spec['frame_budget'])
    if observed['generate_calls']!=1 or observed['prefill_calls']!=1 or not 1<=len(observed['generated_token_ids'])<=16:
        raise ValueError('Generation count/length differs')
    if observed['model_generation_config']!=spec['historical_generation_config']:
        raise ValueError('Actual model generation defaults differ')
    if spec['profile']=='llava8' and (observed['visual_tokens']!=n*210 or observed['max_new_tokens']!=16):
        raise ValueError('LLaVA media/token protocol differs')
    if spec['profile']=='qwen32' and (observed['generation_kwargs']['max_new_tokens']!=16 or observed['encoding_fps']!=2):
        raise ValueError('Qwen media/token protocol differs')
    metric=load_task().videomme_process_results(task_doc(row),[answer['raw_output']])['videomme_perception_score']
    if metric!=answer['framework_metric'] or result['correct']!=(metric['answer']==metric['pred_answer']):
        raise ValueError('Reference scoring changed')
    if result['timing_kind']=='historical_input_verified_import':
        if not result['reference_sources']:raise ValueError('Import lacks physical provenance')
        for path,h in result['reference_sources'].items():
            if sha256(path)!=h:raise ValueError('Imported raw result changed')
    else:
        rep=result['representative'];primary=study.CONTROLS[rep]
        call=read_json(run/'calls'/primary/(q+'.json')); raw=read_json(run/'raw_generations'/primary/(q+'.json'))
        if call['status']!='completed' or call['answer']!=answer:
            raise ValueError('Physical primary call/answer differs')
        if raw['generation_state']!=dict(started=True,returned=True) or raw['identity']!=dict(
                question_id=q,method=primary,protocol_sha256=result['protocol_sha256'],gpu=call['gpu']):
            raise ValueError('Raw return identity differs')
    return result


def recover_group(run,row,group,spec):
    """接口：生成已返回只恢复后处理；未返回/身份不明禁止再次生成。"""
    run=Path(run);q=row['question_id'];rep=group['representative'];method=study.CONTROLS[rep]
    cp=run/'calls'/method/(q+'.json')
    if not cp.exists():return False
    call=read_json(cp);raw=read_json(run/'raw_generations'/method/(q+'.json'))
    fp=sha256(run/'protocol.json')
    if call['protocol_sha256']!=fp or raw['identity']!=dict(question_id=q,method=method,protocol_sha256=fp,gpu=call['gpu']):
        raise ValueError('Recoverable call fingerprint differs')
    if raw['generation_state']!=dict(started=True,returned=True):
        raise ValueError('Unknown generation may not be repeated')
    if call['status']!='completed':
        if spec['profile']=='qwen32':
            from videoqa_budget32.observe import result_from_raw
            answer=result_from_raw(raw,row)
        else:
            from .llava_observe import result_from_raw
            value=result_from_raw(raw,row);answer=dict(value['answer'],observed=value['observed'])
        call.update(status='completed',answer=answer,generation_state=raw['generation_state'],recovered_postprocessing=True)
        durable(cp,call)
    for control in group['controls']:
        dest=run/'results'/study.CONTROLS[control]/(q+'.json')
        if not dest.exists():durable(dest,result_from_answer(run,row,control,call['answer'],call['gpu'],
            'returned_raw_postprocessing' if control==rep else 'same_run_input_shared',representative=rep))
    return True


def process_question(run,row,backend,adapter,gpu,spec):
    """接口：同题两控制按冻结次序轮换，仅未调用的物理输入组进入模型。"""
    run=Path(run);q=row['question_id'];plan=read_json(run/'inputs'/(q+'.json'))
    shift=spec['question_ids'].index(q)%2
    order=list(study.CONTROLS);order=order[shift:]+order[:shift]
    groups=sorted(plan['groups'],key=lambda g:min(order.index(c) for c in g['controls']))
    for group in groups:
        if group['reference'] is not None:continue
        rep=group['representative'];method=study.CONTROLS[rep]
        if recover_group(run,row,group,spec):continue
        with exclusive(run/'question_locks'/method/(q+'.lock')):
            item=plan['items'][rep];began=time.perf_counter()
            images,rgb=FrameProvider(item['selection']).decode()
            try:
                tools=(backend.processor,backend.tokenizer if spec['profile']=='llava8' else None)
                if rgb!=item['rgb_sha256'] or cpu_identity(spec['profile'],tools,images,row)!=item['processed']:
                    raise ValueError('Runtime actual input differs before generation')
                decoded=time.perf_counter();fp=sha256(run/'protocol.json')
                identity=dict(question_id=q,method=method,protocol_sha256=fp,gpu=gpu)
                def invoke(state):
                    path=run/'raw_generations'/method/(q+'.json')
                    if spec['profile']=='qwen32':
                        from videoqa_budget32.observe import observe
                        return observe(backend,adapter,images,row,state,path,identity)
                    from .llava_observe import observe
                    value=observe(backend,adapter,images,item['selection'],row,state,path,identity)
                    return dict(value['answer'],observed=value['observed'])
                answer,_,checkpoint=invoke_once(run,q,method,fp,gpu,invoke)
            finally:
                for im in images:im.close()
            if read_json(run/'calls'/method/(q+'.json'))['gpu']!=gpu:
                raise ValueError('Partially completed question moved physical GPU')
            for control in group['controls']:
                result=result_from_answer(run,row,control,answer,gpu,
                    'new_physical_call' if control==rep else 'same_run_input_shared',representative=rep,
                    timings=dict(final_decode_and_identity_seconds=decoded-began,
                        cached_input_to_persisted_answer_seconds=time.perf_counter()-began,
                        checkpoint_seconds=checkpoint) if control==rep else {})
                durable(run/'results'/study.CONTROLS[control]/(q+'.json'),result)
    for control in study.CONTROLS:validate_result(run,row,control,spec)


def synthetic_cases(profile,extended):
    """接口：首卡固定边界验收，其余每卡一次预热，两模型最多21次合成。"""
    if not extended:return [(study.budget(profile),(896,672))]
    return ([(1,(320,240)),(7,(720,1280)),(8,(896,672))] if profile=='llava8' else
            [(1,(320,240)),(11,(720,1280)),(31,(896,672)),(32,(896,672))])


def worker(run,gpu,jobs,events,stop,extended):
    """接口：一卡常驻一个正式后端，合成与真实调用分账，失败后停止领取任务。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    offline_environment();run=Path(run);active=None
    with (run/'logs'/('gpu_'+gpu+'.log')).open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import torch
            from PIL import Image
            from videoqa_lmms_bridge.bridge import synthetic_row
            torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(2027)
            spec=read_json(run/'protocol.json');started=time.perf_counter()
            if spec['profile']=='qwen32':
                from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
                from videoqa_budget32.qwen_adapter import FrozenInputQwen
                backend=QwenFA2VideoBackend();adapter=FrozenInputQwen(backend)
            else:
                from videoqa_runtime.llava_backend import LlavaBackend
                from .llava_budget8 import adapter_for
                backend=LlavaBackend();adapter=adapter_for(backend)
            if backend.model.generation_config.to_dict()!=spec['historical_generation_config']:
                raise ValueError('Frozen backend generation defaults changed')
            loads=time.perf_counter()-started;warm=[]
            for ordinal,(n,size) in enumerate(synthetic_cases(spec['profile'],extended)):
                path=run/'synthetic'/gpu/(str(ordinal)+'.json')
                if path.exists():raise ValueError('Synthetic call already started; do not silently rewarm')
                durable(path,dict(status='started',gpu=gpu,source_frames=n,utc=utc()))
                images=[Image.new('RGB',size,(30,i*7%256,150)) for i in range(n)]
                try:
                    if spec['profile']=='qwen32':
                        from videoqa_budget32.observe import observe
                        result=observe(backend,adapter,images,synthetic_row(n),{})
                    else:
                        from .llava_observe import observe
                        selection=dict(video=dict(path='synthetic.mp4',duration_seconds=float(n)),
                                       selected_frames=[dict(timestamp_seconds=float(i)) for i in range(n)])
                        value=observe(backend,adapter,images,selection,synthetic_row(n),{})
                        result=dict(value['answer'],observed=value['observed'])
                    warm.append(result['observed'])
                    durable(path,dict(status='completed',gpu=gpu,source_frames=n,answer=result,utc=utc()))
                finally:
                    for im in images:im.close()
            durable(run/'workers'/('gpu_'+gpu+'.json'),dict(gpu=gpu,pid=os.getpid(),sid=os.getsid(0),
                load_seconds=loads,synthetic_calls=len(warm),load_report=backend.load_report,utc=utc()))
            events.put(dict(type='ready',gpu=gpu))
            while not stop.is_set():
                row=jobs.get()
                if row is None:break
                active=row['question_id'];events.put(dict(type='active',gpu=gpu,question_id=active))
                process_question(run,row,backend,adapter,gpu,spec)
                events.put(dict(type='completed',gpu=gpu,question_id=active));active=None
        except BaseException as exc:
            stop.set();traceback.print_exc()
            durable(run/'failures'/('gpu_'+gpu+'.json'),dict(question_id=active,error=repr(exc),utc=utc()))
            events.put(dict(type='failed',gpu=gpu,error=repr(exc)))


def audit(run):
    """接口：5400方法结果与冻结物理调用闭合；完成态不加载模型。"""
    run=Path(run);spec=verify(run);data=rows();calls=list((run/'calls').glob('*/*.json'))
    for row in data:
        for control in study.CONTROLS:validate_result(run,row,control,spec)
    if len(calls)!=spec['max_new_calls']:
        raise ValueError('Physical generation count differs from frozen input groups')
    keys=[]
    for p in calls:
        value=read_json(p)
        if value['status']!='completed':raise ValueError('Unresolved physical generation')
        keys.append(value['call_id'])
    if len(set(keys))!=len(keys):raise ValueError('Duplicate physical call IDs')
    synthetic=list((run/'synthetic').glob('*/*.json'))
    if any(read_json(p)['status']!='completed' for p in synthetic):raise ValueError('Unresolved synthetic generation')
    result=dict(status='passed',profile=spec['profile'],method_results=5400,
                physical_new_calls=len(calls),synthetic_calls=len(synthetic),new_blip_calls=0,
                new_selection_visual_calls=0,protocol_sha256=sha256(run/'protocol.json'),utc=utc())
    study.freeze(run/'completion_audit.json',{k:v for k,v in result.items() if k!='utc'})
    return result


def execute(run):
    """接口：先恢复已返回答案，再固定空闲卡，自动先行验收后调度全部余题。"""
    run=Path(run);spec=verify(run);data=rows();materialize_imports(run,spec)
    for row in data:
        plan=read_json(run/'inputs'/(row['question_id']+'.json'))
        for group in plan['groups']:
            if group['reference'] is None:recover_group(run,row,group,spec)
    pending=[r for r in data if any(not (run/'results'/m/(r['question_id']+'.json')).exists() for m in study.CONTROLS.values())]
    if not pending:return audit(run)
    from videoqa_runtime.baseline_run import available_gpus
    allocation=run/'gpu_allocation.json'
    while True:
        try:free,_=available_gpus(spec['allowed_gpus'])
        except RuntimeError:free=[]
        if allocation.exists():
            gpus=read_json(allocation)['gpus'];ready=set(gpus)<=set(free)
        else:gpus=list(free);ready=bool(gpus)
        if ready:break
        durable(run/'status.json',dict(status='waiting_resources',utc=utc()));time.sleep(15)
    if not allocation.exists():study.freeze(allocation,dict(gpus=gpus))
    for folder in ('logs','workers','failures'):(run/folder).mkdir(exist_ok=True)
    ctx=mp.get_context('spawn');events=ctx.Queue();stop=ctx.Event()
    queues={g:ctx.Queue() for g in gpus};processes={}
    began=time.perf_counter();done=0;active={};ready=set()
    # 步骤1所有卡加载后只允许首卡执行固定先行题，其余等待放行。
    for i,g in enumerate(gpus):
        p=ctx.Process(target=worker,args=(str(run),g,queues[g],events,stop,i==0));p.start();processes[g]=p
    pilot=[r for r in pending if r['question_id'] in spec['pilot_question_ids']]
    remaining=sorted([r for r in pending if r['question_id'] not in spec['pilot_question_ids']],
                     key=lambda r:-spec['video_stats'][r['video_id']]['duration_seconds'])
    pilot_ids={r['question_id'] for r in pilot};pilot_sent=False;full=False;idle=set();completed_pilot=set()
    assigned={}
    for path in (run/'calls').glob('*/*.json'):
        call=read_json(path);q=call['question_id'];g=call['gpu']
        if q in assigned and assigned[q]!=g:raise ValueError('Same question has multiple physical GPU assignments')
        assigned[q]=g
    try:
        while done<len(pending):
            try:event=events.get(timeout=5)
            except queue.Empty:
                failed=[g for g,p in processes.items() if p.exitcode is not None]
                if failed:raise RuntimeError('Worker exited: '+str(failed))
                durable(run/'heartbeat.json',dict(pid=os.getpid(),sid=os.getsid(0),completed=done,total=len(pending),
                    active=active,seconds=time.perf_counter()-began,utc=utc()));continue
            kind=event['type'];gpu=event['gpu']
            if kind=='failed':raise RuntimeError(str(event))
            if kind=='ready':ready.add(gpu);idle.add(gpu)
            if kind=='active':active[gpu]=event['question_id']
            if kind=='completed':
                done+=1;active.pop(gpu,None);idle.add(gpu)
                if event['question_id'] in pilot_ids:completed_pilot.add(event['question_id'])
            if len(ready)==len(gpus) and not pilot_sent:
                for r in pilot:
                    g=assigned.get(r['question_id'],gpus[0]);queues[g].put(r);idle.discard(g)
                pilot_sent=True
            if pilot_sent and completed_pilot==pilot_ids and not full:
                for q in spec['pilot_question_ids']:
                    row=next(r for r in data if r['question_id']==q)
                    for c in study.CONTROLS:validate_result(run,row,c,spec)
                durable(run/'pilot_acceptance.json',dict(status='passed',method_results=14,utc=utc()))
                full=True
            if full:
                for g in sorted(idle):
                    eligible=next((i for i,r in enumerate(remaining) if assigned.get(r['question_id'],g)==g),None)
                    if eligible is not None:
                        row=remaining.pop(eligible);queues[g].put(row);active[g]=row['question_id']
                idle={g for g in idle if g not in active}
            elapsed=time.perf_counter()-began
            durable(run/'status.json',dict(status='answering',completed=done,total=len(pending),
                active=active,seconds=elapsed,eta_seconds=elapsed/max(done,1)*(len(pending)-done),utc=utc()))
        for g in gpus:queues[g].put(None)
        for p in processes.values():p.join(timeout=60)
        if any(p.is_alive() or p.exitcode!=0 for p in processes.values()):raise RuntimeError('Worker completion not clean')
    except BaseException:
        stop.set()
        for q in queues.values():q.put(None)
        for p in processes.values():p.join(timeout=60)
        durable(run/'status.json',dict(status='paused',utc=utc()));raise
    durable(run/'execution.json',dict(status='completed',gpus=gpus,questions_with_new_calls=len(pending),
                                     wall_seconds=time.perf_counter()-began,utc=utc()))
    return audit(run)
