"""同粗池双配置开发问答：输入先冻结、有限调用、后台执行和安全恢复。"""
import contextlib
from fractions import Fraction
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback

from . import selection_structure as method
from . import selection_structure_run as study


def input_groups(items):
    """接口：同一题内按完整输入身份共享；固定G→W代表顺序，与答案无关。"""
    groups = []
    for control in method.CONTROLS:
        item = items[control]
        found = next((g for g in groups if g['identity']==item['identity']),None)
        if found is None:
            found = dict(representative=control,methods=[],identity=item['identity']); groups.append(found)
        found['methods'].append(control)
    return groups


def historical_context(path,value):
    """接口：普通结果读取本层协议；旧逐字节imported副本按父协议的显式imports溯源。"""
    path=Path(path); base=path.parents[2]; protocol=base/'protocol.json'; sources={}
    # 步骤1不按相似文件名猜测协议；只使用该导入批次已经冻结的origin及files映射。
    if not protocol.exists():
        if base.name!='imported': raise ValueError('Historical protocol missing')
        wrapper_path=base.parent/'protocol.json'; wrapper=study.read(wrapper_path)
        imported=wrapper['imports'][value['question_id']]
        for relative,h in imported['files'].items():
            artifact=base.parent/relative
            if not artifact.resolve().is_relative_to(base.resolve()) or study.sha(artifact)!=h:
                raise ValueError('Historical imported artifact differs')
            sources[str(artifact)]=h
        if str(path) not in sources or imported['protocol_sha256']!=value['protocol_sha256']:
            raise ValueError('Historical imported protocol binding differs')
        protocol=Path(imported['origin'])/'protocol.json'
        sources[str(wrapper_path)]=study.sha(wrapper_path)
    if study.sha(protocol)!=value['protocol_sha256']: raise ValueError('Historical protocol differs')
    sources[str(protocol)]=study.sha(protocol)
    return study.read(protocol),sources


def physical_sources(path, value, seen=None):
    """接口：沿引用追溯真实调用及原始返回；无法闭合的旧结果禁止导入。"""
    path = Path(path); seen = set() if seen is None else seen
    if str(path) in seen: raise ValueError('Cyclic historical reference')
    seen.add(str(path)); sources = {str(path):study.sha(path)}
    # 步骤1旧引用的所有哈希均需相同；候选来源按路径稳定排序，不读取正确性。
    for p,h in value.get('reference_sources',{}).items():
        if study.sha(p)!=h: raise ValueError('Historical reference source changed')
        sources[p]=h
    q=value['question_id']; base=path.parents[2]
    spec,context_sources=historical_context(path,value);sources.update(context_sources)
    primary=value.get('representative',value['method'])
    primary=spec.get('methods',{}).get(primary,primary) if isinstance(spec.get('methods'),dict) else primary
    cp=base/'calls'/primary/(q+'.json'); rp=base/'raw_generations'/primary/(q+'.json')
    if cp.exists():
        call,raw=study.read(cp),study.read(rp)
        identity=dict(question_id=q,method=primary,protocol_sha256=value['protocol_sha256'],gpu=call['gpu'])
        if (call['status']!='completed' or call['answer']!=value['answer']
                or call['protocol_sha256']!=value['protocol_sha256'] or raw['identity']!=identity
                or raw['generation_state']!=dict(started=True,returned=True)):
            raise ValueError('Historical physical return is uncertain or mismatched')
        sources.update({str(cp):study.sha(cp),str(rp):study.sha(rp)})
        return sources
    ancestors=[Path(p) for p in value.get('reference_sources',{}) if '/results/' in p and Path(p).name==q+'.json']
    for candidate in sorted(ancestors):
        old=study.read(candidate)
        if old.get('answer')==value['answer']:
            sources.update(physical_sources(candidate,old,seen)); return sources
    raise ValueError('Historical result has no confirmed physical provenance')


def histories(profile,q):
    """接口：预先固定P、TopK、exp12、exp07、Uniform优先级，不按得分挑答案。"""
    from videoqa_audit.region_scope_runtime import parent_result
    first=study.PARENT_RUN/'qa'/profile/'results'/method.PARENT/(q+'.json')
    return [first]+[parent_result(profile,q,m) for m in
        ('BASE-BLIP-TopK','RD-1.2-exp12','RD-1.2-exp07','BASE-Uniform')]


def _model_hashes(spec):
    """接口：模型字节身份独立于旧文件mtime，避免将相同路径误当作同一权重。"""
    return {p:v['sha256'] for p,v in spec['model_files'].items()}


def _reference(path,old,item,row,source_spec,profile):
    """接口：完整输入及后端协议相同才返回导入引用，否则拒绝不确定来源。"""
    from videoqa_generalization.component_ablation_runtime import processed_matches
    if any(old[k]!=row[k] for k in ('question_id','video_id','question','options','stratum')):
        raise ValueError('Historical task identity differs')
    if study.source_identity(old['selection'])!=item['identity']['source']: return None
    if old['rgb_sha256']!=item['rgb_sha256'] or not processed_matches(profile,item['processed'],old['answer']['observed']):
        raise ValueError('Same source has different historical actual input')
    old_spec,_=historical_context(path,old)
    if (old['model']!=source_spec['model'] or _model_hashes(old_spec)!=_model_hashes(source_spec)
            or old_spec['environment']!=source_spec['environment'] or old_spec['framework']!=source_spec['framework']
            or old['answer']['observed']['model_generation_config']!=source_spec['historical_generation_config']):
        raise ValueError('Historical weights/environment/task identity differs')
    return dict(path=str(path),sources=physical_sources(path,old))


def processor_boundaries(profile,tools):
    """接口：官方CPU处理器边界验收，不加载模型或生成Token；合成生成额度另计。"""
    from PIL import Image
    from videoqa_generalization.component_ablation_runtime import cpu_identity
    from videoqa_lmms_bridge.bridge import synthetic_row
    cases=(1,7,8) if profile=='llava8' else (1,11,31,32)
    observations=[]
    # 步骤1按不同横竖尺寸构造无答案合成RGB；步骤2核验实际媒体时间/空间网格。
    for ordinal,n in enumerate(cases):
        size=(896,672) if ordinal%2==0 else (672,896)
        images=[Image.new('RGB',size,(40,i*7%256,120)) for i in range(n)]
        try:
            actual=cpu_identity(profile,tools,images,synthetic_row(n))
            if profile=='llava8':
                pixels=tools[0].preprocess(images,return_tensors='pt')['pixel_values']
                if list(pixels.shape)!=[n,3,384,384]: raise ValueError('Official LLaVA processor truncated/reordered frame slots')
                detail=dict(shape=list(pixels.shape),expected_visual_tokens=n*210)
            else:
                grid=actual['video_grid_thw']
                if len(grid)!=1 or grid[0][0]!=(n+1)//2: raise ValueError('Official Qwen temporal patch differs')
                t,h,w=grid[0]; pixels=h*w*14*14
                if h%2 or w%2 or not 100352<=pixels<=602112: raise ValueError('Official Qwen spatial bounds differ')
                detail=dict(video_grid_thw=grid,encoded_frames=t*2,internal_padding_frames=t*2-n,
                            expected_visual_tokens=t*h*w//4,processed_pixels_per_frame=pixels)
            observations.append(dict(source_frames=n,source_size=list(size),**detail,
                pixel_sha256=actual['pixel_sha256'],input_token_count=len(actual['input_token_ids'])))
        finally:
            for im in images: im.close()
    return dict(status='passed',profile=profile,cases=observations,new_model_calls=0)


def prepare(root,profile,allowed_gpus):
    """接口：只为开发200题解码及官方CPU预处理，冻结≤400次物理输入组。"""
    from videoqa_full.state import exclusive
    from videoqa_full.prepare import environment
    from videoqa_lmms_bridge.common import framework_identity
    from videoqa_generalization import component_ablation_runtime as prior
    from videoqa_budget32.frames import FrameProvider,validate_frames
    root=Path(root); run=root/'qa'/profile; k=study.budget(profile)
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists(): return verify(run)
        offline=study.verify_offline(root); data=study.development_rows()
        src=study.PARENT_RUN/'qa'/profile/'protocol.json'; source_spec=study.read(src)
        if environment(study.PYTHONS[profile])!=source_spec['environment'] or framework_identity()!=source_spec['framework']:
            raise ValueError('Frozen QA environment or task changed')
        model_files=prior.copy_file_stats(source_spec['model_files'])
        video_stats={}
        for vid in sorted({r['video_id'] for r in data}):
            old=source_spec['video_stats'][vid]; stat=Path(old['path']).stat()
            if stat.st_size!=old['bytes'] or stat.st_mtime_ns!=old['mtime_ns']:
                if study.sha(old['path'])!=source_spec['video_hashes'][vid]: raise ValueError('Source video changed')
            video_stats[vid]=dict(path=old['path'],bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
        code={p:study.sha(study.ROOT/p) for p in set((*study.FILES,*prior.QA_FILES))}
        binding=dict(profile=profile,code_sha256=code,model_hashes=_model_hashes(source_spec),
                     parent_protocol_sha256=study.sha(src),offline_summary_sha256=study.sha(root/'offline_summary.json'),
                     dev_manifest_sha256=study.sha(study.DEV),allowed_gpus=allowed_gpus,video_stats=video_stats)
        study.freeze(run/'prepare_binding.json',binding); binding_sha=study.sha(run/'prepare_binding.json')
        tools=prior.cpu_tools(profile); exports={}; new={}; durations={}; began=time.perf_counter()
        boundary_path=run/'processor_boundaries.json'
        if not boundary_path.exists(): study.freeze(boundary_path,processor_boundaries(profile,tools))
        exports['processor_boundaries.json']=study.sha(boundary_path)
        # 步骤1题目全文仅传给官方CPU模板；计分标签不进入选择或分组。
        for ordinal,row in enumerate(data):
            q=row['question_id']; dest=run/'inputs'/(q+'.json')
            if dest.exists():
                plan=study.read(dest)
                if plan['question_id']!=q or plan['profile']!=profile or plan['prepare_binding_sha256']!=binding_sha:
                    raise ValueError('Cached input slot or preparation binding differs')
                for item in plan['items'].values():
                    if study.sha(item['offline_source'])!=item['offline_source_sha256']: raise ValueError('Cached source changed')
                for group in [*plan['groups'],plan['parent_group']]:
                    if group['reference']:
                        for p,h in group['reference']['sources'].items():
                            if study.sha(p)!=h: raise ValueError('Cached historical origin changed')
            else:
                items={}; bank={}; candidates=[(p,study.read(p)) for p in histories(profile,q)]
                for m in (method.PARENT,*method.CONTROLS):
                    sp=root/'selections'/profile/m/(q+'.json')
                    if study.sha(sp)!=offline['exports'][str(sp.relative_to(root))]: raise ValueError('Frozen selection changed')
                    selection=study.read(sp)['selection']; validate_frames(selection,k)
                    identity=study.source_identity(selection); key=study.digest(identity)
                    if key not in bank:
                        images,rgb=FrameProvider(selection).decode()
                        try: processed=prior.cpu_identity(profile,tools,images,row)
                        finally:
                            for im in images: im.close()
                        bank[key]=(rgb,processed)
                    rgb,processed=bank[key]
                    items[m]=dict(method=m,selection=selection,rgb_sha256=rgb,processed=processed,
                        offline_source=str(sp),offline_source_sha256=study.sha(sp),
                        identity=dict(source=identity,rgb_sha256=rgb,processed=processed,
                            model_hashes=_model_hashes(source_spec),generation=dict(seed=2027,max_new_tokens=16,do_sample=False)))
                groups=input_groups(items)
                parent_ref=_reference(*candidates[0],items[method.PARENT],row,source_spec,profile)
                if parent_ref is None: raise ValueError('Parent replay model input differs')
                for group in groups:
                    group['reference']=None
                    for path,old in candidates:
                        reference=_reference(path,old,items[group['representative']],row,source_spec,profile)
                        if reference is not None: group['reference']=reference; break
                    group.pop('identity')
                plan=dict(question_id=q,profile=profile,items=items,groups=groups,
                          prepare_binding_sha256=binding_sha,
                          parent_group=dict(representative=method.PARENT,methods=[method.PARENT],reference=parent_ref),
                          duration_seconds=float(Fraction(items[method.PARENT]['selection']['video']['duration_fraction'])))
                study.freeze(dest,plan)
            new[q]=[g['representative'] for g in plan['groups'] if g['reference'] is None]
            durations[q]=plan['duration_seconds'];exports[str(dest.relative_to(run))]=study.sha(dest)
            if ordinal%5==0:
                from videoqa_full.state import durable
                durable(run/'prepare_status.json',dict(status='running',completed=ordinal+1,total=200,
                    elapsed_seconds=time.perf_counter()-began,pid=os.getpid(),sid=os.getsid(0),utc=study.utc()))
        # 步骤2同输入引用及新调用分区先冻结，再加载语言模型；两个配置合计不得超过800。
        cap=sum(map(len,new.values()))
        if cap>400: raise ValueError('Per-profile call cap exceeded')
        generation=study.read(histories(profile,data[0]['question_id'])[0])['answer']['observed']['model_generation_config']
        spec=dict(profile=profile,model=source_spec['model'],frame_budget=k,methods=list(method.CONTROLS),
            question_ids=[r['question_id'] for r in data],pilot_question_ids=[r['question_id'] for r in data[:5]],
            max_new_calls=cap,new_groups=new,input_exports=exports,durations=durations,
            allowed_gpus=allowed_gpus,max_gpus=2,model_files=model_files,video_stats=video_stats,
            environment=source_spec['environment'],framework=source_spec['framework'],
            historical_generation_config=generation,max_new_tokens=16,seed=2027,
            parent_protocol=str(src),parent_protocol_sha256=study.sha(src),
            offline_summary_sha256=study.sha(root/'offline_summary.json'),
            prepare_binding_sha256=binding_sha,code_sha256=code)
        study.freeze(run/'protocol.json',spec); materialize_imports(run,spec)
        return spec


def verify(run):
    """接口：恢复核对代码、来源、输入、配置和额度；本函数不加载模型。"""
    from videoqa_full.prepare import environment
    from videoqa_lmms_bridge.common import framework_identity
    run=Path(run); spec=study.read(run/'protocol.json'); root=run.parents[1]
    if (len(spec['question_ids'])!=200 or len(set(spec['question_ids']))!=200
            or spec['question_ids']!=[r['question_id'] for r in study.development_rows()]
            or spec['methods']!=list(method.CONTROLS) or spec['frame_budget']!=study.budget(spec['profile'])
            or spec['max_new_calls']!=sum(map(len,spec['new_groups'].values())) or spec['max_new_calls']>400):
        raise ValueError('Frozen profile/population/call budget differs')
    if (study.sha(root/'offline_summary.json')!=spec['offline_summary_sha256']
            or study.sha(spec['parent_protocol'])!=spec['parent_protocol_sha256']
            or environment(study.PYTHONS[spec['profile']])!=spec['environment'] or framework_identity()!=spec['framework']):
        raise ValueError('Frozen input/environment identity differs')
    for p,h in spec['code_sha256'].items():
        if study.sha(study.ROOT/p)!=h: raise ValueError('Execution code changed: '+p)
    for p,h in spec['input_exports'].items():
        if study.sha(run/p)!=h: raise ValueError('Frozen model input changed')
    if study.sha(run/'prepare_binding.json')!=spec['prepare_binding_sha256']: raise ValueError('Preparation binding changed')
    for old in spec['video_stats'].values():
        stat=Path(old['path']).stat()
        if stat.st_size!=old['bytes'] or stat.st_mtime_ns!=old['mtime_ns']: raise ValueError('Source video changed after freeze')
    for p,old in spec['model_files'].items():
        stat=Path(p).stat()
        if stat.st_size!=old['bytes'] or stat.st_mtime_ns!=old['mtime_ns']:
            raise ValueError('Frozen model files changed')
    return spec


def result(run,row,m,answer,gpu,kind,representative,origins=None,timings=None):
    """接口：原始返回保存后才调用参考计分；引用结果不伪造独立推理成本。"""
    from videoqa_lmms_bridge.common import load_task,task_doc
    run=Path(run); plan=study.read(run/'inputs'/(row['question_id']+'.json')); item=plan['items'][m]
    metric=load_task().videomme_process_results(task_doc(row),[answer['raw_output']])['videomme_perception_score']
    if metric!=answer['framework_metric']: raise ValueError('Framework raw-output scoring differs')
    return dict(status='completed',profile=plan['profile'],model=study.read(run/'protocol.json')['model'],
        method=m,**{k:row[k] for k in ('question_id','video_id','question','options','stratum')},
        selection=item['selection'],rgb_sha256=item['rgb_sha256'],answer=answer,gpu=gpu,
        correct=metric['answer']==metric['pred_answer'],protocol_sha256=study.sha(run/'protocol.json'),
        timing_kind=kind,representative=representative,reference_sources=origins or {},timings=timings or {})


def materialize_imports(run,spec):
    """接口：只读导入已冻结来源，保持引用链并拒绝来源漂移。"""
    run=Path(run); bank={r['question_id']:r for r in study.development_rows()}
    for q in spec['question_ids']:
        plan=study.read(run/'inputs'/(q+'.json'))
        for group in [plan['parent_group'],*plan['groups']]:
            reference=group['reference']
            if reference is None: continue
            for p,h in reference['sources'].items():
                if study.sha(p)!=h: raise ValueError('Imported physical source changed')
            old=study.read(reference['path'])
            for m in group['methods']:
                study.freeze(run/'results'/m/(q+'.json'),result(run,bank[q],m,old['answer'],old['gpu'],
                    'historical_input_verified_import',group['representative'],reference['sources']))


def recover_group(run,row,group,spec):
    """接口：已返回原文只恢复后处理；未确认返回的已启动调用禁止重新生成。"""
    from videoqa_full.state import durable
    run=Path(run); q=row['question_id']; m=group['representative']; cp=run/'calls'/m/(q+'.json')
    if not cp.exists(): return False
    call=study.read(cp); rp=run/'raw_generations'/m/(q+'.json')
    if not rp.exists(): raise ValueError('Started call without confirmed raw return; pause')
    raw=study.read(rp)
    if (call['protocol_sha256']!=study.sha(run/'protocol.json') or raw['identity']!=dict(
            question_id=q,method=m,protocol_sha256=call['protocol_sha256'],gpu=call['gpu'])
            or raw['generation_state']!=dict(started=True,returned=True)):
        raise ValueError('Uncertain generation or changed call fingerprint')
    if call['status']!='completed':
        if spec['profile']=='qwen32':
            from videoqa_budget32.observe import result_from_raw
            answer=result_from_raw(raw,row)
        else:
            from videoqa_generalization.llava_observe import result_from_raw
            value=result_from_raw(raw,row); answer=dict(value['answer'],observed=value['observed'])
        call.update(status='completed',answer=answer,generation_state=raw['generation_state'],recovered_postprocessing=True)
        durable(cp,call)
    for m2 in group['methods']:
        dest=run/'results'/m2/(q+'.json')
        if not dest.exists(): study.freeze(dest,result(run,row,m2,call['answer'],call['gpu'],
            'returned_raw_postprocessing' if m2==m else 'same_run_input_shared',m))
    return True


def validate_call_population(run,spec,complete=False):
    """接口：生成前拒绝未授权槽位及超额账本；完成时要求恰好覆盖冻结物理输入组。"""
    run=Path(run); expected={(m,q) for q,methods in spec['new_groups'].items() for m in methods}
    if len(expected)!=spec['max_new_calls']: raise ValueError('Duplicate frozen call slots')
    if any(m not in method.CONTROLS for m,q in expected): raise ValueError('Unregistered physical control')
    for folder in ('calls','raw_generations'):
        actual={(p.parent.name,p.stem) for p in (run/folder).glob('*/*.json')}
        if not actual<=expected or (complete and actual!=expected): raise ValueError('Physical call population differs')
    return expected


def process_question(run,row,backend,adapter,gpu,spec,stop):
    """接口：同题G/W同卡同进程、交替先后；已开始调用不得因其他方法共享而重发。"""
    from videoqa_generalization.component_ablation_runtime import cpu_identity
    from videoqa_budget32.frames import FrameProvider
    from videoqa_full.state import exclusive,invoke_once
    run=Path(run); q=row['question_id']; plan=study.read(run/'inputs'/(q+'.json'))
    order=list(method.CONTROLS); shift=spec['question_ids'].index(q)%2; order=order[shift:]+order[:shift]
    groups=sorted(plan['groups'],key=lambda g:min(order.index(m) for m in g['methods']))
    for group in groups:
        if stop.is_set(): return
        if group['reference'] is not None or recover_group(run,row,group,spec): continue
        m=group['representative']
        if m not in spec['new_groups'][q]: raise ValueError('Call absent from frozen budget')
        with exclusive(run/'question_locks'/m/(q+'.lock')):
            item=plan['items'][m]; began=time.perf_counter(); images,rgb=FrameProvider(item['selection']).decode()
            try:
                tools=(backend.processor,backend.tokenizer if spec['profile']=='llava8' else None)
                if rgb!=item['rgb_sha256'] or cpu_identity(spec['profile'],tools,images,row)!=item['processed']:
                    raise ValueError('Actual input differs before generation')
                decoded=time.perf_counter(); identity=dict(question_id=q,method=m,protocol_sha256=study.sha(run/'protocol.json'),gpu=gpu)
                def invoke(state):
                    path=run/'raw_generations'/m/(q+'.json')
                    if spec['profile']=='qwen32':
                        from videoqa_budget32.observe import observe
                        return observe(backend,adapter,images,row,state,path,identity)
                    from videoqa_generalization.llava_observe import observe
                    value=observe(backend,adapter,images,item['selection'],row,state,path,identity)
                    return dict(value['answer'],observed=value['observed'])
                answer,_,checkpoint=invoke_once(run,q,m,identity['protocol_sha256'],gpu,invoke)
            finally:
                for im in images: im.close()
            for m2 in group['methods']:
                study.freeze(run/'results'/m2/(q+'.json'),result(run,row,m2,answer,gpu,
                    'new_physical_call' if m2==m else 'same_run_input_shared',m,
                    timings=dict(final_decode_identity_seconds=decoded-began,
                        input_to_persisted_answer_seconds=time.perf_counter()-began,checkpoint_seconds=checkpoint) if m2==m else {}))


def validate_result(run,row,m,spec):
    """接口：真实像素、提示、生成及原始计分独立复核；解析失败保留为错误。"""
    from videoqa_generalization.component_ablation_runtime import processed_matches
    from videoqa_lmms_bridge.common import load_task,task_doc
    run=Path(run); q=row['question_id']; value=study.read(run/'results'/m/(q+'.json'))
    item=study.read(run/'inputs'/(q+'.json'))['items'][m]; observed=value['answer']['observed']
    if (value['method']!=m or value['protocol_sha256']!=study.sha(run/'protocol.json')
            or value['selection']!=item['selection'] or value['rgb_sha256']!=item['rgb_sha256']
            or not processed_matches(spec['profile'],item['processed'],observed)
            or any(value[k]!=row[k] for k in ('question_id','video_id','question','options','stratum'))):
        raise ValueError('Result actual input/task identity differs')
    if (observed['generate_calls']!=1 or observed['prefill_calls']!=1
            or not 1<=len(observed['generated_token_ids'])<=16
            or observed['model_generation_config']!=spec['historical_generation_config']):
        raise ValueError('Observed generation differs')
    n=len(item['selection']['selected_frames'])
    if spec['profile']=='llava8' and (observed['visual_tokens']!=n*210 or observed['max_new_tokens']!=16):
        raise ValueError('LLaVA visual/token budget differs')
    if spec['profile']=='qwen32' and (observed['generation_kwargs']['max_new_tokens']!=16 or observed['encoding_fps']!=2):
        raise ValueError('Qwen video/generation protocol differs')
    metric=load_task().videomme_process_results(task_doc(row),[value['answer']['raw_output']])['videomme_perception_score']
    if value['answer']['framework_metric']!=metric or value['correct']!=(metric['answer']==metric['pred_answer']):
        raise ValueError('Independent raw scoring differs')
    physical_sources(run/'results'/m/(q+'.json'),value)
    return value


def worker(run,gpu,jobs,events,stop):
    """接口：一卡长期后端；每模型每卡两次合成，总计最多8次，不静默重新预热。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    run=Path(run); (run/'logs').mkdir(parents=True,exist_ok=True); active=None
    with (run/'logs'/('gpu_'+gpu+'.log')).open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import torch
            from PIL import Image
            from videoqa_full.state import durable
            from videoqa_lmms_bridge.bridge import synthetic_row
            torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(2027)
            spec=study.read(run/'protocol.json'); began=time.perf_counter()
            if spec['profile']=='llava8':
                from videoqa_runtime.llava_backend import LlavaBackend
                from videoqa_generalization.llava_budget8 import adapter_for
                backend=LlavaBackend(); adapter=adapter_for(backend)
            else:
                from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
                from videoqa_budget32.qwen_adapter import FrozenInputQwen
                backend=QwenFA2VideoBackend(); adapter=FrozenInputQwen(backend)
            if backend.model.generation_config.to_dict()!=spec['historical_generation_config']:
                raise ValueError('Actual backend defaults differ')
            cases=[(1,(320,240)),(8,(672,896))] if spec['profile']=='llava8' else [(11,(320,240)),(32,(672,896))]
            for i,(n,size) in enumerate(cases):
                path=run/'synthetic'/gpu/(str(i)+'.json')
                if path.exists():
                    old=study.read(path)
                    if old['status']!='completed' or old['source_frames']!=n or old['gpu']!=gpu:
                        raise ValueError('Uncertain synthetic call; repetition forbidden')
                    continue
                durable(path,dict(status='started',gpu=gpu,source_frames=n,utc=study.utc()))
                images=[Image.new('RGB',size,(30,j*7%256,150)) for j in range(n)]
                try:
                    if spec['profile']=='qwen32':
                        from videoqa_budget32.observe import observe
                        answer=observe(backend,adapter,images,synthetic_row(n),{})
                    else:
                        from videoqa_generalization.llava_observe import observe
                        selection=dict(video=dict(path='synthetic.mp4',duration_seconds=float(n)),
                                       selected_frames=[dict(timestamp_seconds=float(j)) for j in range(n)])
                        out=observe(backend,adapter,images,selection,synthetic_row(n),{})
                        answer=dict(out['answer'],observed=out['observed'])
                    durable(path,dict(status='completed',gpu=gpu,source_frames=n,answer=answer,utc=study.utc()))
                finally:
                    for im in images: im.close()
            study.freeze(run/'workers'/('gpu_'+gpu)/('pid_'+str(os.getpid())+'.json'),dict(gpu=gpu,pid=os.getpid(),sid=os.getsid(0),
                load_and_synthetic_seconds=time.perf_counter()-began,load_report=backend.load_report))
            events.put(dict(type='ready',gpu=gpu))
            while not stop.is_set():
                row=jobs.get()
                if row is None: break
                active=row['question_id']; events.put(dict(type='active',gpu=gpu,question_id=active))
                process_question(run,row,backend,adapter,gpu,spec,stop)
                if not stop.is_set(): events.put(dict(type='completed',gpu=gpu,question_id=active))
                active=None
        except BaseException as exc:
            stop.set(); traceback.print_exc()
            from videoqa_full.state import durable
            durable(run/'failures'/('gpu_'+gpu+'.json'),dict(question_id=active,error=repr(exc),utc=study.utc()))
            events.put(dict(type='failed',gpu=gpu,error=repr(exc)))


def audit(run):
    """接口：完成态只校验600条结果和已冻结物理组，不加载模型或新增生成。"""
    run=Path(run); spec=verify(run); data=study.development_rows(); expected=set()
    for row in data:
        q=row['question_id']
        for m in (method.PARENT,*method.CONTROLS): validate_result(run,row,m,spec)
        expected.update((m,q) for m in spec['new_groups'][q])
    for m in (method.PARENT,*method.CONTROLS):
        if {p.stem for p in (run/'results'/m).glob('*.json')}!=set(spec['question_ids']): raise ValueError('Extra/missing method results')
    validate_call_population(run,spec,complete=True)
    calls=[study.read(run/'calls'/m/(q+'.json')) for m,q in expected]
    if any(v['status']!='completed' for v in calls) or len({v['call_id'] for v in calls})!=len(calls):
        raise ValueError('Duplicate/unresolved call')
    synthetic=list((run/'synthetic').glob('*/*.json'))
    if len(synthetic)>4 or any(study.read(p)['status']!='completed' for p in synthetic): raise ValueError('Synthetic budget/state differs')
    value=dict(status='passed',profile=spec['profile'],method_results=600,new_real_qa=len(calls),synthetic_qa=len(synthetic),
        new_blip_calls=0,new_visual_calls=0,protocol_sha256=study.sha(run/'protocol.json'))
    study.freeze(run/'completion_audit.json',value)
    return value


def execute(run):
    """接口：后台自动先行验收、时长调度和停止；部分题始终沿用原物理卡。"""
    from videoqa_full.state import durable,exclusive
    from videoqa_runtime.baseline_run import available_gpus
    run=Path(run)
    with exclusive(run/'execute.lock'):
        spec=verify(run); data=study.development_rows(); validate_call_population(run,spec); materialize_imports(run,spec)
        for row in data:
            for group in study.read(run/'inputs'/(row['question_id']+'.json'))['groups']:
                if group['reference'] is None: recover_group(run,row,group,spec)
        pending=[r for r in data if any(not (run/'results'/m/(r['question_id']+'.json')).exists() for m in method.CONTROLS)]
        if not pending: return audit(run)
        assignment={}
        for path in (run/'calls').glob('*/*.json'):
            call=study.read(path); q=call['question_id']; gpu=call['gpu']
            if q in assignment and assignment[q]!=gpu: raise ValueError('Question moved physical GPU')
            assignment[q]=gpu
        allocation=run/'gpu_allocation.json'
        # 步骤1首次只选空闲卡，最多两张；恢复固定原集合，原卡不可用就等待。
        while True:
            try: free=set(available_gpus(spec['allowed_gpus'])[0])
            except RuntimeError: free=set()
            gpus=study.read(allocation)['gpus'] if allocation.exists() else sorted(free)[:2]
            if gpus and set(gpus)<=free and set(assignment.values())<=set(gpus): break
            durable(run/'status.json',dict(status='waiting_for_idle_gpus',required=gpus,utc=study.utc()))
            time.sleep(10)
        study.freeze(allocation,dict(gpus=gpus))
        ctx=mp.get_context('spawn'); events=ctx.Queue(); stop=ctx.Event(); jobs={g:ctx.Queue() for g in gpus}
        processes={g:ctx.Process(target=worker,args=(str(run),g,jobs[g],events,stop)) for g in gpus}
        for p in processes.values(): p.start()
        pilot=[r for r in pending if r['question_id'] in spec['pilot_question_ids']]
        remaining=sorted([r for r in pending if r['question_id'] not in spec['pilot_question_ids']],
                         key=lambda r:(-spec['durations'][r['question_id']],spec['question_ids'].index(r['question_id'])))
        waiting=pilot; full=False; idle=set(); active={}; done=0; began=time.perf_counter(); ready=set()
        try:
            # 步骤2先行全部验收后才领取剩余题；每卡最多一个活动任务。
            while done<len(pending):
                try: event=events.get(timeout=5)
                except queue.Empty:
                    if any(p.exitcode is not None for p in processes.values()): raise RuntimeError('Worker exited unexpectedly')
                    event=None
                if event:
                    gpu=event['gpu']
                    if event['type']=='failed': raise RuntimeError(str(event))
                    if event['type']=='ready': ready.add(gpu); idle.add(gpu)
                    if event['type']=='active': active[gpu]=event['question_id']
                    if event['type']=='completed': done+=1; active.pop(gpu,None); idle.add(gpu)
                if len(ready)==len(gpus):
                    if not full and not waiting and not active:
                        for row in data[:5]:
                            for m in (method.PARENT,*method.CONTROLS): validate_result(run,row,m,spec)
                        study.freeze(run/'pilot_acceptance.json',dict(status='passed',question_ids=spec['pilot_question_ids']))
                        full=True; waiting=remaining
                    for gpu in sorted(idle):
                        i=next((i for i,r in enumerate(waiting) if assignment.get(r['question_id'],gpu)==gpu),None)
                        if i is not None:
                            row=waiting.pop(i); jobs[gpu].put(row); active[gpu]=row['question_id']; idle.remove(gpu)
                elapsed=time.perf_counter()-began
                durable(run/'heartbeat.json',dict(status='answering',pid=os.getpid(),sid=os.getsid(0),
                    completed=done,total=len(pending),active=active,elapsed_seconds=elapsed,
                    eta_seconds=None if not done else elapsed/done*(len(pending)-done),utc=study.utc()))
            for q in jobs.values(): q.put(None)
            for p in processes.values(): p.join(timeout=60)
            if any(p.is_alive() or p.exitcode!=0 for p in processes.values()): raise RuntimeError('Unclean worker completion')
        except BaseException:
            stop.set()
            for q in jobs.values(): q.put(None)
            for p in processes.values(): p.join(timeout=60)
            durable(run/'status.json',dict(status='paused',active=active,utc=study.utc())); raise
        study.freeze(run/'execution.json',dict(status='completed',gpus=gpus,wall_seconds=time.perf_counter()-began))
        return audit(run)
