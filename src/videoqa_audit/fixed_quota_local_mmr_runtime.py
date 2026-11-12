"""固定区域局部MMR的输入身份、有限问答与后台恢复；旧运行只读。"""
import contextlib
from fractions import Fraction
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

from . import fixed_quota_local_mmr as method
from . import fixed_quota_local_mmr_run as study
from .selection_structure_runtime import (historical_context,physical_sources,_model_hashes,
    _reference,result,materialize_imports,recover_group,validate_result)

_TOOLS=None


def histories(profile,q):
    """接口：固定P→G→W，不能根据历史正确性选择引用。"""
    return [study.PREVIOUS/'qa'/profile/'results'/m/(q+'.json')
            for m in (method.PARENT,method.GLOBAL,method.WFS)]


def _init_prepare(profile):
    """接口：CPU子进程只加载官方处理器和Tokenizer，每进程两线程。"""
    global _TOOLS
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import torch
    from videoqa_generalization.component_ablation_runtime import cpu_tools
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    _TOOLS=cpu_tools(profile)


def prepare_one(job):
    """接口：一题CPU身份冻结；相同旧输入只复用已验收且源码相同的完整缓存。"""
    from videoqa_budget32.frames import FrameProvider,validate_frames
    from videoqa_generalization.component_ablation_runtime import cpu_identity
    root,profile,row,binding_sha=job;root=Path(root);run=root/'qa'/profile;q=row['question_id']
    dest=run/'inputs'/(q+'.json')
    if dest.exists():
        value=study.read(dest)
        if value['prepare_binding_sha256']!=binding_sha or value['question_id']!=q or value['profile']!=profile:
            raise ValueError('Cached input binding differs')
        return q,study.sha(dest)
    oldrun=study.PREVIOUS/'qa'/profile; source_spec=study.read(oldrun/'protocol.json')
    oldpath=oldrun/'inputs'/(q+'.json')
    if study.sha(oldpath)!=source_spec['input_exports'][str(oldpath.relative_to(oldrun))]:
        raise ValueError('Prior complete input cache changed')
    oldplan=study.read(oldpath); source= root/'selections'/profile/method.CONTROL/(q+'.json')
    selection=study.read(source)['selection'];validate_frames(selection,study.budget(profile))
    identity=study.source_identity(selection); began=time.perf_counter()
    # 步骤1源身份相同才复用旧实际像素、Token缓存；否则精确解码和官方预处理。
    matched=[v for v in oldplan['items'].values() if v['identity']['source']==identity]
    if matched:
        if any(v['processed']!=matched[0]['processed'] or v['rgb_sha256']!=matched[0]['rgb_sha256'] for v in matched):
            raise ValueError('Conflicting historical input cache')
        rgb,processed=matched[0]['rgb_sha256'],matched[0]['processed']; cache=True
    else:
        images,rgb=FrameProvider(selection).decode()
        try: processed=cpu_identity(profile,_TOOLS,images,row)
        finally:
            for im in images: im.close()
        cache=False
    item=dict(method=method.CONTROL,selection=selection,rgb_sha256=rgb,processed=processed,
        offline_source=str(source),offline_source_sha256=study.sha(source),
        identity=dict(source=identity,rgb_sha256=rgb,processed=processed,
            model_hashes=_model_hashes(source_spec),generation=dict(seed=2027,max_new_tokens=16,do_sample=False)))
    # 步骤2检查所有匹配来源是否冲突，再按固定优先级选引用，不能遇到第一项就漏查冲突。
    references=[]; predictions=[]
    for p in histories(profile,q):
        old=study.read(p); ref=_reference(p,old,item,row,source_spec,profile)
        if ref is not None:
            references.append(ref);predictions.append(old['answer']['parsed_answer'])
    if len(set(predictions))>1: raise ValueError('Same-input historical prediction conflict')
    parent_path=histories(profile,q)[0];parent_old=study.read(parent_path)
    parent_item=oldplan['items'][method.PARENT]
    parent_ref=_reference(parent_path,parent_old,parent_item,row,source_spec,profile)
    if parent_ref is None: raise ValueError('Parent actual input cache differs')
    plan=dict(question_id=q,profile=profile,prepare_binding_sha256=binding_sha,
        items={method.PARENT:parent_item,method.CONTROL:item},
        groups=[dict(representative=method.CONTROL,methods=[method.CONTROL],reference=references[0] if references else None)],
        parent_group=dict(representative=method.PARENT,methods=[method.PARENT],reference=parent_ref),
        duration_seconds=float(Fraction(selection['video']['duration_fraction'])),
        complete_input_cache_reused=cache,input_preparation_seconds=time.perf_counter()-began,
        previous_input_sha256=study.sha(oldpath))
    study.freeze(dest,plan);return q,study.sha(dest)


def prepare(root,profile,allowed_gpus):
    """接口：4进程准备200题，先冻结全部输入和历史来源，再允许生成。"""
    from videoqa_full.state import exclusive,durable
    from videoqa_full.prepare import environment
    from videoqa_lmms_bridge.common import framework_identity
    from videoqa_generalization import component_ablation_runtime as prior
    from .selection_structure_run import FILES as OLD_FILES
    root=Path(root);run=root/'qa'/profile
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists(): return verify(run)
        study.verify_offline(root); source=study.PREVIOUS/'qa'/profile/'protocol.json';old=study.read(source)
        if environment(study.PYTHONS[profile])!=old['environment'] or framework_identity()!=old['framework']:
            raise ValueError('Frozen environment/framework changed')
        for p,h in old['code_sha256'].items():
            if study.sha(study.ROOT/p)!=h: raise ValueError('Historical input-processing code changed: '+p)
        model_files=prior.copy_file_stats(old['model_files']); video_stats={}
        for vid,item in old['video_stats'].items():
            st=Path(item['path']).stat()
            if st.st_size!=item['bytes'] or st.st_mtime_ns!=item['mtime_ns']:
                raise ValueError('Source video changed after prior acceptance')
            video_stats[vid]=item
        code={p:study.sha(study.ROOT/p) for p in set((*study.FILES,*OLD_FILES,*prior.QA_FILES))}
        binding=dict(profile=profile,allowed_gpus=allowed_gpus,code_sha256=code,
            source_protocol_sha256=study.sha(source),offline_sha256=study.sha(root/'offline_summary.json'),
            dev_sha256=study.sha(study.DEV),model_hashes=_model_hashes(old),video_stats=video_stats)
        study.freeze(run/'prepare_binding.json',binding);h=study.sha(run/'prepare_binding.json');data=study.development_rows()
        # 步骤1仅并行CPU媒体处理，有限队列防止长视频预取占满内存。
        began=time.perf_counter();done=0;exports={}
        with ProcessPoolExecutor(max_workers=4,initializer=_init_prepare,initargs=(profile,)) as ex:
            jobs=[(str(root),profile,row,h) for row in data]
            for q,ph in study.bounded_results(ex,prepare_one,jobs,limit=8):
                done+=1;exports['inputs/'+q+'.json']=ph
                durable(run/'prepare_status.json',dict(status='running',completed=done,total=200,
                    elapsed_seconds=time.perf_counter()-began,pid=os.getpid(),sid=os.getsid(0),utc=study.utc()))
        new={};durations={}
        for row in data:
            q=row['question_id']; plan=study.read(run/'inputs'/(q+'.json'))
            new[q]=[method.CONTROL] if plan['groups'][0]['reference'] is None else []
            durations[q]=plan['duration_seconds']
        cap=sum(map(len,new.values()))
        if cap>200: raise ValueError('Per-profile call cap exceeded')
        spec=dict(profile=profile,model=old['model'],frame_budget=study.budget(profile),methods=list(method.CONTROLS),
            question_ids=[r['question_id'] for r in data],pilot_question_ids=[r['question_id'] for r in data[:5]],
            max_new_calls=cap,new_groups=new,input_exports=exports,durations=durations,
            allowed_gpus=allowed_gpus,max_gpus=4,model_files=model_files,video_stats=video_stats,
            environment=old['environment'],framework=old['framework'],
            historical_generation_config=old['historical_generation_config'],max_new_tokens=16,seed=2027,
            parent_protocol=str(source),parent_protocol_sha256=study.sha(source),
            offline_summary_sha256=study.sha(root/'offline_summary.json'),prepare_binding_sha256=h,code_sha256=code,
            preparation_wall_seconds=time.perf_counter()-began)
        study.freeze(run/'protocol.json',spec);materialize_imports(run,spec);return spec


def verify(run):
    """接口：恢复只检查协议、源文件及完整输入身份，不加载模型或改写结果。"""
    from videoqa_full.prepare import environment
    from videoqa_lmms_bridge.common import framework_identity
    run=Path(run);spec=study.read(run/'protocol.json')
    if (spec['methods']!=list(method.CONTROLS) or spec['question_ids']!=[r['question_id'] for r in study.development_rows()]
            or spec['frame_budget']!=study.budget(spec['profile'])
            or spec['max_new_calls']!=sum(map(len,spec['new_groups'].values())) or spec['max_new_calls']>200):
        raise ValueError('Frozen population/budget differs')
    if (study.sha(run.parents[1]/'offline_summary.json')!=spec['offline_summary_sha256']
            or study.sha(spec['parent_protocol'])!=spec['parent_protocol_sha256']
            or study.sha(run/'prepare_binding.json')!=spec['prepare_binding_sha256']
            or environment(study.PYTHONS[spec['profile']])!=spec['environment'] or framework_identity()!=spec['framework']):
        raise ValueError('Frozen protocol/environment differs')
    for p,h in spec['code_sha256'].items():
        if study.sha(study.ROOT/p)!=h: raise ValueError('Execution code changed: '+p)
    for p,h in spec['input_exports'].items():
        if study.sha(run/p)!=h: raise ValueError('Actual input cache changed')
    for item in [*spec['video_stats'].values(),*spec['model_files'].values()]:
        st=Path(item['path']).stat() if 'path' in item else None
        if st is not None and (st.st_size!=item['bytes'] or st.st_mtime_ns!=item['mtime_ns']):
            raise ValueError('Frozen video changed')
    for p,item in spec['model_files'].items():
        st=Path(p).stat()
        if st.st_size!=item['bytes'] or st.st_mtime_ns!=item['mtime_ns']: raise ValueError('Frozen weights changed')
    return spec


def validate_call_population(run,spec,complete=False):
    """接口：唯一H调用必须来自事前冻结额度；完整恢复禁止新增调用槽位。"""
    run=Path(run);expected={(m,q) for q,ms in spec['new_groups'].items() for m in ms}
    if len(expected)!=spec['max_new_calls'] or any(m!=method.CONTROL for m,q in expected):
        raise ValueError('Unregistered or duplicate call slots')
    for folder in ('calls','raw_generations'):
        actual={(p.parent.name,p.stem) for p in (run/folder).glob('*/*.json')}
        if not actual<=expected or (complete and actual!=expected): raise ValueError('Physical call population differs')
    return expected


def process_question(run,row,backend,adapter,gpu,spec,stop):
    """接口：单题H同卡单进程生成；已返回原文仅恢复后处理。"""
    from videoqa_generalization.component_ablation_runtime import cpu_identity
    from videoqa_budget32.frames import FrameProvider
    from videoqa_full.state import exclusive,invoke_once
    run=Path(run);q=row['question_id'];plan=study.read(run/'inputs'/(q+'.json'));group=plan['groups'][0]
    if stop.is_set() or group['reference'] is not None or recover_group(run,row,group,spec): return
    m=method.CONTROL
    if spec['new_groups'][q]!=[m]: raise ValueError('Call absent from frozen plan')
    with exclusive(run/'question_locks'/m/(q+'.lock')):
        item=plan['items'][m];began=time.perf_counter();images,rgb=FrameProvider(item['selection']).decode()
        try:
            tools=(backend.processor,backend.tokenizer if spec['profile']=='llava8' else None)
            if rgb!=item['rgb_sha256'] or cpu_identity(spec['profile'],tools,images,row)!=item['processed']:
                raise ValueError('Actual input differs before generation')
            decoded=time.perf_counter();identity=dict(question_id=q,method=m,protocol_sha256=study.sha(run/'protocol.json'),gpu=gpu)
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
        study.freeze(run/'results'/m/(q+'.json'),result(run,row,m,answer,gpu,'new_physical_call',m,
            timings=dict(final_decode_identity_seconds=decoded-began,
                input_to_persisted_answer_seconds=time.perf_counter()-began,checkpoint_seconds=checkpoint)))


def audit(run):
    """接口：完成态验收400条H/P结果及物理账本；不加载模型。"""
    run=Path(run);spec=verify(run);data=study.development_rows()
    for row in data:
        for m in (method.PARENT,method.CONTROL): validate_result(run,row,m,spec)
    for m in (method.PARENT,method.CONTROL):
        if {p.stem for p in (run/'results'/m).glob('*.json')}!=set(spec['question_ids']):
            raise ValueError('Extra/missing result identities')
    expected=validate_call_population(run,spec,complete=True)
    calls=[study.read(run/'calls'/m/(q+'.json')) for m,q in expected]
    if any(c['status']!='completed' for c in calls) or len({c['call_id'] for c in calls})!=len(calls):
        raise ValueError('Duplicate or unresolved physical generation')
    synthetic=list((run/'synthetic').glob('*/*.json'))
    if len(synthetic)!=4 or any(study.read(p)['status']!='completed' for p in synthetic):
        raise ValueError('Synthetic population/state differs')
    value=dict(status='passed',profile=spec['profile'],method_results=400,new_real_qa=len(calls),synthetic_qa=4,
        new_blip_calls=0,new_visual_calls=0,protocol_sha256=study.sha(run/'protocol.json'))
    study.freeze(run/'completion_audit.json',value);return value


def worker(run,gpu,jobs,events,stop):
    """接口：一卡长期后端；每模型固定四次合成，按空闲卡顺序轮流分配，不静默重新预热。"""
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
            allocation=study.read(run/'gpu_allocation.json')['gpus']
            cases=[(1,(320,240)),(7,(240,320)),(8,(896,672)),(8,(672,896))] if spec['profile']=='llava8' else [(1,(320,240)),(11,(240,320)),(31,(896,672)),(32,(672,896))]
            for i,(n,size) in enumerate(cases):
                if allocation[i%len(allocation)]!=gpu: continue
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
        # 步骤1首次只选空闲卡，最多四张；恢复固定原集合，原卡不可用就等待。
        while True:
            try: free=set(available_gpus(spec['allowed_gpus'])[0])
            except RuntimeError: free=set()
            gpus=study.read(allocation)['gpus'] if allocation.exists() else sorted(free)[:4]
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
