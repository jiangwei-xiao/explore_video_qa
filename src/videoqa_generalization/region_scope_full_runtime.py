"""exp27全量问答执行：单方法、历史引用、实际输入观测和严格一次调用。"""
from pathlib import Path
import contextlib
import multiprocessing as mp
import os
import queue
import time
import traceback
from videoqa_runtime.common import ROOT, read_json, sha256, offline_environment
from videoqa_full.state import durable, exclusive, rows, utc, invoke_once
from videoqa_full.prepare import environment
from videoqa_budget32.frames import FrameProvider, validate_frames
from videoqa_lmms_bridge.common import framework_identity, load_task, task_doc
from .component_ablation_runtime import processed_matches, cpu_identity
from . import region_scope_full as study
from .region_scope_full import verify

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
                if not target.exists():durable(target,result_from_answer(run,bank[q],c,old['answer'],ref['gpu'],
                    'historical_input_verified_import',ref['sources'],group['representative']))

def validate_result(run,row,control,spec):
    """接口：审计方法结果与物理调用/引用；解析失败计错，不重新回答。"""
    # 步骤1核验协议和实际源身份；步骤2核对官方像素/Token与原始返回的计分。
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
        ref=plan['groups'][0]['reference']
        if ref is None or result['reference_sources']!=ref['sources'] or answer!=read_json(ref['path'])['answer'] or (run/'calls'/method/(q+'.json')).exists():raise ValueError('Import lacks matching physical provenance')
        for path,h in result['reference_sources'].items():
            if sha256(path)!=h:raise ValueError('Imported raw result changed')
    else:
        if plan['groups'][0]['reference'] is not None:raise ValueError('Import consumed a new call')
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
    # 步骤1先检查账本/返回身份；步骤2只重建已返回调用的结果，不触发模型。
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
    """接口：单题单方法按冻结次序轮换，仅未调用的物理输入组进入模型。"""
    # 步骤1跳过已核验引用或恢复返回；步骤2精确解码后仅调用一次，再立即落盘。
    run=Path(run);q=row['question_id'];plan=read_json(run/'inputs'/(q+'.json'))
    shift=spec['question_ids'].index(q)%len(study.CONTROLS)
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

def worker(run,gpu,jobs,events,stop,extended):
    """接口：一卡常驻一个正式后端，每工作进程一次合成与真实调用分账，失败后停止领取任务。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    offline_environment();run=Path(run);active=None
    with (run/'logs'/('gpu_'+gpu+'.log')).open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import torch
            from PIL import Image
            from videoqa_lmms_bridge.bridge import synthetic_row
            torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(2027)
            # 步骤1固定后端与原生成配置；步骤2一次合成后循环领取真实任务。
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
            for ordinal,(n,size) in enumerate([(study.budget(spec['profile']),(896,672))]):
                path=run/'synthetic'/gpu/(str(ordinal)+'.json')
                if path.exists():
                    if read_json(path)['status']!='completed':raise ValueError('Uncertain synthetic return')
                    warm.append(read_json(path)['answer']['observed']);continue
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
    """接口：2700方法结果与冻结物理调用闭合；完成态不加载模型。"""
    # 步骤1核验2700个结果及实际来源；步骤2闭合物理调用和合成额度。
    run=Path(run);spec=verify(run);data=rows();calls=list((run/'calls').glob('*/*.json'))
    if {p.stem for p in (run/'results'/study.METHOD).glob('*.json')}!=set(spec['question_ids']):raise ValueError('Result population differs')
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
    result=dict(status='passed',profile=spec['profile'],method_results=2700,
                physical_new_calls=len(calls),synthetic_calls=len(synthetic),new_blip_calls=0,
                new_selection_visual_calls=0,protocol_sha256=sha256(run/'protocol.json'),utc=utc())
    study.freeze(run/'completion_audit.json',{k:v for k,v in result.items() if k!='utc'})
    return result

def execute(run):
    """接口：先恢复已返回答案，再固定空闲卡，自动先行验收后调度全部余题。"""
    # 步骤1恢复引用与已完整返回的后处理；无剩余任务直接只读审计。
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
                if done in (min(5,len(pending)),100):durable(run/('throughput_'+str(done)+'.json'),dict(completed=done,total=len(pending),seconds=time.perf_counter()-began,eta_seconds=(time.perf_counter()-began)/done*(len(pending)-done),utc=utc()))
                if event['question_id'] in pilot_ids:completed_pilot.add(event['question_id'])
            if len(ready)==len(gpus) and not pilot_sent:
                for r in pilot:
                    g=assigned.get(r['question_id'],gpus[0]);queues[g].put(r);idle.discard(g)
                pilot_sent=True
            if pilot_sent and completed_pilot==pilot_ids and not full:
                for q in spec['pilot_question_ids']:
                    row=next(r for r in data if r['question_id']==q)
                    for c in study.CONTROLS:validate_result(run,row,c,spec)
                study.freeze(run/'pilot_acceptance.json',dict(status='passed',question_ids=spec['pilot_question_ids'],accuracy_gate=False))
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
        for p in processes.values():p.join(timeout=30)
        if any(p.is_alive() or p.exitcode!=0 for p in processes.values()):raise RuntimeError('Worker completion not clean')
    except BaseException:
        stop.set()
        for q in queues.values():q.put(None)
        for p in processes.values():p.join(timeout=30)
        durable(run/'status.json',dict(status='paused',utc=utc()));raise
    durable(run/'execution.json',dict(status='completed',gpus=gpus,questions_with_new_calls=len(pending),
                                     wall_seconds=time.perf_counter()-began,utc=utc()))
    return audit(run)
