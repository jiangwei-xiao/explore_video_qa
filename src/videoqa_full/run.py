"""全量后台阶段协调器与长期GPU工作进程；冻结算法与新模型适配解耦。"""
import contextlib
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
import traceback
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256,offline_environment
from videoqa_runtime.protocol import parse_answer,question_text
from .state import (durable,utc,rows,digest,exclusive,stage_methods,method_order,
                    invoke_once,validate_frames,frame_identity,save_array,PYTHONS)
from .prepare import verify_runtime


def replay(selection,features,cfg):
    """通用选帧验收只依赖本次池，适用于开发题以外的所有题及短池。"""
    from videoqa_audit.density_reward_decay import select_density_decay
    from videoqa_audit.local_distance_scale import select_local_distance
    rows_=selection['candidates'];plan=selection['plan'];video=selection['video']
    _,budget=select_density_decay(rows_,features,plan,video,cfg)
    ids,trace=select_local_distance(rows_,features,plan,video,cfg,budget['region_budgets'])
    assert budget==selection['budget_trace'] and trace==selection['selection_trace']
    assert ids==selection['selected_indices']
    assert selection['selected_frames']==[rows_[i] for i in ids]
    validate_frames(selection)
    return dict(passed=True,effective_budget=budget['effective_frame_budget'])


def load_selection(run,method,qid,spec):
    """核验源选择指纹并读取；缓存只包含源位置与身份，不向后端泄露答案。"""
    path=run/'selections'/method/(qid+'.json')
    if method=='RD-1.2':
        result=read_json(run/'llava/results/RD-1.2'/(qid+'.json'))
        expected=result['selection_sha256']
    else:
        expected=spec['selection_exports'][str(path.relative_to(run))]
    assert sha256(path)==expected
    saved=read_json(path)
    assert saved['question_id']==qid and saved['method']==method
    assert saved['source_video_sha256']==spec['video_hashes'][saved['video_id']]
    validate_frames(saved['selection'])
    return saved,expected


def validate_result(run,model,method,row,spec,fingerprint,deep=False):
    """验收身份、真实输入、计分、账本与Token；恢复不加载模型、不重新生成。"""
    qid=row['question_id'];stage=run/model
    record=read_json(stage/'results'/method/(qid+'.json'))
    assert record['status']=='completed' and record['protocol_sha256']==fingerprint
    assert record['model']==model and record['method']==method
    for k in ('question_id','video_id','question','options','stratum'): assert record[k]==row[k]
    saved,h=load_selection(run,method,qid,spec)
    assert h==record['selection_sha256'] and saved['question_sha256']==digest([row['question'],row['options']])
    assert record['frame_identity']==[list(x) for x in frame_identity(saved['selection'])]
    a=record['answer'];n=len(saved['selection']['selected_frames'])
    assert len(a['generated_token_ids'])<=8 and a['parsed_answer']==parse_answer(a['raw_output'])
    assert record['reference_answer']=='ABCD'[row['answer_index']]
    assert record['correct']==(a['parsed_answer']==record['reference_answer'])
    video=saved['selection']['video'];times=[r['timestamp_seconds'] for r in saved['selection']['selected_frames']]
    if model=='llava':
        assert a['pixel_shape']==[n,3,384,384] and a['visual_tokens']==210*n
        assert a['prefill_tokens']==a['text_input_tokens']-1+210*n
        assert question_text(row['question'],row['options'],video['duration_seconds'],times) in a['prompt']
    else:
        from .qwen_backend import qa_text,padded_count,MIN_PIXELS,MAX_PIXELS
        assert qa_text(row['question'],row['options'],video['duration_seconds'],times) in a['prompt']
        assert a['source_frame_count']==n and a['encoded_frame_count']==padded_count(n)
        assert a['actual_timestamps']==times and a['encoding_fps']==2.0 and a['second_per_grid_ts']==[1.0]
        grid=a['video_grid_thw'];assert grid[0][0]==padded_count(n)//2
        assert a['visual_tokens']==int(np.prod(grid[0])//4)
        assert MIN_PIXELS<=np.prod(a['processed_size'])<=MAX_PIXELS
        assert a['generation_config']['max_new_tokens']==8 and not a['generation_config']['do_sample']
    assert a['prefill_calls']==1 and a['checked_logit_steps']>0
    call=read_json(stage/'calls'/method/(qid+'.json'))
    assert call['status']=='completed' and call['answer']==a and call['protocol_sha256']==fingerprint
    assert call['generation_state']=={'started':True,'returned':True}
    assert all(np.isfinite(v) and v>=-1e-5 for v in record['timings'].values())
    if deep and model=='llava':
        checkpoint=read_json(run/'llava/checkpoints'/(qid+'.json'))
        assert checkpoint['protocol_sha256']==fingerprint
        assert sha256(run/checkpoint['feature_file'])==checkpoint['feature_sha256']
        selection=read_json(run/checkpoint['pool_file'])
        assert sha256(run/checkpoint['pool_file'])==checkpoint['pool_sha256']
        assert selection['selected_frames']==saved['selection']['selected_frames']
        replay(selection,np.load(run/checkpoint['feature_file'],allow_pickle=False),spec['config'])
    return record


def execute_question(run,model,method,row,gpu,backend,scorer,spec,fingerprint):
    """单题核心流程：生成前检查点→精确帧→持久调用→计分；来源答案不传入backend。"""
    import torch
    from videoqa_methods.region_context import read_exact_frames
    qid=row['question_id'];stage=run/model;destination=stage/'results'/method/(qid+'.json')
    if destination.exists():
        validate_result(run,model,method,row,spec,fingerprint);return
    with exclusive(stage/'task_locks'/(qid+'.lock')):
        asset=spec['video_stats'][row['video_id']];st=(ROOT/asset['path']).stat()
        assert st.st_size==asset['bytes'] and st.st_mtime_ns==asset['mtime_ns'], 'Video identity changed'
        torch.cuda.synchronize();resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();timing={};resumed=False;counts=dict(scorer.counts) if scorer else {}
        # 步骤1：LLaVA独立从原片计算RD；Qwen只读取经过冻结的对应方法输出。
        if model=='llava':
            checkpoint=stage/'checkpoints'/(qid+'.json')
            if checkpoint.exists():
                resumed=True;cp=read_json(checkpoint);assert cp['protocol_sha256']==fingerprint
                assert sha256(run/cp['pool_file'])==cp['pool_sha256']
                assert sha256(run/cp['feature_file'])==cp['feature_sha256']
                selection=read_json(run/cp['pool_file']);features=np.load(run/cp['feature_file'],allow_pickle=False)
                timing['restored_selection_seconds']=time.perf_counter()-start
                replay(selection,features,spec['config'])
            else:
                from videoqa_audit.e1_pipeline import prepare_density
                selection,features,timing=prepare_density(ROOT/'data'/row['video_relative_path'],row['question'],
                    scorer,spec['config'],run/'fine_frames'/qid)
                t=time.perf_counter();replay(selection,features,spec['config'])
                if qid in spec['historical_rd']:
                    old=ROOT/'outputs/methods/density_e1_200_20260922_r1/results/e1'/(qid+'.json')
                    assert sha256(old)==spec['historical_rd'][qid]
                    from videoqa_audit.e1_validation import verify_selection
                    verify_selection(selection,features,spec['config'],qid)
                timing['selection_validation_seconds']=time.perf_counter()-t
                t=time.perf_counter();feature=stage/'features'/(qid+'.npy');h=save_array(feature,features)
                pool=stage/'pools'/(qid+'.json');durable(pool,selection)
                cp=dict(protocol_sha256=fingerprint,feature_file=str(feature.relative_to(run)),feature_sha256=h,
                    pool_file=str(pool.relative_to(run)),pool_sha256=sha256(pool),original_timings=timing.copy())
                durable(checkpoint,cp);timing['selection_persist_seconds']=time.perf_counter()-t
            t=time.perf_counter()
            saved=dict(question_id=qid,video_id=row['video_id'],method=method,
                question_sha256=digest([row['question'],row['options']]),
                source_video_sha256=spec['video_hashes'][row['video_id']],protocol_sha256=fingerprint,
                pool_sha256=cp['pool_sha256'],selection=dict(video=selection['video'],selected_frames=selection['selected_frames']))
            selected_path=run/'selections'/method/(qid+'.json')
            if selected_path.exists():assert read_json(selected_path)==saved
            else:durable(selected_path,saved)
            selected_hash=sha256(selected_path);timing['selection_export_seconds']=time.perf_counter()-t
        else:
            saved,selected_hash=load_selection(run,method,qid,spec)
            timing['cached_selection_read_seconds']=time.perf_counter()-start
        s=saved['selection'];validate_frames(s)
        assert saved['question_sha256']==digest([row['question'],row['options']])
        # 步骤2：按精确PTS读取，不把缓存图片或来源模型预测作为新问答输入。
        t=time.perf_counter();images=read_exact_frames(s['video'],s['selected_frames'])
        timing['final_decode_seconds']=time.perf_counter()-t
        torch.cuda.synchronize();selection_end=time.perf_counter();selector_peak=torch.cuda.max_memory_allocated()
        parsed_at=[]
        def answer_once(state):
            answer=backend.answer(images,[f['timestamp_seconds'] for f in s['selected_frames']],
                s['video']['duration_seconds'],row['question'],row['options'],generation_state=state)
            torch.cuda.synchronize();parsed_at.append(time.perf_counter());return answer
        try:
            answer,reused,begin_checkpoint=invoke_once(stage,qid,method,fingerprint,gpu,answer_once)
        finally:
            for image in images:image.close()
        ended=parsed_at[0] if parsed_at else time.perf_counter()
        # 步骤3：直接墙钟终点为解析完成，最终答案账本完成落盘与结果落盘另报。
        after=time.perf_counter();component=sum(timing.values());qa=ended-selection_end
        timing.update(selection_total_seconds=selection_end-start,qa_total_seconds=qa,
            end_to_end_seconds=ended-start,stage_sum_seconds=component+qa,
            remaining_seconds=ended-start-component-qa,call_start_checkpoint_seconds=begin_checkpoint,
            answer_completion_checkpoint_seconds=after-ended)
        core_keys=('coarse_rgb_seconds','coarse_preprocess_seconds','coarse_forward_seconds','coarse_misc_seconds',
                   'region_build_seconds','fine_rgb_seconds','fine_visual_preprocess_seconds','fine_visual_forward_seconds','selection_compete_seconds')
        timing['selection_core_seconds']=sum(timing.get(k,0.) for k in core_keys)
        result=dict(status='completed',model=model,method=method,protocol_sha256=fingerprint,
            **{k:row[k] for k in ('question_id','video_id','question','options','stratum')},
            selection_sha256=selected_hash,frame_identity=frame_identity(s),source_video_sha256=saved['source_video_sha256'],
            answer=answer,reference_answer='ABCD'[row['answer_index']],correct=answer['parsed_answer']=='ABCD'[row['answer_index']],
            physical_gpu=gpu,worker_pid=os.getpid(),timings=timing,
            timing_kind='resumed_not_primary' if resumed or reused else ('raw_e2e' if model=='llava' else 'cached_selection_qa_not_raw_e2e'),
            blip_calls={k:scorer.counts[k]-counts[k] for k in counts},
            memory=dict(resident_allocated_gib=resident/2**30,selection_incremental_peak_gib=max(0,selector_peak-resident)/2**30,
                task_peak_allocated_gib=max(selector_peak/2**30,answer['peak_allocated_gib']),qa_peak_reserved_gib=answer['peak_reserved_gib']),
            completed_at=utc())
        durable(destination,result)
        validate_result(run,model,method,row,spec,fingerprint)


def worker(run,model,gpu,tasks,events,stop,fingerprint):
    """每卡长期进程；首次合成预热单列，异常发停止信号，不重复真实问答。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    offline_environment();run=Path(run);stage=run/model;spec=read_json(run/'protocol.json');active=None;scorer=None
    with (stage/'logs'/f'gpu_{gpu}.log').open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import torch
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            t=time.perf_counter()
            if model=='llava':
                from videoqa_runtime.llava_backend import LlavaBackend
                from videoqa_audit.e1_pipeline import DensityScorer
                backend=LlavaBackend();scorer=DensityScorer(read_json(ROOT/'configs/local_model_inventory.json')['models']['blip']['path'])
            else:
                from .qwen_backend import QwenVideoBackend
                backend=QwenVideoBackend()
            loading=time.perf_counter()-t;t=time.perf_counter();warmup=backend.warmup()
            blip=scorer.synthetic_acceptance() if scorer else None
            durable(stage/'workers'/f'{gpu}_{os.getpid()}.json',dict(gpu=gpu,pid=os.getpid(),load_seconds=loading,
                warmup_seconds=time.perf_counter()-t,synthetic_qa_forwards=1 if model=='llava' else 4,
                blip_acceptance=blip,qwen_acceptance=warmup,load_report=backend.load_report,
                blip_warmup_counts=dict(scorer.counts) if scorer else {}))
            events.put(dict(type='ready',gpu=gpu))
            while not stop.is_set():
                job=tasks.get()
                if job is None:break
                active=job['row']['question_id']
                for method in method_order(model,job['ordinal']):
                    if stop.is_set():break
                    events.put(dict(type='active',gpu=gpu,question_id=active,method=method))
                    execute_question(run,model,method,job['row'],gpu,backend,scorer,spec,fingerprint)
                if stop.is_set():break
                events.put(dict(type='completed',gpu=gpu,question_id=active));active=None
        except BaseException as error:
            stop.set();traceback.print_exc()
            durable(stage/'failures'/f'{gpu}_{time.time_ns()}.json',dict(gpu=gpu,question_id=active,error=repr(error),traceback=traceback.format_exc(),utc=utc()))
            events.put(dict(type='failed',gpu=gpu,question_id=active,error=repr(error)))
        finally:
            if scorer:durable(stage/'workers'/f'{gpu}_{os.getpid()}_final.json',dict(blip_counts=scorer.counts,utc=utc()))


def recovery(run,model,data,spec,fingerprint):
    """启动时检查所有固定调用ID，生成状态不确定则拒绝恢复。"""
    ids={r['question_id'] for r in data};complete=set();stage=run/model
    for method in stage_methods(model):
        calls=list((stage/'calls'/method).glob('*.json'))
        assert len(calls)<=2700
        for path in calls:
            c=read_json(path)
            assert path.stem in ids and c['method']==method and c['model']==model
            assert c['protocol_sha256']==fingerprint and c['status']=='completed', 'Uncertain call: '+str(path)
        for path in (stage/'results'/method).glob('*.json'):assert path.stem in ids
    for row in data:
        found=0
        for method in stage_methods(model):
            if (stage/'results'/method/(row['question_id']+'.json')).exists():
                validate_result(run,model,method,row,spec,fingerprint);found+=1
        if found==len(stage_methods(model)):complete.add(row['question_id'])
    return complete


def execute_stage(run,model,gpus):
    """接口：准备后单阶段执行；先行7题自动协议验收，通过后才调度余题。"""
    from videoqa_runtime.baseline_run import available_gpus
    run=Path(run);stage=run/model;stage.mkdir(parents=True,exist_ok=True)
    with exclusive(stage/'coordinator.lock'):
        spec=verify_runtime(run,model);data=rows();byid={r['question_id']:r for r in data};fingerprint=sha256(run/'protocol.json')
        for d in ('logs','workers','failures','calls','results','checkpoints','features','pools','task_locks'):(stage/d).mkdir(exist_ok=True)
        completed=recovery(run,model,data,spec,fingerprint)
        if len(completed)==2700:
            durable(stage/'status.json',dict(status='completed',completed=2700,utc=utc()));return
        if model=='qwen25vl':
            assert read_json(run/'llava/status.json')['status']=='completed'
            for row in data:load_selection(run,'RD-1.2',row['question_id'],spec)
        selected,states=available_gpus(list(dict.fromkeys(gpus))[:8])
        ctx=mp.get_context('spawn');events=ctx.Queue();tasks=ctx.Queue();stop=ctx.Event();processes={}
        pilot=spec['pilot_question_ids'];phase='loading';pending=[];active={};ready=set();inflight=set();failure=None
        start=time.perf_counter();initial=len(completed);ordinal={r['question_id']:i for i,r in enumerate(data)}
        durable(stage/'pid.json',dict(pid=os.getpid(),session_id=os.getsid(0),utc=utc(),gpus=selected))
        def publish():
            elapsed=time.perf_counter()-start;new=len(completed)-initial
            durable(stage/'status.json',dict(status=phase,completed=len(completed),total=2700,
                completed_answer_count=sum(len(list((stage/'results'/m).glob('*.json'))) for m in stage_methods(model)),
                active=active,pending=len(pending),gpus=selected,gpu_states_before=states,pid=os.getpid(),
                elapsed_seconds=elapsed,eta_seconds=(2700-len(completed))*elapsed/new if new else None,
                failure=failure,utc=utc()))
        def next_phase():
            nonlocal phase,pending
            if not set(pilot)<=completed:
                phase='pilot';pending=[q for q in pilot if q not in completed]
            else:
                # 先行题按真实输入验收，不读取或比较准确率来控制调度。
                for q in pilot:
                    checked=[validate_result(run,model,m,byid[q],spec,fingerprint,deep=True) for m in stage_methods(model)]
                    assert len({r['physical_gpu'] for r in checked})==1
                durable(stage/'pilot_acceptance.json',dict(status='passed',question_ids=pilot,accuracy_not_gate=True,utc=utc()))
                phase='running'
                pending=sorted([q for q in byid if q not in completed],key=lambda q:(-duration_for(run,byid[q],spec),ordinal[q]))
        def refill():
            while pending and len(inflight)<len(selected):
                q=pending.pop(0);inflight.add(q);tasks.put(dict(row=byid[q],ordinal=ordinal[q]))
        try:
            for gpu in selected:
                p=ctx.Process(target=worker,args=(str(run),model,gpu,tasks,events,stop,fingerprint));p.start();processes[gpu]=p
            publish()
            while len(completed)<2700:
                try:event=events.get(timeout=5)
                except queue.Empty:event=None
                if event:
                    kind=event['type'];gpu=event['gpu']
                    if kind=='failed':raise RuntimeError(str(event))
                    if kind=='ready':
                        ready.add(gpu)
                        if len(ready)==len(selected):next_phase();refill()
                    elif kind=='active':active[gpu]=event
                    elif kind=='completed':
                        q=event['question_id'];completed.add(q);inflight.remove(q);active.pop(gpu,None)
                        if phase=='pilot' and not inflight and not pending:next_phase()
                        refill()
                for gpu,p in processes.items():
                    if p.exitcode is not None:raise RuntimeError(f'Worker exited unexpectedly: {gpu} {p.exitcode}')
                publish()
            phase='validating';publish()
        except BaseException as error:
            failure=repr(error);phase='paused';stop.set();publish();raise
        finally:
            # 在途正常调用允许完成保存；不因主循环停止而杀死已开始生成。
            for _ in processes:tasks.put(None)
            while any(p.is_alive() for p in processes.values()):
                for p in processes.values():p.join(timeout=1)
                publish()
        if failure is None:
            complete=recovery(run,model,data,spec,fingerprint);assert len(complete)==2700
            phase='completed';publish()
            durable(stage/'execution.json',dict(status='completed',elapsed_seconds=time.perf_counter()-start,
                gpus=selected,new_questions=2700-initial,completed=2700,utc=utc()))


def duration_for(run,row,spec):
    """调度时长只读历史均匀元数据，不使用题目答案。"""
    path=run/'selections/BASE-Uniform'/(row['question_id']+'.json')
    return read_json(path)['selection']['video']['duration_seconds']


def execute_all(run,gpus):
    """独立后台串联两个环境：任何阶段失败即停止，不自动重启不确定生成。"""
    run=Path(run)
    with exclusive(run/'pipeline.lock'):
        durable(run/'pipeline_pid.json',dict(pid=os.getpid(),session_id=os.getsid(0),utc=utc()))
        for model in ('llava','qwen25vl'):
            command=[PYTHONS[model],str(ROOT/'scripts/run_full_videoqa.py'),'--run-id',run.name,
                     '--stage',model,'--gpus',','.join(gpus),'--resume']
            durable(run/'pipeline_status.json',dict(status='running',stage=model,utc=utc()))
            result=subprocess.run(command,cwd=ROOT)
            if result.returncode:
                durable(run/'pipeline_status.json',dict(status='paused',stage=model,exit_code=result.returncode,utc=utc()))
                raise RuntimeError('Stage paused: '+model)
        from .report import summarize
        summarize(run)
        durable(run/'pipeline_status.json',dict(status='completed',utc=utc()))
