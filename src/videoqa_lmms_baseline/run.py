"""全2700题原生源帧均匀：复用已核验位置，不复用答案，不运行BLIP。"""
import contextlib
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import time
import traceback
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256,offline_environment
from videoqa_full.state import durable,utc,exclusive,invoke_once,rows,pilot_ids,validate_frames
from videoqa_full.prepare import environment,verify_frozen_rd
from videoqa_lmms_bridge.common import framework_identity,LMMS,WFS,GENERATION,POST_PROMPT,load_task,task_doc
from videoqa_lmms_bridge.frames import FrameProvider,native_indices

BRIDGE=ROOT/'outputs/analysis/lmms_bridge_videomme_20260923_r1'
PYTHON='/App/conda/envs/explore_video_qa_lmms/bin/python'
METHOD='BASE-Uniform'


def prepare(run):
    """接口：新运行冻结模型/视频/位置/代码；输入来自900视频已验收索引，无新解码扫描。"""
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():return verify(run)
        start=time.perf_counter();data=rows();identity=framework_identity();verify_frozen_rd()
        profile=read_json(ROOT/'configs/lmms_videomme_reference_v1.json')
        assert profile['generation']==GENERATION and profile['frame_budget']==16
        assert (profile['conversation_template'],profile['dtype'],profile['attention'])==('qwen_1_5','bfloat16','sdpa')
        assert profile['extra_time_instruction'] is False and profile['visual_size']==[384,384]
        assert read_json(BRIDGE/'completion_audit.json')['status']=='passed'
        summary=read_json(BRIDGE/'uniform_summary.json');assert summary['videos']==900
        assert read_json(BRIDGE/'framework_uniform_check.json')['count']==900
        sources={};videos={};stats={}
        # 步骤1：再次验证源视频字节，但不重复生成候选或运行任何模型。
        for vid,h in summary['record_hashes'].items():
            p=BRIDGE/'uniform_records'/(vid+'.json');assert sha256(p)==h
            x=read_json(p);assert sha256(BRIDGE/x['pts_file'])==x['pts_sha256']
            assert [f['source_frame_index'] for f in x['new_frames']]==native_indices(x['source_frame_count'])
            path=Path(x['video']['path']);assert sha256(path)==x['source_video_sha256']
            sources[vid]=dict(path=str(p),sha256=h,pts_path=str(BRIDGE/x['pts_file']),pts_sha256=x['pts_sha256'])
            videos[vid]=x['source_video_sha256'];st=path.stat()
            stats[vid]=dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns)
        inventory=read_json(ROOT/'configs/local_model_inventory.json');model_files={}
        for model in inventory['models'].values():
            for name,entry in model['files'].items():
                p=Path(model['path'])/name;assert sha256(p)==entry['sha256']
                st=p.stat();model_files[str(p)]=dict(sha256=entry['sha256'],bytes=st.st_size,mtime_ns=st.st_mtime_ns)
        print('900 video/source-index hashes and model inventory verified',flush=True)
        # 步骤2：沿用桥接已经核验的同一环境和模型层，不搜索能达到60.6%的配置。
        env=environment(PYTHON);prior=read_json(BRIDGE/'bridge_protocol.json')
        assert env==prior['environment']
        files=list((ROOT/'src/videoqa_lmms_baseline').glob('*.py'))
        files+=list((ROOT/'src/videoqa_lmms_bridge').glob('*.py'))
        files+=[ROOT/'scripts/run_lmms_uniform_full.py',ROOT/'src/videoqa_runtime/llava_backend.py',
                ROOT/'src/videoqa_runtime/common.py',ROOT/'src/videoqa_runtime/protocol.py',
                ROOT/'src/videoqa_full/state.py',ROOT/'configs/local_model_inventory.json',
                ROOT/'configs/llava_source_manifest.json',ROOT/'configs/lmms_videomme_reference_v1.json',ROOT/'data/manifests/video_mme_full_2700.json']
        spec=dict(version='lmms-videomme-reference-v1',run_id=run.name,method=METHOD,model='LLaVA-Video-7B-Qwen2',
            question_count=2700,video_count=900,max_answer_calls=2700,automatic_retries=0,
            framework=identity,reference_profile=profile,environment=env,source_uniform=sources,video_hashes=videos,video_stats=stats,
            model_files=model_files,generation=GENERATION,post_prompt=POST_PROMPT,template='qwen_1_5',
            dtype='bfloat16',attention='sdpa',visual_size=[384,384],visual_tokens_per_frame=210,
            frame_budget=16,selection_rule='linspace original source indices; exact PTS; cap min(16,N_source)',
            source_domain='original frames, not 1 FPS candidates',extra_time_instruction=False,
            seed=2027,pilot_question_ids=pilot_ids(data),BLIP_calls=0,query_calls=0,
            timing='cached verified source positions -> exact final decoding -> reference QA; not raw full-scan E2E',
            limitations=['Reference task protocol with official model settings; exact author 60.6 run command/logs unavailable',
                'Not an exact reproduction claim based merely on rounded accuracy',
                'No parameter search or automatic repeat based on correctness'],
            code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in files},prepared_seconds=time.perf_counter()-start,
            created_at=utc())
        for p in files:
            dest=run/'code_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        durable(run/'protocol.json',spec)
        return spec


def verify(run):
    """恢复前核对协议与输入指纹；已完成结果不得因代码变化静默重答。"""
    spec=read_json(Path(run)/'protocol.json');assert framework_identity()==spec['framework']
    assert environment(PYTHON)==spec['environment']
    for name,h in spec['code_sha256'].items():assert sha256(ROOT/name)==h,name
    for p,x in spec['model_files'].items():
        st=Path(p).stat();assert st.st_size==x['bytes'] and st.st_mtime_ns==x['mtime_ns']
    return spec


def selection(spec,vid):
    """接口：仅按video_id取得冻结原生均匀帧，不接收问题、选项或标准答案。"""
    source=spec['source_uniform'][vid];assert sha256(source['path'])==source['sha256']
    r=read_json(source['path']);s=dict(video=r['video'],selected_frames=r['new_frames'])
    validate_frames(s);assert [f['source_frame_index'] for f in s['selected_frames']]==native_indices(r['source_frame_count'])
    return s


def validate(run,row,spec,fp):
    """逐题复验实际视觉输入预算、任务计分和成功调用，不调用模型。"""
    q=row['question_id'];r=read_json(run/'results'/(q+'.json'))
    assert r['protocol_sha256']==fp and r['status']=='completed' and r['method']==METHOD
    for k in ('question_id','video_id','question','options','stratum'):assert r[k]==row[k]
    s=selection(spec,row['video_id']);assert r['selection']==s
    a=r['result']['answer'];o=r['result']['observed'];n=len(s['selected_frames'])
    assert o['generate_calls']==o['prefill_calls']==1 and o['checked_logit_steps']>0
    assert o['max_new_tokens']==16 and len(o['generated_token_ids'])<=16
    assert o['visual_tokens']==n*210 and o['prefill_tokens']==o['text_input_tokens']-1+n*210
    assert o['pixel_sha256']==o['expected_pixel_sha256']
    task=load_task();metric=task.videomme_process_results(task_doc(row),[a['raw_output']])['videomme_perception_score']
    assert metric==a['framework_metric'] and r['correct']==(metric['pred_answer']==metric['answer'])
    assert a['context']==task.videomme_doc_to_text(task_doc(row),{'post_prompt':POST_PROMPT})
    call=read_json(run/'calls'/METHOD/(q+'.json'))
    assert call['status']=='completed' and call['protocol_sha256']==fp and call['answer']==r['result']
    assert call['generation_state']=={'started':True,'returned':True}
    return r


def process_question(run,row,backend,adapter,gpu,spec,fp):
    """一步问答：固定源帧→精确解码→调用前落盘→框架生成/计分→结果原子保存。"""
    from videoqa_lmms_bridge.llava_adapter import observe_generate
    q=row['question_id'];dest=run/'results'/(q+'.json')
    if dest.exists():validate(run,row,spec,fp);return
    with exclusive(run/'task_locks'/(q+'.lock')):
        start=time.perf_counter();s=selection(spec,row['video_id']);stats=spec['video_stats'][row['video_id']]
        st=Path(stats['path']).stat();assert st.st_size==stats['bytes'] and st.st_mtime_ns==stats['mtime_ns']
        images,hashes=FrameProvider(s).decode();decode_end=time.perf_counter()
        def answer(state):
            a,o=observe_generate(backend,images,16,state,lambda:adapter.ask(images,
                [f['timestamp_seconds'] for f in s['selected_frames']],s['video']['duration_seconds'],row,s['video']['path']))
            return dict(answer=a,observed=o)
        try:result,reused,checkpoint=invoke_once(run,q,METHOD,fp,gpu,answer)
        finally:
            for im in images:im.close()
        end=time.perf_counter();metric=result['answer']['framework_metric']
        r=dict(status='completed',method=METHOD,**{k:row[k] for k in ('question_id','video_id','question','options','stratum')},
            protocol_sha256=fp,source_video_sha256=spec['video_hashes'][row['video_id']],selection=s,
            rgb_sha256=hashes,result=result,correct=metric['pred_answer']==metric['answer'],gpu=gpu,
            reused_answer=reused,timing_kind='recovered' if reused else 'cached_positions_final_decode_qa',
            timings=dict(position_read_and_final_decode_seconds=decode_end-start,
                cached_input_to_persisted_answer_seconds=end-start,call_start_checkpoint_seconds=checkpoint),completed_at=utc())
        durable(dest,r);validate(run,row,spec,fp)


def worker(run,gpu,jobs,events,stop,fp):
    """每卡常驻单一LLaVA；无BLIP，失败停止，不自动重试真实回答。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    offline_environment();run=Path(run);active=None
    with (run/'logs'/f'gpu_{gpu}.log').open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import torch
            from videoqa_runtime.llava_backend import LlavaBackend
            from videoqa_lmms_bridge.llava_adapter import FrozenInputLlavaVid
            torch.set_num_threads(2);torch.set_num_interop_threads(1);start=time.perf_counter()
            backend=LlavaBackend();adapter=FrozenInputLlavaVid(backend);loading=time.perf_counter()-start
            start=time.perf_counter();backend.warmup()
            durable(run/'workers'/f'{gpu}_{os.getpid()}.json',dict(gpu=gpu,pid=os.getpid(),load_seconds=loading,
                warmup_seconds=time.perf_counter()-start,synthetic_forwards=1,load_report=backend.load_report))
            spec=read_json(run/'protocol.json');events.put(dict(type='ready',gpu=gpu))
            while not stop.is_set():
                row=jobs.get()
                if row is None:break
                active=row['question_id'];events.put(dict(type='active',gpu=gpu,question_id=active))
                process_question(run,row,backend,adapter,gpu,spec,fp)
                events.put(dict(type='completed',gpu=gpu,question_id=active));active=None
        except BaseException as error:
            stop.set();traceback.print_exc()
            durable(run/'failures'/f'{gpu}_{time.time_ns()}.json',dict(error=repr(error),question_id=active,traceback=traceback.format_exc(),utc=utc()))
            events.put(dict(type='failed',gpu=gpu,error=repr(error)))


def execute(run,requested):
    """固定2700调用ID，前7题按输入协议验收；背景调度与持久状态不依赖会话。"""
    from videoqa_runtime.baseline_run import available_gpus
    run=Path(run)
    with exclusive(run/'coordinator.lock'):
        spec=verify(run);data=rows();byid={r['question_id']:r for r in data};fp=sha256(run/'protocol.json')
        for d in ('logs','workers','failures','results','calls','task_locks'):(run/d).mkdir(exist_ok=True)
        calls=list((run/'calls'/METHOD).glob('*.json'));assert len(calls)<=2700
        for p in calls:
            c=read_json(p);assert p.stem in byid and c['status']=='completed' and c['protocol_sha256']==fp,'Uncertain call: '+str(p)
        done=set()
        for p in (run/'results').glob('*.json'):
            assert p.stem in byid;validate(run,byid[p.stem],spec,fp);done.add(p.stem)
        if len(done)==2700:
            from .report import summarize
            return summarize(run)
        gpus,states=available_gpus(requested);ctx=mp.get_context('spawn');jobs=ctx.Queue();events=ctx.Queue();stop=ctx.Event()
        children={};ready=set();active={};pending=[];inflight=set();phase='loading';start=time.perf_counter();failure=None
        def publish():durable(run/'status.json',dict(status=phase,completed=len(done),total=2700,pid=os.getpid(),session_id=os.getsid(0),
            active=active,gpus=gpus,elapsed_seconds=time.perf_counter()-start,failure=failure,utc=utc()))
        def schedule():
            nonlocal pending,phase
            pilot=spec['pilot_question_ids']
            if not set(pilot)<=done:phase='pilot';pending=[q for q in pilot if q not in done]
            else:
                for q in pilot:validate(run,byid[q],spec,fp)
                durable(run/'pilot_acceptance.json',dict(status='passed',question_ids=pilot,accuracy_not_gate=True))
                phase='running';pending=sorted([q for q in byid if q not in done],key=lambda q:(-selection(spec,byid[q]['video_id'])['video']['duration_seconds'],q))
        def refill():
            while pending and len(inflight)<len(gpus):
                q=pending.pop(0);inflight.add(q);jobs.put(byid[q])
        try:
            for gpu in gpus:
                p=ctx.Process(target=worker,args=(str(run),gpu,jobs,events,stop,fp));p.start();children[gpu]=p
            publish()
            while len(done)<2700:
                try:e=events.get(timeout=5)
                except queue.Empty:e=None
                if e:
                    if e['type']=='failed':raise RuntimeError(str(e))
                    if e['type']=='ready':
                        ready.add(e['gpu'])
                        if len(ready)==len(gpus):schedule();refill()
                    elif e['type']=='active':active[e['gpu']]=e['question_id']
                    elif e['type']=='completed':
                        q=e['question_id'];done.add(q);inflight.remove(q);active.pop(e['gpu'],None)
                        if phase=='pilot' and not pending and not inflight:schedule()
                        refill()
                for gpu,p in children.items():
                    if p.exitcode is not None:raise RuntimeError(f'Worker {gpu} exited {p.exitcode}')
                publish()
            phase='validating';publish()
        except BaseException as error:stop.set();failure=repr(error);phase='paused';publish();raise
        finally:
            for _ in children:jobs.put(None)
            while any(p.is_alive() for p in children.values()):
                for p in children.values():p.join(timeout=1)
                publish()
        if failure is None:
            from .report import summarize
            summarize(run);phase='completed';publish()
            durable(run/'execution.json',dict(status='completed',wall_seconds=time.perf_counter()-start,gpus=gpus,new_answer_calls=len(list((run/'calls'/METHOD).glob('*.json'))),utc=utc()))
