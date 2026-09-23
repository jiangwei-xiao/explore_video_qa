"""三组冻结源帧的Qwen全量评测，调用与后台状态不依赖交互会话。"""
import contextlib
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import time
import traceback
from videoqa_runtime.common import ROOT,read_json,sha256,offline_environment
from videoqa_full.state import durable,utc,exclusive,invoke_once,rows,pilot_ids,validate_frames,digest
from videoqa_full.prepare import environment,verify_frozen_rd
from videoqa_lmms_bridge.common import framework_identity,GENERATION,load_task,task_doc,POST_PROMPT
from videoqa_lmms_bridge.frames import FrameProvider
from videoqa_lmms_baseline.methods import UNIFORM,SOURCE
from .observe import observe,result_from_raw

METHODS=('BASE-Uniform','BASE-BLIP-TopK','RD-1.2')
GPUS=('4','5','6','7')
PYTHON='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'
LLAVA=ROOT/'outputs/full_videoqa/topk_rd12__llava__lmms_ref_v1__videomme2700__20260923__r01'
PROFILE=ROOT/'configs/lmms_videomme_qwen25vl_reference_v1.json'


def method_order(ordinal):
    """三种方法按冻结题序轮换，顺序不取决于预测或标准答案。"""
    k=ordinal%3;return METHODS[k:]+METHODS[:k]


def critical_code():
    """合成验收绑定实际适配器和观测代码，不把旧验收冒充新入口已验收。"""
    names=['src/videoqa_lmms_qwen/observe.py','src/videoqa_lmms_qwen/run.py',
        'src/videoqa_lmms_bridge/qwen_adapter.py','src/videoqa_full/qwen_fa2_backend.py',
        'src/videoqa_full/qwen_backend.py','configs/lmms_videomme_qwen25vl_reference_v1.json']
    return {n:sha256(ROOT/n) for n in names}


def synthetic(run):
    """接口：仅合成输入验收，允许后四卡中的首张空闲卡，不产生真实题答案。"""
    from videoqa_runtime.baseline_run import available_gpus
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    gpus,_=available_gpus(list(GPUS));gpu=gpus[0]
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    offline_environment()
    import torch
    import numpy as np
    from PIL import Image
    from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
    from videoqa_lmms_bridge.qwen_adapter import FrozenInputQwen
    from videoqa_lmms_bridge.bridge import synthetic_row
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    report=dict(status='running',gpu=gpu,real_calls=0,synthetic_calls_started=0,checks=[],code_sha256=critical_code(),framework=framework_identity(),environment=environment(PYTHON))
    try:
        backend=QwenFA2VideoBackend();adapter=FrozenInputQwen(backend);report['load_report']=backend.load_report
        # 步骤1：覆盖变长、竖屏、低分辨率及精确像素上限，不根据合成答案调配置。
        for n,size in [(1,(320,240)),(11,(720,1280)),(14,(1280,720)),(16,(896,672))]:
            frames=[Image.new('RGB',size,(20+i*11,100,150)) for i in range(n)]
            report['synthetic_calls_started']+=1;state={}
            result=observe(backend,adapter,frames,synthetic_row(n),state)
            assert state=={'started':True,'returned':True}
            # 步骤2：奇数官方自动补齐必须与显式复制末帧的处理器产物一致。
            if n%2:
                videos=np.stack([np.asarray(f) for f in frames])
                prompt=backend.processor.apply_chat_template([dict(role='system',content='You are a helpful assistant.'),
                    dict(role='user',content=[dict(type='video'),dict(type='text',text=result['observed']['context'])])],
                    tokenize=False,add_generation_prompt=True)
                args=dict(fps=2.0,images_kwargs={'size':dict(shortest_edge=100352,longest_edge=602112)},truncation=False,return_tensors='pt')
                # 处理器会原地展开text列表，每次对照必须新建列表，不能重用已展开的提示词。
                a=backend.processor(text=[prompt],videos=[videos],**args)
                b=backend.processor(text=[prompt],videos=[np.concatenate([videos,videos[-1:]],axis=0)],**args)
                assert torch.equal(a['pixel_values_videos'],b['pixel_values_videos'])
            report['checks'].append(dict(n=n,size=list(size),result=result))
            for f in frames:f.close()
        report['status']='passed'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        durable(run/'synthetic_attempts'/f'{time.time_ns()}.json',report)
        if report['status']=='passed':durable(run/'synthetic_acceptance.json',report)


def prepare(run):
    """接口：合成通过后冻结8100份无答案输入清单、环境及模型/视频字节身份。"""
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():return verify(run)
        start=time.perf_counter();profile=read_json(PROFILE);env=environment(PYTHON)
        syn=read_json(run/'synthetic_acceptance.json');assert syn['status']=='passed' and syn['code_sha256']==critical_code()
        assert syn['environment']==env and syn['framework']==framework_identity()
        assert env['torch'].startswith('2.6.0') and env['transformers']=='4.50.0'
        assert env.get('flash-attn',env.get('flash_attn'))=='2.7.4.post1'
        assert profile['generation']==GENERATION and profile['allowed_gpus']==list(GPUS)
        assert read_json(UNIFORM/'completion_audit.json')['status']==read_json(LLAVA/'completion_audit.json')['status']=='passed'
        verify_frozen_rd();data=rows();exports={};sources={};source_summaries={}
        base=read_json(UNIFORM/'protocol.json');old=read_json(SOURCE/'protocol.json')
        # 步骤1：源清单不含历史预测/答案，仅携带身份、已选帧与RGB哈希。
        for r in data:
            q=r['question_id']
            for m in METHODS:
                path=(UNIFORM/'results' if m==METHODS[0] else LLAVA/'results'/m)/(q+'.json');item=read_json(path)
                assert item['status']=='completed' and item['method']==m
                for k in ('question_id','video_id','question','options','stratum'):assert item[k]==r[k]
                validate_frames(item['selection']);h=sha256(path);sources[str(path)]=h
                clean=dict(question_id=q,video_id=r['video_id'],method=m,question_sha256=digest([r['question'],r['options']]),
                    selection=item['selection'],rgb_sha256=item['rgb_sha256'],source_path=str(path),source_sha256=h,
                    source_video_sha256=base['video_hashes'][r['video_id']])
                dest=run/'selections'/m/(q+'.json');durable(dest,clean);exports[str(dest.relative_to(run))]=sha256(dest)
        for source in (UNIFORM,LLAVA):
            source_summaries[str(source/'summary.json')]=sha256(source/'summary.json')
        # 步骤2：完整复核900视频和Qwen权重；不改变共享模型，不加载BLIP。
        for v,st in base['video_stats'].items():assert sha256(st['path'])==base['video_hashes'][v]
        model_files={p:x for p,x in old['model_files'].items() if p.startswith(profile['model_path']+'/')}
        assert model_files and any(p.endswith('.safetensors') for p in model_files)
        for p,x in model_files.items():
            assert sha256(p)==x['sha256'];st=Path(p).stat();assert st.st_size==x['bytes'] and st.st_mtime_ns==x['mtime_ns']
        paths=list((ROOT/'src').glob('**/*.py'))+list((ROOT/'scripts').glob('*.py'))+[PROFILE,ROOT/'data/manifests/video_mme_full_2700.json']
        code={str(p.relative_to(ROOT)):sha256(p) for p in paths}
        for name in code:
            dest=run/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        assert shutil.disk_usage(run).free>20*1024**3
        spec=dict(run_id=run.name,model=profile['model'],profile=profile,framework=framework_identity(),environment=env,
            methods=list(METHODS),max_calls_per_method=2700,max_calls=8100,allowed_gpus=list(GPUS),
            synthetic_acceptance_sha256=sha256(run/'synthetic_acceptance.json'),selection_exports=exports,source_results=sources,
            source_summaries=source_summaries,model_files=model_files,video_stats=base['video_stats'],video_hashes=base['video_hashes'],
            pilot_question_ids=pilot_ids(data),code_sha256=code,prepared_seconds=time.perf_counter()-start,utc=utc(),
            timing='cached selection + final decode + instrumented QA; not raw selection E2E')
        durable(run/'protocol.json',spec);print('Prepared 8100 frozen inputs',flush=True);return spec


def verify(run):
    """启动/恢复前校验已冻结环境及代码；任何变化停止，不重新生成追求一致。"""
    spec=read_json(Path(run)/'protocol.json');assert framework_identity()==spec['framework']
    assert environment(PYTHON)==spec['environment'];verify_frozen_rd()
    for n,h in spec['code_sha256'].items():assert sha256(ROOT/n)==h,n
    for p,h in spec['source_summaries'].items():assert sha256(p)==h
    for p,x in spec['model_files'].items():
        st=Path(p).stat();assert st.st_size==x['bytes'] and st.st_mtime_ns==x['mtime_ns']
    return spec


def selection(run,spec,q,m):
    """只读输入接口：唯一源帧及预期RGB来自已核验LLaVA运行，不做新选择。"""
    rel=f'selections/{m}/{q}.json';p=run/rel;assert sha256(p)==spec['selection_exports'][rel]
    item=read_json(p);validate_frames(item['selection']);return item


def recover_call(run,row,m,spec):
    """只恢复已完整落盘的原始生成；无原始检查点或状态不确定时拒绝重复调用。"""
    from .observe import result_from_raw
    q=row['question_id'];p=run/'calls'/m/(q+'.json')
    if not p.exists():return
    call=read_json(p);assert call['protocol_sha256']==sha256(run/'protocol.json')
    if call['status']=='completed':return
    raw=read_json(run/'raw_generations'/m/(q+'.json'))
    assert raw['identity']==dict(question_id=q,method=m,protocol_sha256=call['protocol_sha256'],gpu=call['gpu'])
    assert raw['generation_state']=={'started':True,'returned':True}
    answer=result_from_raw(raw,row)
    call.update(status='completed',answer=answer,generation_state=raw['generation_state'],recovered_postprocessing=True,completed_at=utc())
    durable(p,call)


def validate(run,row,m,spec):
    """全量输入/计分复验；模型Token使用Qwen网格，不套用LLaVA的210每帧。"""
    from videoqa_full.qwen_backend import padded_count,MIN_PIXELS,MAX_PIXELS
    q=row['question_id'];r=read_json(run/'results'/m/(q+'.json'));fp=sha256(run/'protocol.json')
    assert r['status']=='completed' and r['model']==spec['model'] and r['method']==m and r['protocol_sha256']==fp
    for k in ('question_id','video_id','question','options','stratum'):assert r[k]==row[k]
    item=selection(run,spec,q,m);assert r['selection']==item['selection'] and r['rgb_sha256']==item['rgb_sha256']
    a=r['answer'];o=a['observed'];n=len(item['selection']['selected_frames']);h,w=o['processed_size']
    assert o['generate_calls']==o['prefill_calls']==1 and o['checked_logit_steps']>0
    assert o['prefill_tokens']==o['actual_prefill_tokens']==len(o['input_token_ids'])
    assert o['source_frames']==n and o['encoded_frames']==padded_count(n)
    assert o['internal_padding_frames']==padded_count(n)-n and o['second_per_grid_ts']==[1.0]
    assert h%28==w%28==0 and MIN_PIXELS<=h*w<=MAX_PIXELS
    assert o['visual_tokens']==o['video_grid_thw'][0][0]*o['video_grid_thw'][0][1]*o['video_grid_thw'][0][2]//4
    assert o['processor_pixel_sha256']==o['model_pixel_sha256']
    assert 1<=len(o['generated_token_ids'])<=16 and o['generation_kwargs']['max_new_tokens']==16
    assert o['generation_kwargs']['do_sample'] is False and o['generation_kwargs']['num_beams']==1
    task=load_task();metric=task.videomme_process_results(task_doc(row),[a['raw_output']])['videomme_perception_score']
    assert metric==a['framework_metric'] and r['correct']==(metric['answer']==metric['pred_answer'])
    assert o['context']==task.videomme_doc_to_text(task_doc(row),{'post_prompt':POST_PROMPT})
    call=read_json(run/'calls'/m/(q+'.json'));assert call['status']=='completed' and call['answer']==a and call['protocol_sha256']==fp
    assert call['generation_state']=={'started':True,'returned':True} and call['gpu']==r['gpu'] and r['gpu'] in GPUS
    raw=read_json(run/'raw_generations'/m/(q+'.json'));assert raw['generation_state']==call['generation_state']
    assert raw['identity']==dict(question_id=q,method=m,protocol_sha256=fp,gpu=r['gpu'])
    assert result_from_raw(raw,row)['raw_output']==a['raw_output']
    return r


def process(run,row,m,backend,adapter,gpu,spec):
    """步骤1精确取帧，步骤2一次生成，步骤3结果落盘；已完成答案仅恢复后处理。"""
    q=row['question_id'];dest=run/'results'/m/(q+'.json')
    if dest.exists():validate(run,row,m,spec);return
    with exclusive(run/'locks'/m/(q+'.lock')):
        start=time.perf_counter();item=selection(run,spec,q,m);s=item['selection'];st=spec['video_stats'][row['video_id']]
        actual=Path(st['path']).stat();assert actual.st_size==st['bytes'] and actual.st_mtime_ns==st['mtime_ns']
        images,hashes=FrameProvider(s).decode();assert hashes==item['rgb_sha256'];decoded=time.perf_counter();fp=sha256(run/'protocol.json')
        identity=dict(question_id=q,method=m,protocol_sha256=fp,gpu=gpu)
        try:
            answer,reused,checkpoint=invoke_once(run,q,m,fp,gpu,lambda state:observe(backend,adapter,images,row,state,
                run/'raw_generations'/m/(q+'.json'),identity))
        finally:
            for image in images:image.close()
        ended=time.perf_counter();metric=answer['framework_metric']
        result=dict(status='completed',model=spec['model'],method=m,protocol_sha256=fp,
            **{k:row[k] for k in ('question_id','video_id','question','options','stratum')},
            selection=s,rgb_sha256=hashes,answer=answer,gpu=gpu,reused_answer=reused,
            correct=metric['answer']==metric['pred_answer'],timings=dict(final_decode_seconds=decoded-start,
                cached_input_to_persisted_answer_seconds=ended-start,call_start_checkpoint_seconds=checkpoint),utc=utc())
        durable(dest,result);validate(run,row,m,spec)


def worker(run,gpu,jobs,events,stop):
    """每卡常驻Qwen，三方法同题同进程；不加载LLaVA/BLIP或调用旧Qwen answer。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    offline_environment();run=Path(run);active=None
    with (run/'logs'/f'gpu_{gpu}.log').open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import torch
            from PIL import Image
            from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
            from videoqa_lmms_bridge.qwen_adapter import FrozenInputQwen
            from videoqa_lmms_bridge.bridge import synthetic_row
            torch.set_num_threads(2);torch.set_num_interop_threads(1);t=time.perf_counter()
            backend=QwenFA2VideoBackend();adapter=FrozenInputQwen(backend);loading=time.perf_counter()-t
            # 步骤1：新进程在像素上限进行固定合成预热，不产生真实题答案。
            images=[Image.new('RGB',(896,672),(40,i*10,150)) for i in range(16)];t=time.perf_counter()
            syn=observe(backend,adapter,images,synthetic_row(16),{})
            for image in images:image.close()
            durable(run/'workers'/f'{gpu}_{os.getpid()}.json',dict(gpu=gpu,pid=os.getpid(),load_seconds=loading,
                warmup_seconds=time.perf_counter()-t,synthetic_generations=1,warmup=syn,load_report=backend.load_report))
            spec=read_json(run/'protocol.json');events.put(dict(type='ready',gpu=gpu))
            # 步骤2：同题三方法固定轮换，失败停止派发，不自动重试。
            while not stop.is_set():
                job=jobs.get()
                if job is None:break
                row,ordinal=job;active=row['question_id']
                for m in method_order(ordinal):
                    if stop.is_set():break
                    events.put(dict(type='active',gpu=gpu,question_id=active,method=m))
                    process(run,row,m,backend,adapter,gpu,spec)
                    events.put(dict(type='answer',gpu=gpu,question_id=active,method=m))
                events.put(dict(type='completed',gpu=gpu,question_id=active));active=None
        except BaseException as exc:
            stop.set();traceback.print_exc();durable(run/'failures'/f'{gpu}_{time.time_ns()}.json',
                dict(error=repr(exc),question_id=active,traceback=traceback.format_exc(),utc=utc()))
            events.put(dict(type='failed',gpu=gpu,error=repr(exc)))


def execute(run):
    """后台协调接口：仅后四卡，资源等待/先行放行/完成汇总均无需交互连接。"""
    from videoqa_runtime.baseline_run import available_gpus
    run=Path(run)
    with exclusive(run/'coordinator.lock'):
        spec=verify(run);data=rows();byid={r['question_id']:r for r in data};ordinals={r['question_id']:i for i,r in enumerate(data)}
        done=set();pinned={};start=time.perf_counter();phase='checking';active={};gpus=[];failure=None
        def publish():
            elapsed=time.perf_counter()-start
            durable(run/'status.json',dict(status=phase,completed=len(done),total=8100,
                counts={m:sum(x[1]==m for x in done) for m in METHODS},pid=os.getpid(),session_id=os.getsid(0),
                active=active,gpus=gpus,elapsed_seconds=elapsed,failure=failure,utc=utc()))
        # 步骤1：固定方法×qid调用ID；失败只有存在完整原始答案才允许后处理恢复。
        for m in METHODS:
            calls=list((run/'calls'/m).glob('*.json'));assert len(calls)<=2700
            for p in calls:
                assert p.stem in byid;recover_call(run,byid[p.stem],m,spec);c=read_json(p)
                assert c['status']=='completed' and c['gpu'] in GPUS
                assert pinned.setdefault(p.stem,c['gpu'])==c['gpu']
            for p in (run/'results'/m).glob('*.json'):
                assert p.stem in byid;validate(run,byid[p.stem],m,spec);done.add((p.stem,m))
        if len(done)==8100:
            from .report import summarize
            return summarize(run)
        # 步骤2：全忙或部分完成题原卡不可用时持久等待，不抢卡、不迁移半题。
        required={gpu for q,gpu in pinned.items() if any((q,m) not in done for m in METHODS)}
        while True:
            try:gpus,states=available_gpus(list(GPUS))
            except RuntimeError:gpus=[];states={}
            if gpus and required<=set(gpus):break
            phase='waiting_resources';publish();time.sleep(15)
        assert set(gpus)<=set(GPUS)
        for d in ('logs','workers','failures'):(run/d).mkdir(exist_ok=True)
        durable(run/'gpu_preflight.json',dict(allowed=list(GPUS),actual=gpus,states=states,utc=utc()))
        ctx=mp.get_context('spawn');queues={g:ctx.Queue() for g in gpus};events=ctx.Queue();stop=ctx.Event()
        children={};ready=set();busy=set();pending=[];phase='loading'
        def schedule():
            nonlocal phase,pending
            pilot=spec['pilot_question_ids'];missing=lambda q:any((q,m) not in done for m in METHODS)
            if any(missing(q) for q in pilot):phase='pilot';pending=[q for q in pilot if missing(q)]
            else:
                for q in pilot:
                    for m in METHODS:validate(run,byid[q],m,spec)
                durable(run/'pilot_acceptance.json',dict(status='passed',question_ids=pilot,calls=21,accuracy_not_gate=True))
                phase='running';pending=sorted([q for q in byid if missing(q)],key=lambda q:(
                    -selection(run,spec,q,METHODS[0])['selection']['video']['duration_seconds'],q))
        def refill():
            for gpu in gpus:
                if gpu in busy:continue
                q=next((q for q in pending if q not in pinned or pinned[q]==gpu),None)
                if q is None:continue
                pending.remove(q);busy.add(gpu);queues[gpu].put((byid[q],ordinals[q]))
        # 步骤3：先21条输入门槛，再全量；不以准确率决定放行或改参数。
        try:
            for gpu in gpus:
                p=ctx.Process(target=worker,args=(str(run),gpu,queues[gpu],events,stop));p.start();children[gpu]=p
            publish()
            while len(done)<8100:
                try:e=events.get(timeout=5)
                except queue.Empty:e=None
                if e:
                    if e['type']=='failed':raise RuntimeError(str(e))
                    if e['type']=='ready':
                        ready.add(e['gpu'])
                        if len(ready)==len(gpus):schedule();refill()
                    elif e['type']=='active':active[e['gpu']]={k:e[k] for k in ('question_id','method')}
                    elif e['type']=='answer':done.add((e['question_id'],e['method']))
                    elif e['type']=='completed':
                        busy.remove(e['gpu']);active.pop(e['gpu'],None)
                        if phase=='pilot' and not pending and not busy:schedule()
                        refill()
                for gpu,p in children.items():
                    if p.exitcode is not None:raise RuntimeError(f'Worker {gpu} exited {p.exitcode}')
                publish()
            phase='validating';active.clear();publish()
        except BaseException as exc:
            stop.set();failure=repr(exc);phase='paused';publish();raise
        finally:
            for q in queues.values():q.put(None)
            while any(p.is_alive() for p in children.values()):
                for p in children.values():p.join(timeout=1)
                publish()
        from .report import summarize
        summarize(run);phase='completed';publish()
        durable(run/'execution.json',dict(status='completed',wall_seconds=time.perf_counter()-start,gpus=gpus,new_answer_calls=8100,utc=utc()))
