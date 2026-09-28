"""冻结开发200题选帧的独立Qwen评测；复用已验收生成路径，不修改旧全量入口。"""
import contextlib
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import time
import traceback
from videoqa_runtime.common import ROOT,read_json,sha256,offline_environment
from videoqa_full.state import durable,rows,exclusive,utc
from videoqa_runtime.baseline_report import wilson,distribution
from videoqa_full.report import compare
from .run import verify as verify_parent,process,validate,recover_call,PYTHON,GPUS


def selected_rows(spec):
    """接口：只取冻结开发清单，保持名单与顺序，不按历史对错筛题。"""
    data={r['question_id']:r for r in rows()}
    ids=spec['question_ids'];assert len(ids)==len(set(ids))==200
    return [data[q] for q in ids]


def prepare(run,input_dir,allowed):
    """接口：冻结200份输入、模型/环境/代码及调用上限，问答前不读取分数择优。"""
    run,input_dir=Path(run),Path(input_dir).resolve();run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():return verify(run)
        source=input_dir/'selection_manifest.json';s=read_json(source)
        assert s['status']=='completed' and s['question_count']==200 and s['frame_budget']==32
        assert sha256(s['dataset_manifest'])==s['manifest_sha256']
        assert s['question_ids']==[r['question_id'] for r in read_json(s['dataset_manifest'])['rows']]
        parent=Path(s['parent_run']);assert sha256(parent/'protocol.json')==s['parent_protocol_sha256']
        base=verify_parent(parent);method=s['method']
        assert allowed and len(set(allowed))==len(allowed) and set(allowed)<=set(GPUS)
        # 步骤1：只导入冻结选择和RGB身份；不导入任何历史答案作为新回答。
        exports={};sources={str(source):sha256(source),s['dataset_manifest']:s['manifest_sha256']}
        for q in s['question_ids']:
            p=input_dir/(q+'.json');assert sha256(p)==s['exports'][q]
            item=read_json(p);assert item['question_id']==q and item['method']==method
            for path,h in item['sources'].items():assert sha256(path)==h
            sources.update(item['sources']);sources[str(p)]=sha256(p)
            rel=f'selections/{method}/{q}.json';dest=run/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha256(dest)==sha256(p)
            else:shutil.copy2(p,dest)
            exports[rel]=sha256(dest)
        from .selection import OLD_QWEN
        for reference in (parent,OLD_QWEN):
            for m in ('BASE-Uniform','BASE-BLIP-TopK','RD-1.2'):
                for q in s['question_ids']:
                    p=reference/'results'/m/(q+'.json');sources[str(p)]=sha256(p)
        # 步骤2：旧生成/计分代码保持原样；新增协调层单独快照和校验。
        code=dict(base['code_sha256'])
        for n in ('src/videoqa_budget32/development.py','scripts/run_frozen_development.py'):
            code[n]=sha256(ROOT/n)
        for n,h in code.items():
            assert sha256(ROOT/n)==h
            dest=run/'code_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
        spec=dict(base,run_id=run.name,methods=[method],configuration=s['configuration'],diagnostic=s['diagnostic'],
            question_ids=s['question_ids'],question_count=200,max_calls=200,max_calls_per_method=200,
            allowed_gpus=allowed,pilot_question_ids=s['question_ids'][:5],selection_exports=exports,
            code_sha256=code,input_sources=sources,input_manifest=str(source),parent_run=str(parent),
            parent_protocol_sha256=sha256(parent/'protocol.json'),utc=utc())
        durable(run/'protocol.json',spec);return spec


def verify(run):
    """接口：恢复时严格复核来源；不静默接受新输入、依赖或模型变化。"""
    spec=verify_parent(Path(run))
    assert spec['max_calls']==spec['question_count']==200 and len(spec['methods'])==1
    assert sha256(Path(spec['parent_run'])/'protocol.json')==spec['parent_protocol_sha256']
    for path,h in spec['input_sources'].items():assert sha256(path)==h,path
    selected_rows(spec)
    return spec


def summarize(run):
    """接口：全部200条闭合后比较历史同协议结果，诊断和新算法成绩明确区分。"""
    from .selection import OLD_QWEN
    run=Path(run);spec=verify(run);data=selected_rows(spec);method=spec['methods'][0]
    ids=set(spec['question_ids']);results={};refs={}
    for folder in ('results','calls','raw_generations'):
        assert {p.stem for p in (run/folder/method).glob('*.json')}==ids
    # 步骤1：正式输入、账本与参考任务计分都经过原校验器。
    for row in data:results[row['question_id']]=validate(run,row,method,spec)
    parent=Path(spec['parent_run'])
    for budget,source in [('K32',parent),('K16',OLD_QWEN)]:
        for m in ('BASE-Uniform','BASE-BLIP-TopK','RD-1.2'):
            refs[budget+'__'+m]={r['question_id']:read_json(source/'results'/m/(r['question_id']+'.json')) for r in data}
    # 步骤2：一次冻结比较集合；不根据得分追加生成或更换统计分母。
    metrics={}
    for layer in ('all','short','medium','long'):
        group=[results[r['question_id']] for r in data if layer=='all' or r['stratum']==layer]
        c=sum(r['correct'] for r in group);metrics[layer]=dict(n=len(group),correct=c,accuracy=c/len(group),wilson_95=wilson(c,len(group)))
    summary=dict(status='completed',method=method,configuration=spec['configuration'],diagnostic=spec['diagnostic'],
        new_answer_calls=200,results=metrics,comparisons={name:compare(data,results,ref) for name,ref in refs.items()},
        parse_failures=[q for q,r in results.items() if r['answer']['parsed_answer'] is None],
        qa_seconds=distribution([r['answer']['observed'].get('qa_seconds',r['answer']['observed']['qa_through_generation_seconds']) for r in results.values()]),
        protocol_sha256=sha256(run/'protocol.json'),utc=utc(),
        scope='Frozen development200; same Qwen reference protocol; cached source-frame inputs, not raw-video E2E')
    durable(run/'summary.json',summary);return summary


def worker(run,gpu,jobs,events,stop):
    """接口：单卡常驻后端，逐题一次生成；任何异常停止派发，不自动重试。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    offline_environment();run=Path(run);active=None
    with (run/'logs'/f'gpu_{gpu}.log').open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import torch
            from PIL import Image
            from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
            from .qwen_adapter import FrozenInputQwen
            from .observe import observe
            from videoqa_lmms_bridge.bridge import synthetic_row
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            t=time.perf_counter();backend=QwenFA2VideoBackend();adapter=FrozenInputQwen(backend);load=time.perf_counter()-t
            # 步骤1：合成预热另记，不以开发题答案选配置。
            frames=[Image.new('RGB',(896,672),(40,i*7,150)) for i in range(32)]
            t=time.perf_counter();syn=observe(backend,adapter,frames,synthetic_row(32),{})
            for im in frames:im.close()
            durable(run/'workers'/f'{gpu}_{os.getpid()}.json',dict(gpu=gpu,pid=os.getpid(),load_seconds=load,
                warmup_seconds=time.perf_counter()-t,synthetic_generations=1,warmup=syn,load_report=backend.load_report))
            spec=read_json(run/'protocol.json');method=spec['methods'][0];events.put(dict(type='ready',gpu=gpu))
            # 步骤2：复用已验收的精确解码、调用账本、真实生成观测和原始返回检查点。
            while not stop.is_set():
                row=jobs.get()
                if row is None:break
                active=row['question_id'];events.put(dict(type='active',gpu=gpu,question_id=active))
                process(run,row,method,backend,adapter,gpu,spec)
                events.put(dict(type='completed',gpu=gpu,question_id=active));active=None
        except BaseException as exc:
            stop.set();traceback.print_exc()
            durable(run/'failures'/f'{gpu}_{time.time_ns()}.json',dict(error=repr(exc),question_id=active,traceback=traceback.format_exc(),utc=utc()))
            events.put(dict(type='failed',gpu=gpu,error=repr(exc)))


def execute(run):
    """接口：脱离会话协调200题；前五题只按运行正确性放行，完成恢复不加载模型。"""
    from videoqa_runtime.baseline_run import available_gpus
    run=Path(run)
    with exclusive(run/'coordinator.lock'):
        spec=verify(run);data=selected_rows(spec);byid={r['question_id']:r for r in data};method=spec['methods'][0]
        done=set();pinned={};start=time.perf_counter();phase='checking';active={};gpus=[];failure=None
        def publish():
            elapsed=time.perf_counter()-start
            durable(run/'status.json',dict(status=phase,completed=len(done),total=200,pid=os.getpid(),session_id=os.getsid(0),
                active=active,gpus=gpus,elapsed_seconds=elapsed,eta_seconds=(200-len(done))*elapsed/len(done) if done else None,failure=failure,utc=utc()))
        # 步骤1：已有返回只恢复后处理；不确定调用会在recover_call处停止，调用ID不能扩增。
        calls=list((run/'calls'/method).glob('*.json'));assert len(calls)<=200
        for p in calls:
            assert p.stem in byid;recover_call(run,byid[p.stem],method,spec)
            c=read_json(p);assert c['status']=='completed' and c['gpu'] in spec['allowed_gpus'];pinned[p.stem]=c['gpu']
        for p in (run/'results'/method).glob('*.json'):
            assert p.stem in byid;validate(run,byid[p.stem],method,spec);done.add(p.stem)
        if len(done)==200:return summarize(run)
        required={g for q,g in pinned.items() if q not in done}
        while True:
            try:gpus,states=available_gpus(spec['allowed_gpus'])
            except RuntimeError:gpus=[];states={}
            if gpus and required<=set(gpus):break
            phase='waiting_resources';publish();time.sleep(15)
        for d in ('logs','workers','failures'):(run/d).mkdir(exist_ok=True)
        durable(run/'gpu_preflight.json',dict(allowed=spec['allowed_gpus'],actual=gpus,states=states,utc=utc()))
        ctx=mp.get_context('spawn');jobs={g:ctx.Queue() for g in gpus};events=ctx.Queue();stop=ctx.Event()
        children={};ready=set();busy=set();pending=[]
        def schedule():
            nonlocal phase,pending
            pilot=spec['pilot_question_ids']
            if any(q not in done for q in pilot):phase='pilot';pending=[q for q in pilot if q not in done]
            else:
                for q in pilot:validate(run,byid[q],method,spec)
                durable(run/'pilot_acceptance.json',dict(status='passed',question_ids=pilot,accuracy_not_gate=True))
                phase='running'
                pending=sorted([q for q in byid if q not in done],key=lambda q:(-read_json(run/'selections'/method/(q+'.json'))['selection']['video']['duration_seconds'],q))
        def refill():
            for gpu in gpus:
                if gpu in busy:continue
                q=next((q for q in pending if q not in pinned or pinned[q]==gpu),None)
                if q is None:continue
                pending.remove(q);busy.add(gpu);jobs[gpu].put(byid[q])
        # 步骤2：先行/全量切换均由后台协调器完成，观察连接中断不触发重启。
        try:
            phase='loading'
            for gpu in gpus:
                p=ctx.Process(target=worker,args=(str(run),gpu,jobs[gpu],events,stop));p.start();children[gpu]=p
            publish()
            while len(done)<200:
                try:e=events.get(timeout=5)
                except queue.Empty:e=None
                if e:
                    if e['type']=='failed':raise RuntimeError(str(e))
                    if e['type']=='ready':
                        ready.add(e['gpu'])
                        if len(ready)==len(gpus):schedule();refill()
                    elif e['type']=='active':active[e['gpu']]=e['question_id']
                    elif e['type']=='completed':
                        done.add(e['question_id']);busy.remove(e['gpu']);active.pop(e['gpu'],None)
                        if phase=='pilot' and not pending and not busy:schedule()
                        refill()
                for gpu,p in children.items():
                    if p.exitcode is not None:raise RuntimeError(f'Worker {gpu} exited {p.exitcode}')
                publish()
        except BaseException as exc:
            failure=repr(exc);stop.set();phase='paused';publish();raise
        finally:
            for q in jobs.values():q.put(None)
            while any(p.is_alive() for p in children.values()):
                for p in children.values():p.join(timeout=1)
                publish()
        summary=summarize(run);phase='completed';publish()
        durable(run/'execution.json',dict(status='completed',wall_seconds=time.perf_counter()-start,actual_gpus=gpus,new_answer_calls=200,utc=utc()))
        return summary


def audit(run):
    """接口：完成后实际执行恢复入口，验证600份原始记录不变、生成不增加。"""
    run=Path(run)
    before={str(p.relative_to(run)):sha256(p) for d in ('results','calls','raw_generations') for p in (run/d).glob('*/*.json')}
    assert len(before)==600
    execute(run)
    after={str(p.relative_to(run)):sha256(p) for d in ('results','calls','raw_generations') for p in (run/d).glob('*/*.json')}
    assert before==after
    durable(run/'completion_audit.json',dict(status='passed',unchanged_files=600,extra_generations=0,utc=utc()))
