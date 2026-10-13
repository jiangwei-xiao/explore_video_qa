"""后台多卡协调：长驻进程、先行门槛、心跳、物理GPU固定与失败暂停。"""
import os
import time
import queue
import traceback
import multiprocessing as mp
from pathlib import Path
from videoqa_runtime.common import read_json
from videoqa_full.state import durable,rows,utc,exclusive,pilot_ids
from .preparation import METHODS,BASE


def choose_gpus(allowed,required=(),status_path=None):
    """接口：只用实际空闲卡；原物理卡不可用时等待，不能抢占其他进程。"""
    from videoqa_runtime.baseline_run import available_gpus
    while True:
        try:free,states=available_gpus(allowed)
        except RuntimeError:free=[];states={}
        if free and set(required)<=set(free):return sorted(required)+[g for g in free if g not in required],states
        if status_path is not None:durable(status_path,dict(status='waiting_resources',required=list(required),utc=utc(),pid=os.getpid(),sid=os.getsid(0)))
        time.sleep(15)


def worker(root,stage,gpu,jobs,events,stop):
    """接口：一卡一个长期进程，模型加载/合成单列，错误即通知协调器暂停。"""
    root=Path(root);os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    started=time.perf_counter();active=None
    try:
        # 步骤1：在子进程导入模型，防止主进程CUDA状态跨spawn继承。
        if stage in ('coarse','refine'):
            from .preparation import initialize_scorer,coarse_video,refine_one
            initialize_scorer(root,gpu,stage)
        else:
            from .runtime import qa_initialize,process_question
            run=root/'qa'/stage;spec=read_json(run/'protocol.json');backend,adapter=qa_initialize(run,gpu)
        events.put(dict(type='ready',gpu=gpu))
        while not stop.is_set():
            row=jobs.get()
            if row is None:break
            members=row if isinstance(row,list) else [row]
            active=members[0]['question_id']
            events.put(dict(type='active',gpu=gpu,question_id=active))
            # 步骤2：整个视频题目任务不跨卡，同题不同方法共享一个GPU和进程。
            if stage=='coarse':coarse_video(root,members)
            elif stage=='refine':refine_one(root,row)
            else:process_question(run,row,backend,adapter,gpu,spec)
            events.put(dict(type='completed',gpu=gpu,question_ids=[r['question_id'] for r in members]))
            active=None
        if stage in ('coarse','refine'):
            from . import preparation
            durable(root/'worker_totals'/(stage+'_'+gpu+'_'+str(os.getpid())+'.json'),
                dict(stage=stage,gpu=gpu,model_calls=preparation._SCORER.counts,
                     peak_allocated_gib=preparation._SCORER.torch.cuda.max_memory_allocated()/2**30,
                     wall_seconds=time.perf_counter()-started,utc=utc()))
    except BaseException as exc:
        stop.set();traceback.print_exc()
        from . import preparation
        counts=None if preparation._SCORER is None else dict(preparation._SCORER.counts)
        durable(root/'failures'/(stage+'_gpu_'+gpu+'_'+str(os.getpid())+'.json'),dict(error=repr(exc),
            question_id=active,elapsed_seconds=time.perf_counter()-started,scorer_call_attempts=counts,
            traceback=traceback.format_exc(),utc=utc()))
        events.put(dict(type='failed',gpu=gpu,error=repr(exc)))


def run_stage(root,stage,gpus):
    """接口：后台自动先行放行与时长队列；恢复跳过完成项、不重复不确定生成。"""
    root=Path(root);data=rows();bank={r['question_id']:r for r in data};pinned={};done=set()
    if stage in ('coarse','refine'):
        for row in data:
            q=row['question_id']
            folders=('coarse',) if stage=='coarse' else ('qwen32','llava8')
            if all((root/'checkpoints'/folder/(q+'.json')).exists() for folder in folders):
                for folder in folders:
                    from videoqa_runtime.common import sha256
                    check=read_json(root/'checkpoints'/folder/(q+'.json'))
                    for rel,h in check['files'].items():
                        if sha256(root/rel)!=h:raise ValueError('Completed scoring artifact changed')
                    for path,h in check['sources'].items():
                        p=root/'preflight.json' if path=='preflight' else Path(path)
                        if sha256(p)!=h:raise ValueError('Completed scoring origin changed')
                done.add(q)
    else:
        from .runtime import verify,recover_group,import_answers,validate_result,audit
        run=root/'qa'/stage;spec=verify(run);import_answers(run,spec)
        for row in data:
            q=row['question_id'];groups=read_json(run/'groups'/(q+'.json'))
            for group in groups:
                if not group['historical_result']:recover_group(run,row,spec,group)
            cards={read_json(p)['gpu'] for p in (run/'calls').glob('*/'+q+'.json')}
            if len(cards)>1:raise ValueError('Partial question moved GPU')
            if cards:pinned[q]=next(iter(cards))
            if all((run/'results'/m/(q+'.json')).exists() for m in METHODS):
                for m in METHODS:validate_result(run,row,m,spec)
                done.add(q)
    if len(done)==2700:
        if stage not in ('coarse','refine'):audit(root/'qa'/stage)
        return
    required={g for q,g in pinned.items() if q not in done}
    actual,states=choose_gpus(gpus,required,root/'status.json');actual=actual[:min(len(actual),2700-len(done))]
    durable(root/'stage_gpu'/ (stage+'.json'),dict(actual=actual,states=states,utc=utc()))
    ctx=mp.get_context('spawn');events=ctx.Queue();stop=ctx.Event();jobs={g:ctx.Queue() for g in actual}
    children={};ready=set();busy=set();active={};pilot=pilot_ids(data);phase='loading';pending=[];started=time.perf_counter();initial=len(done)
    durations={r['question_id']:read_json(BASE/'pools'/(r['question_id']+'.json'))['video']['duration_seconds'] for r in data}
    def schedule():
        nonlocal pending,phase
        missing=[q for q in pilot if q not in done]
        phase='pilot' if missing else 'running'
        pending=missing or sorted((q for q in bank if q not in done),key=lambda q:(-durations[q],q))
        if not missing:durable(root/'pilot_acceptance'/(stage+'.json'),dict(status='passed',question_ids=pilot,accuracy_not_gate=True,utc=utc()))
    def refill():
        for g in actual:
            if g in busy:continue
            q=next((q for q in pending if q not in pinned or pinned[q]==g),None)
            if q is not None:
                pending.remove(q);busy.add(g)
                if stage=='coarse':
                    group=[r for r in data if r['video_id']==bank[q]['video_id'] and r['question_id'] not in done]
                    for r in group:
                        if r['question_id'] in pending:pending.remove(r['question_id'])
                    jobs[g].put(group)
                else:jobs[g].put(bank[q])
    with exclusive(root/'locks'/(stage+'.lock')):
        try:
            # 步骤1：所有常驻模型与合成验收成功后，再进入固定先行批次。
            for g in actual:
                child=ctx.Process(target=worker,args=(str(root),stage,g,jobs[g],events,stop));child.start();children[g]=child
            while len(done)<2700:
                try:event=events.get(timeout=5)
                except queue.Empty:event=None
                if event:
                    g=event['gpu']
                    if event['type']=='failed':raise RuntimeError(str(event))
                    if event['type']=='ready':
                        ready.add(g)
                        if len(ready)==len(actual):schedule();refill()
                    elif event['type']=='active':active[g]=event['question_id']
                    elif event['type']=='completed':
                        done.update(event['question_ids']);busy.remove(g);active.pop(g,None)
                        if phase=='pilot' and not pending and not busy:schedule()
                        refill()
                for g,child in children.items():
                    if child.exitcode is not None:raise RuntimeError('Worker exited before stage completed: '+g)
                elapsed=time.perf_counter()-started;new=len(done)-initial
                durable(root/'status.json',dict(stage=stage,status=phase,completed=len(done),total=2700,
                    pid=os.getpid(),sid=os.getsid(0),gpus=actual,active=active,wall_seconds=elapsed,
                    eta_seconds=None if new<7 else elapsed/new*(2700-len(done)),utc=utc()))
        except BaseException as exc:
            stop.set();durable(root/'status.json',dict(stage=stage,status='paused',completed=len(done),error=repr(exc),utc=utc()));raise
        finally:
            # 步骤2：协调器失败不遗留无监督生成；保留不确定账本，恢复禁止重发。
            for g in actual:jobs[g].put(None)
            for child in children.values():
                child.join(20)
                if child.is_alive():child.terminate();child.join()
    if stage not in ('coarse','refine'):audit(root/'qa'/stage)
    durable(root/'stage_completion'/(stage+'.json'),dict(status='completed',questions=2700,
        actual_gpus=actual,wall_seconds=time.perf_counter()-started,utc=utc()))
