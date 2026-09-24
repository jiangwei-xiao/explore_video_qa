"""独立后台阶段驱动；模型选择与问答分进程，连接不参与调度。"""
import contextlib
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import time
import traceback
from videoqa_runtime.common import ROOT,read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,rows,utc
from .selection import select_question,precheck,config,configure_visual_precision


def selection_worker(run,gpu,jobs,events,stop):
    """接口：每卡只加载BLIP，常驻执行补查纯视觉计算；无真实问答。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    offline_environment();run=Path(run)
    with (run/'logs'/f'selection_gpu_{gpu}.log').open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import torch
            from videoqa_methods.density_pipeline import DensityScorer
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            precision=configure_visual_precision(torch)
            scorer=DensityScorer(read_json(ROOT/'configs/local_model_inventory.json')['models']['blip']['path'])
            check=scorer.synthetic_acceptance()
            durable(run/'selection_workers'/f'{gpu}_{os.getpid()}.json',dict(gpu=gpu,pid=os.getpid(),synthetic=check,visual_precision=precision))
            # 步骤1：合成调用另记，真实题仅纯视觉，不计算新ITM。
            scorer.counts=dict(itm_batches=0,itm_frames=0,visual_batches=0,visual_frames=0)
            while not stop.is_set():
                row=jobs.get()
                if row is None:break
                events.put(dict(kind='active',gpu=gpu,question_id=row['question_id']))
                result=select_question(run,row,scorer)
                assert scorer.counts['itm_frames']==0
                events.put(dict(kind='done',gpu=gpu,question_id=row['question_id'],new_count=result['new_count']))
            durable(run/'selection_workers'/f'{gpu}_{os.getpid()}_counts.json',scorer.counts)
        except BaseException as exc:
            stop.set();traceback.print_exc();events.put(dict(kind='failed',gpu=gpu,error=repr(exc)))
            durable(run/'failures'/f'selection_{gpu}_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),utc=utc()))


def select_all(run):
    """接口：冻结代码与粗资产后独立准备2700题；阶段结束确保释放BLIP进程。"""
    from videoqa_runtime.baseline_run import available_gpus
    run=Path(run)
    with exclusive(run/'selection.lock'):
        summary=read_json(run/'precheck_summary.json');assert summary['status']=='passed'
        identity=dict(config=config(),visual_precision=dict(matmul_allow_tf32=False,cudnn_allow_tf32=False,feature_dtype='float32'),
            precheck_sha256=sha256(run/'precheck_summary.json'),
            code={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'src/videoqa_budget32'/n for n in ('selection.py','frames.py','driver.py')]+
                [ROOT/'src/videoqa_methods/density_pipeline.py',ROOT/'src/videoqa_audit/density_reward_decay.py',ROOT/'src/videoqa_audit/local_distance_scale.py']})
        if (run/'selection_protocol.json').exists():assert read_json(run/'selection_protocol.json')==identity
        else:durable(run/'selection_protocol.json',identity)
        # 步骤1：正式视觉计算前核验视频及BLIP模型字节，不能只依赖历史存在性。
        from .selection import SOURCE,INDEX
        old=read_json(SOURCE/'protocol.json')
        blip_path=read_json(ROOT/'configs/local_model_inventory.json')['models']['blip']['path']
        model_files={p:x for p,x in old['model_files'].items() if p.startswith(blip_path+'/')}
        assert model_files
        for p,x in model_files.items():assert sha256(p)==x['sha256']
        for p in (INDEX/'uniform_records').glob('*.json'):
            record=read_json(p);assert sha256(record['video']['path'])==record['source_video_sha256']
        for d in ('logs','selection_workers','failures'):(run/d).mkdir(parents=True,exist_ok=True)
        data=rows();complete=[];pending=[]
        for row in data:
            q=row['question_id'];assert sha256(run/'precheck'/(q+'.json'))==summary['records'][q]
            p=run/'selection_checkpoints'/(q+'.json')
            if p.exists():
                item=read_json(p)
                for rel,h in item['files'].items():assert sha256(run/rel)==h
                complete.append(q)
            else:pending.append(row)
        # 步骤2：仅用实际空闲卡；资源等待状态持久化，不抢占。
        while pending:
            try:gpus,states=available_gpus([str(i) for i in range(8)])
            except RuntimeError:gpus=[]
            if gpus:break
            durable(run/'selection_status.json',dict(status='waiting_resources',utc=utc()));time.sleep(15)
        start=time.perf_counter();active={}
        if pending:
            pending.sort(key=lambda r:(-read_json(INDEX/'uniform_records'/(r['video_id']+'.json'))['video']['duration_seconds'],r['question_id']))
            ctx=mp.get_context('spawn');jobs=ctx.Queue();events=ctx.Queue();stop=ctx.Event()
            for row in pending:jobs.put(row)
            for _ in gpus:jobs.put(None)
            children=[ctx.Process(target=selection_worker,args=(str(run),g,jobs,events,stop)) for g in gpus]
            for p in children:p.start()
            try:
                while len(complete)<2700:
                    try:event=events.get(timeout=5)
                    except queue.Empty:event=None
                    if event:
                        if event['kind']=='failed':raise RuntimeError(str(event))
                        if event['kind']=='active':active[event['gpu']]=event['question_id']
                        if event['kind']=='done':complete.append(event['question_id']);active.pop(event['gpu'],None)
                    if any(p.exitcode not in (None,0) for p in children):raise RuntimeError('Selection worker exited abnormally')
                    if all(p.exitcode is not None for p in children) and event is None:raise RuntimeError('Incomplete selection after worker exit')
                    durable(run/'selection_status.json',dict(status='running',completed=len(complete),total=2700,
                        gpus=gpus,active=active,pid=os.getpid(),session_id=os.getsid(0),seconds=time.perf_counter()-start,utc=utc()))
            finally:
                stop.set()
                for p in children:p.join()
        # 步骤3：全部选择文件复验并冻结，允许下一阶段加载Qwen。
        hashes={}
        for row in data:
            p=run/'selection_checkpoints'/(row['question_id']+'.json');item=read_json(p)
            for rel,h in item['files'].items():assert sha256(run/rel)==h
            hashes[row['question_id']]=sha256(p)
        durable(run/'selection_summary.json',dict(status='passed',questions=2700,records=hashes,new_answer_calls=0,
            new_itm_calls=0,seconds=time.perf_counter()-start,utc=utc()))


def all_stages(run):
    """接口：后台串行门控CPU、BLIP、Qwen及报告，任一异常暂停而非重启。"""
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    script=ROOT/'scripts/run_budget32_experiments.py'
    pyold='/App/conda/envs/explore_video_qa/bin/python';pyq='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'
    with exclusive(run/'pipeline.lock'):
        try:
            # 步骤1：子阶段独立进程使环境与显存生命周期清晰。
            for stage,python in [('precheck',pyold),('selection',pyold),('synthetic',pyq),('prepare',pyq),('qa',pyq),('audit',pyq)]:
                if stage=='synthetic' and (run/'synthetic_acceptance.json').exists():
                    from .run import critical_code
                    syn=read_json(run/'synthetic_acceptance.json')
                    assert syn['status']=='passed' and syn['code_sha256']==critical_code()
                    continue
                durable(run/'pipeline_status.json',dict(status='running',stage=stage,pid=os.getpid(),session_id=os.getsid(0),utc=utc()))
                subprocess.run([python,str(script),'--run-id',run.name,'--stage',stage,'--resume'],cwd=ROOT,check=True)
            durable(run/'pipeline_status.json',dict(status='completed',utc=utc()))
        except BaseException as exc:
            durable(run/'pipeline_status.json',dict(status='paused',error=repr(exc),utc=utc()));raise
