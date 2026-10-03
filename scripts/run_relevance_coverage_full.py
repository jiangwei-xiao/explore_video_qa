"""exp12全量独立入口：后台完成选择、合成验收、导入、2500新问答与统计。"""
import argparse
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import re
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_budget32 import relevance_full as full
from videoqa_full.state import durable,exclusive,utc


def wait_for_selection(run,pid):
    """接口：接续已确认存活的CPU准备，不启动第二份任务；原进程失败则暂停。"""
    proc=Path('/proc')/str(pid)/'cmdline'
    # 步骤1：检查真实进程及命令，而非只依赖旧状态文件或等待时间。
    while not (run/'selection_summary.json').exists():
        if not proc.exists():raise RuntimeError('Existing selection exited without completion')
        args=proc.read_bytes().decode().split('\0')
        assert '--stage' in args and args[args.index('--stage')+1]=='selection'
        assert '--run-id' in args and args[args.index('--run-id')+1]==run.name
        durable(run/'pipeline_status.json',dict(status='waiting_existing_selection',selection_pid=pid,
            pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        time.sleep(5)
    # 步骤2：等待原准备释放锁；实际完整性由随后selection阶段再次核验。
    while True:
        try:
            with exclusive(run/'selection.lock'):return
        except BlockingIOError:
            if not proc.exists():raise RuntimeError('Selection lock owned by another process')
            time.sleep(1)


def main():
    """接口：固定运行编号及阶段，观察中断不重启；已存在调用必须显式恢复。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--stage',choices=['all','selection','synthetic','prepare','qa','report','audit'],default='all')
    p.add_argument('--gpus',nargs='+',default=list(map(str,range(8))));p.add_argument('--resume',action='store_true')
    p.add_argument('--wait-selection-pid',type=int)
    args=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id)
    assert args.gpus and len(set(args.gpus))==len(args.gpus) and set(args.gpus)<=set(map(str,range(8)))
    run=ROOT/'outputs/full_videoqa'/args.run_id;run.mkdir(parents=True,exist_ok=True)
    if (run/'calls').exists() and not args.resume:raise ValueError('Existing calls require --resume')
    release=full.release_check()
    assert run.resolve()==(ROOT/release['full_preflight']).parent.resolve(), 'Run must match the frozen preflight'
    # 步骤1：分阶段独立进程避免CPU环境/模型加载互相污染；后台自行推进，不等待交互。
    if args.stage=='all':
        with exclusive(run/'pipeline.lock'):
            durable(run/'launcher_identity.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            if args.wait_selection_pid is not None:
                assert args.wait_selection_pid>0
                wait_for_selection(run,args.wait_selection_pid)
            for stage in ['selection','synthetic','prepare','qa','report','audit']:
                if stage=='synthetic' and (run/'synthetic_acceptance.json').exists():continue
                py='/App/conda/envs/explore_video_qa/bin/python' if stage=='selection' else full.PYTHON
                command=[py,str(Path(__file__).resolve()),'--run-id',args.run_id,'--stage',stage,'--gpus',*args.gpus]
                if args.resume or stage in ['report','audit']:command.append('--resume')
                durable(run/'pipeline_status.json',dict(status='running',stage=stage,utc=utc()))
                try:subprocess.run(command,cwd=ROOT,check=True)
                except BaseException as exc:
                    durable(run/'pipeline_status.json',dict(status='paused',stage=stage,error=repr(exc),utc=utc()));raise
            durable(run/'pipeline_status.json',dict(status='completed',case_review='pending',utc=utc()))
    # 步骤2：正式生成只复用已验收后端；该入口不修改任何选择公式。
    elif args.stage=='selection':full.selection_all(run)
    elif args.stage=='synthetic':
        from videoqa_budget32 import run as runtime
        # 此阶段独立子进程只收窄可用设备列表，不改共享文件或任何数值配置。
        runtime.GPUS=tuple(args.gpus)
        runtime.synthetic(run)
    elif args.stage=='prepare':full.prepare(run,args.gpus)
    elif args.stage=='qa':full.execute(run)
    elif args.stage=='report':full.summarize(run)
    else:full.audit(run)


if __name__=='__main__':main()
