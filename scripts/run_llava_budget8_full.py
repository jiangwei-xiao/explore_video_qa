"""独立后台入口：八帧五策略准备、验收、问答、汇总和完成恢复。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json
from videoqa_full.state import durable,exclusive,utc
from videoqa_generalization import llava_budget8 as experiment


def pipeline_commands(run_id,gpus):
    """接口：每个子阶段独立进程，显式恢复标记，避免阶段切换重复生成。"""
    return [(s,[experiment.LPY,str(Path(__file__).resolve()),'--run-id',run_id,'--stage',s,'--gpus',*gpus,'--resume'])
        for s in ['prepare','synthetic','answer','audit','report']]


def main():
    """接口：后台自动执行全部阶段；已完成恢复只审计，不重做准备或报告。"""
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=['all','prepare','synthetic','answer','report','audit'],default='all')
    parser.add_argument('--gpus',nargs='+',default=list(map(str,range(8))));parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id):raise ValueError('Invalid run ID')
    if not args.gpus or len(set(args.gpus))!=len(args.gpus) or not set(args.gpus)<=set(map(str,range(8))):raise ValueError('Invalid GPU list')
    root=ROOT/'outputs/full_videoqa'/args.run_id;root.mkdir(parents=True,exist_ok=True)
    if (root/'calls').exists() and not args.resume:raise ValueError('Existing calls require resume')
    if args.stage=='all':
        with exclusive(root/'pipeline.lock'):
            status=root/'pipeline_status.json'
            if status.exists() and read_json(status)['status']=='completed':experiment.audit(root);return
            durable(root/'launcher.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            # 步骤1：协调器不等待会话指令，阶段失败后暂停整条流水线。
            for stage,command in pipeline_commands(args.run_id,args.gpus):
                durable(status,dict(status='running',stage=stage,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                try:subprocess.run(command,cwd=ROOT,check=True)
                except BaseException as exc:
                    durable(status,dict(status='paused',stage=stage,error=repr(exc),utc=utc()));raise
            spec=read_json(root/'protocol.json')
            durable(status,dict(status='completed',new_calls=spec['max_calls'],method_results=13500,utc=utc()))
    elif args.stage=='prepare':experiment.prepare(root,args.gpus)
    elif args.stage=='synthetic':experiment.verify(root);experiment.synthetic(root,args.gpus)
    elif args.stage=='answer':experiment.execute(root)
    elif args.stage=='audit':experiment.audit(root)
    else:experiment.summarize(root)


if __name__=='__main__':main()
