"""exp07双模型16帧全量入口，按冻结797/1903划分后台独立运行。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import argparse
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json
from videoqa_full.state import durable,exclusive,utc
from videoqa_generalization import exp07_budget16 as experiment


def pipeline_commands(run_id,gpus):
    """接口：每个子阶段显式恢复，跨模型切换不被前组账本误拦截。"""
    stages=[('prepare',experiment.QPY),('import_qwen',experiment.QPY),('import_llava',experiment.LPY),
        ('synthetic_qwen',experiment.QPY),('synthetic_llava',experiment.LPY),
        ('qwen16',experiment.QPY),('llava16',experiment.LPY),('report',experiment.QPY),('audit',experiment.QPY)]
    return [(stage,[py,str(Path(__file__).resolve()),'--run-id',run_id,'--stage',stage,'--gpus',*gpus,'--resume']) for stage,py in stages]


def main():
    """接口：新运行/安全恢复统一入口；完成恢复只核验原始文件，不加载模型或覆盖报告。"""
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=['all','prepare','import_qwen','import_llava','synthetic_qwen','synthetic_llava','qwen16','llava16','report','audit'],default='all')
    parser.add_argument('--gpus',nargs='+',default=list(map(str,range(8))));parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id):raise ValueError('Invalid run ID')
    if not args.gpus or len(set(args.gpus))!=len(args.gpus) or not set(args.gpus)<=set(map(str,range(8))):raise ValueError('Invalid GPU scope')
    root=ROOT/'outputs/full_videoqa'/args.run_id;root.mkdir(parents=True,exist_ok=True)
    if any(root.glob('*/calls')) and not args.resume:raise ValueError('Existing calls require explicit resume')
    if args.stage=='all':
        with exclusive(root/'pipeline.lock'):
            if (root/'pipeline_status.json').exists() and read_json(root/'pipeline_status.json')['status']=='completed':
                experiment.audit_all(root);return
            durable(root/'launcher.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            # 步骤1：各环境/模型独立子进程，失败暂停全流水线，不改变输入或调用预算。
            for stage,command in pipeline_commands(args.run_id,args.gpus):
                durable(root/'pipeline_status.json',dict(status='running',stage=stage,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                try:subprocess.run(command,cwd=ROOT,check=True)
                except BaseException as exc:
                    durable(root/'pipeline_status.json',dict(status='paused',stage=stage,error=repr(exc),utc=utc()));raise
            durable(root/'pipeline_status.json',dict(status='completed',new_answer_calls=3806,imported=1594,utc=utc()))
    elif args.stage=='prepare':experiment.prepare(root)
    elif args.stage.startswith('import_'):experiment.cpu_import_check(root,args.stage.removeprefix('import_'))
    elif args.stage.startswith('synthetic_'):experiment.synthetic(root,args.stage.removeprefix('synthetic_'),args.gpus)
    elif args.stage in ['qwen16','llava16']:
        # 步骤2：两后端CPU导入与GPU合成均通过后，才进入真实答题队列。
        for kind in ['qwen','llava']:
            if read_json(root/('import_'+kind+'_acceptance.json'))['status']!='passed' or read_json(root/'synthetic'/kind/'acceptance.json')['status']!='passed':
                raise ValueError('Acceptance incomplete')
        experiment.prepare_stage(root,args.stage,args.gpus);experiment.execute(root/args.stage)
    elif args.stage=='report':experiment.summarize(root)
    else:experiment.audit_all(root)


if __name__=='__main__':main()
