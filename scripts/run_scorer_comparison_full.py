"""评分器全量薄入口：跨环境显式阶段、后台自动推进与完成恢复。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import argparse
import subprocess
import sys
import re
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json
from videoqa_full.state import durable,exclusive,utc
from videoqa_scorer_comparison import preparation as prep,runtime


def wait_for_model(root):
    """接口：独立后台等待已校验权重原子发布，下载未完成不占GPU或假称准备通过。"""
    acceptance=prep.MODEL/'download_acceptance.json'
    while not (prep.MODEL/'model.safetensors').exists() or not acceptance.exists():
        durable(root/'pipeline_status.json',dict(status='waiting_model_download',stage='model',
            pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        time.sleep(15)
    from videoqa_methods.retrieval_frontends import REVISION,WEIGHT_SHA256
    verified=read_json(acceptance)
    if verified['status']!='verified' or verified['revision']!=REVISION or verified['sha256']!=WEIGHT_SHA256:
        raise ValueError('Downloaded model revision or official verification differs')


def pipeline_commands(run_id,gpus):
    """接口：检索/问答环境隔离，所有子阶段显式恢复，不依赖会话交互放行。"""
    stages=[('preflight',prep.PYTHON),('coarse',prep.PYTHON),('refine',prep.PYTHON),('mixed',prep.PYTHON),
        ('prepare-qwen32',runtime.PYTHONS['qwen32']),('qwen32',runtime.PYTHONS['qwen32']),
        ('prepare-llava8',runtime.PYTHONS['llava8']),('llava8',runtime.PYTHONS['llava8']),
        ('report',runtime.PYTHONS['qwen32'])]
    return [(stage,[py,str(Path(__file__).resolve()),'--run-id',run_id,'--stage',stage,'--gpus',*gpus,'--resume']) for stage,py in stages]


def main():
    """接口：准备/评分/选择/两后端/报告安全恢复；失败暂停，不自动改精度或参数。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--gpus',nargs='+',default=list(map(str,range(8))))
    p.add_argument('--stage',default='all',choices=['all','controls','preflight','coarse','refine','mixed','prepare-qwen32','qwen32','prepare-llava8','llava8','report','audit'])
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id):raise ValueError('Invalid run ID')
    if not a.gpus or len(set(a.gpus))!=len(a.gpus) or not set(a.gpus)<=set(map(str,range(8))):raise ValueError('Invalid GPU set')
    root=ROOT/'outputs/full_videoqa'/a.run_id;root.mkdir(parents=True,exist_ok=True)
    if list(root.glob('qa/*/calls/*/*.json')) and not a.resume:raise ValueError('Existing calls require explicit resume')
    if a.stage=='all':
        with exclusive(root/'pipeline.lock'):
            durable(root/'launcher.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            if (root/'comparison.json').exists():
                for profile in prep.PROFILES:runtime.audit(root/'qa'/profile)
                durable(root/'restore_audit.json',dict(new_generations=0,utc=utc()));return
            wait_for_model(root)
            for stage,command in pipeline_commands(a.run_id,a.gpus):
                durable(root/'pipeline_status.json',dict(status='running',stage=stage,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                try:subprocess.run(command,cwd=ROOT,check=True)
                except BaseException as exc:
                    durable(root/'pipeline_status.json',dict(status='paused',stage=stage,error=repr(exc),utc=utc()));raise
            durable(root/'pipeline_status.json',dict(status='completed',utc=utc()))
    elif a.stage=='controls':prep.controls(root)
    elif a.stage=='preflight':prep.preflight(root)
    elif a.stage in ('coarse','refine','qwen32','llava8'):
        from videoqa_scorer_comparison.coordinator import run_stage
        run_stage(root,a.stage,a.gpus)
    elif a.stage=='mixed':
        from videoqa_scorer_comparison.coordinator import choose_gpus
        free,_=choose_gpus(a.gpus);os.environ['CUDA_VISIBLE_DEVICES']=free[0];prep.mixed_diagnostics(root)
    elif a.stage.startswith('prepare-'):runtime.prepare_inputs(root,a.stage.removeprefix('prepare-'),a.gpus)
    elif a.stage=='report':
        from videoqa_scorer_comparison.report import summarize
        summarize(root)
    else:
        for profile in prep.PROFILES:runtime.audit(root/'qa'/profile)


if __name__=='__main__':main()
