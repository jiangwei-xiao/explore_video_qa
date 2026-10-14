"""评分器对照执行恢复：已确认零正式生成时修复JSON容器校验，不重算资产。"""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.scorer_comparison_recovery import install_recovery,validate_imports,recovery_worker
from videoqa_scorer_comparison import runtime


def main():
    """接口：剩余两配置按原顺序后台完成；任何不确定生成立即暂停，不增加额度。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--gpus',nargs='+',default=list(map(str,range(8))))
    p.add_argument('--stage',choices=['all','check','qwen32','prepare-llava8','llava8','report','audit'],default='all');a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id) or not a.gpus or len(set(a.gpus))!=len(a.gpus) or not set(a.gpus)<=set(map(str,range(8))):raise ValueError('Invalid run/GPU scope')
    root=ROOT/'outputs/full_videoqa'/a.run_id;install_recovery(root)
    if a.stage=='all':
        with exclusive(root/'pipeline.lock'):
            stages=[('check',runtime.PYTHONS['qwen32']),('qwen32',runtime.PYTHONS['qwen32']),
                ('prepare-llava8',runtime.PYTHONS['llava8']),('llava8',runtime.PYTHONS['llava8']),('report',runtime.PYTHONS['qwen32'])]
            durable(root/'operational_recovery/launcher.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            for stage,python in stages:
                durable(root/'pipeline_status.json',dict(status='running',stage=stage,operational_recovery='JSON identity container',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                try:subprocess.run([python,str(Path(__file__).resolve()),'--run-id',a.run_id,'--stage',stage,'--gpus',*a.gpus],cwd=ROOT,check=True)
                except BaseException as exc:
                    durable(root/'pipeline_status.json',dict(status='paused',stage=stage,error=repr(exc),utc=utc()));raise
            durable(root/'pipeline_status.json',dict(status='completed',operational_recovery='JSON identity container',utc=utc()))
    elif a.stage=='check':print('verified imported results',validate_imports(root),flush=True)
    elif a.stage=='prepare-llava8':runtime.prepare_inputs(root,'llava8',a.gpus)
    elif a.stage in ('qwen32','llava8'):
        from videoqa_scorer_comparison import coordinator
        coordinator.worker=recovery_worker
        coordinator.run_stage(root,a.stage,a.gpus)
    elif a.stage=='report':
        from videoqa_scorer_comparison.report import summarize
        summarize(root)
    else:
        for profile in ('qwen32','llava8'):runtime.audit(root/'qa'/profile)


if __name__=='__main__':main()
