#!/usr/bin/env python3
"""exp07独立全量入口；后台自动准备、验收、生成及汇总，旧200题永不重答。"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_budget32 import prefix_full as full


def main():
    """接口：显式运行编号、GPU范围及恢复；子阶段异常中断，不自动追加调用。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=['all','selection','synthetic','prepare','qa','report','audit'],default='all')
    parser.add_argument('--gpus',nargs='+',default=list(map(str,range(8))))
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args(); assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id)
    run = ROOT/'outputs/full_videoqa'/args.run_id; run.mkdir(parents=True,exist_ok=True)
    if (run/'calls').exists() and not args.resume: raise ValueError('Existing calls require --resume')
    # 步骤1：各阶段独立进程，合成GPU显存释放后才启动正式多卡；选择沿用原CPU环境。
    if args.stage == 'all':
        from videoqa_full.state import exclusive,durable,utc
        with exclusive(run/'pipeline.lock'):
            durable(run/'launcher_identity.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            for stage in ['selection','synthetic','prepare','qa','report','audit']:
                if stage == 'synthetic' and (run/'synthetic_acceptance.json').exists(): continue
                py = '/App/conda/envs/explore_video_qa/bin/python' if stage == 'selection' else full.PYTHON
                cmd = [py,str(Path(__file__).resolve()),'--run-id',args.run_id,'--stage',stage,'--gpus',*args.gpus]
                if args.resume or stage in ('report','audit'): cmd.append('--resume')
                durable(run/'pipeline_status.json',dict(stage=stage,status='running',utc=utc()))
                try: subprocess.run(cmd,check=True,cwd=ROOT)
                except BaseException as exc:
                    durable(run/'pipeline_status.json',dict(stage=stage,status='paused',error=repr(exc),utc=utc())); raise
            durable(run/'pipeline_status.json',dict(status='completed',utc=utc()))
    # 步骤2：单阶段入口可审计恢复，禁止跳过准备指纹或重新生成已完成答案。
    elif args.stage == 'selection': full.selection_all(run)
    elif args.stage == 'synthetic':
        from videoqa_budget32.run import synthetic
        synthetic(run)
    elif args.stage == 'prepare': full.prepare(run,args.gpus)
    elif args.stage == 'qa': full.execute(run)
    elif args.stage == 'report': full.summarize(run)
    else: full.audit(run)


if __name__ == '__main__': main()
