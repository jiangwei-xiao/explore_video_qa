#!/usr/bin/env python3
"""32帧独立阶段入口；全量请使用nohup＋setsid并关闭stdin。"""
import argparse
import os
import re
import sys
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT


def main():
    """接口：运行ID显式给定，恢复不改变预算/方法/源资产。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=['all','precheck','selection','synthetic','prepare','qa','report','audit'],default='all')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id)
    run=ROOT/'outputs/full_videoqa'/args.run_id
    # 步骤1：存在真实调用时必须显式恢复；账本会禁止重复生成。
    if (run/'calls').exists() and not args.resume:raise ValueError('Existing calls require --resume')
    if args.stage=='all':
        from videoqa_budget32.driver import all_stages
        all_stages(run)
    elif args.stage=='precheck':
        from videoqa_budget32.selection import precheck
        precheck(run)
    elif args.stage=='selection':
        from videoqa_budget32.driver import select_all
        select_all(run)
    elif args.stage in ('synthetic','prepare','qa'):
        from videoqa_budget32 import run as runner
        getattr(runner,{'qa':'execute'}.get(args.stage,args.stage))(run)
    else:
        from videoqa_budget32 import report
        getattr(report,{'report':'summarize','audit':'audit'}[args.stage])(run)


if __name__=='__main__':main()
