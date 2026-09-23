#!/usr/bin/env python3
"""Video-MME全量跨模型入口：准备/独立阶段/后台串联/只读汇总。"""
import argparse
import os
import re
import sys
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT


def main():
    """接口：固定全量范围；阶段参数不允许暗中缩小问答数量或改变方法。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=['prepare','llava','qwen25vl','all','report'],default='all')
    parser.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id):raise ValueError('Invalid run ID')
    run=ROOT/'outputs/full_videoqa'/args.run_id
    if args.stage=='prepare':
        from videoqa_full.prepare import prepare
        prepare(run);return
    if not (run/'protocol.json').exists():raise ValueError('Run prepare before GPU execution')
    if args.stage=='report':
        from videoqa_full.report import summarize
        summarize(run);return
    if not args.resume and any((run/m/'calls').exists() for m in ('llava','qwen25vl')):
        raise ValueError('Existing execution requires explicit --resume')
    from videoqa_full.run import execute_all,execute_stage
    gpus=args.gpus.split(',')
    if args.stage=='all':execute_all(run,gpus)
    else:execute_stage(run,args.stage,gpus)


if __name__=='__main__':main()
