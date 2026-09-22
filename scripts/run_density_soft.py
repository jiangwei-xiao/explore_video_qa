#!/usr/bin/env python3
"""密度框架50题入口；可由setsid+nohup启动，恢复不重复生成。"""
import argparse
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_density_soft.density_run import execute

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run=execute(args.run_id,args.gpus.split(','),args.resume)
    from videoqa_density_soft.density_report import summarize
    summary=summarize(run)
    print(summary['accuracy']['Density-soft'],flush=True)
