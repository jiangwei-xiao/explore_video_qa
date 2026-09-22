#!/usr/bin/env python3
"""E1单版本正式200题入口；支持后台执行和不重复生成的恢复。"""
import os,sys,argparse
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_audit.e1_run import execute
from videoqa_audit.e1_report import summarize

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--resume',action='store_true');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    run=execute(a.run_id,a.gpus.split(','),a.resume,a.prepare_only)
    if not a.prepare_only:print(summarize(run)['accuracy']['all200'],flush=True)
