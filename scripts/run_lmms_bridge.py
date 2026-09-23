#!/usr/bin/env python3
"""LMMs-Eval白天桥接入口；没有全量运行分支。"""
import argparse
import os
import sys
import re
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--phase',choices=['prepare','synthetic','bridge','report'],required=True)
    p.add_argument('--gpu',default='0');a=p.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id)
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    from videoqa_runtime.common import ROOT,offline_environment
    offline_environment();run=ROOT/'outputs/analysis'/a.run_id
    from videoqa_lmms_bridge.bridge import prepare,synthetic,run_bridge
    if a.phase=='prepare':prepare(run)
    elif a.phase=='synthetic':synthetic(run,a.gpu)
    elif a.phase=='bridge':run_bridge(run,a.gpu)
    else:
        from videoqa_lmms_bridge.report import summarize
        summarize(run)
