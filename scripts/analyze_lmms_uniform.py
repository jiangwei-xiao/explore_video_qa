#!/usr/bin/env python3
"""只做900视频源帧离线对照，不加载任何模型。"""
import argparse
import os
import re
import sys
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT
from videoqa_lmms_bridge.frames import offline

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id) and 1<=a.workers<=8
    offline(ROOT/'outputs/analysis'/a.run_id,a.workers)
