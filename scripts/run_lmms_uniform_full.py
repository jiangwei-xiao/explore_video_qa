#!/usr/bin/env python3
"""单一LLaVA均匀全量入口；不提供其他方法/模型或参数搜索分支。"""
import os
import sys
import argparse
import re
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT
from videoqa_lmms_baseline.run import prepare,execute

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id)
    run=ROOT/'outputs/baselines'/a.run_id
    if a.prepare_only:prepare(run)
    else:
        assert (run/'protocol.json').exists(),'Prepare and freeze before generating'
        if (run/'calls').exists() and not a.resume:raise ValueError('Existing run requires --resume')
        execute(run,list(dict.fromkeys(a.gpus.split(',')))[:8])
