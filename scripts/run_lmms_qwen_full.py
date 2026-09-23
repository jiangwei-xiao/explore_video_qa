#!/usr/bin/env python3
"""Qwen三策略全量入口：合成、准备及脱离会话运行，GPU范围固定4–7。"""
import argparse
import os
import re
import sys
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT
from videoqa_lmms_qwen.run import synthetic,prepare,execute

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id');p.add_argument('--synthetic',action='store_true')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not a.run_id:
        assert a.synthetic,'Set frozen run ID for preparation/execution/resume'
        a.run_id='uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__'+datetime.now(timezone.utc).strftime('%Y%m%d')+'__r01'
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id)
    run=ROOT/'outputs/full_videoqa'/a.run_id;print('Run:',run,flush=True)
    if a.synthetic:synthetic(run)
    elif a.prepare_only:prepare(run)
    else:
        assert (run/'protocol.json').exists()
        if (run/'calls').exists() and not a.resume:raise ValueError('Existing run requires --resume')
        execute(run)
