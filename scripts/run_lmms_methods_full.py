#!/usr/bin/env python3
"""后四卡Top-K/RD全量参考协议入口；不接受算法参数或其他GPU。"""
import argparse
import os
import re
import sys
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from videoqa_runtime.common import ROOT
from videoqa_lmms_baseline.methods import prepare, execute

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run-id', required=True)
    p.add_argument('--prepare-only', action='store_true'); p.add_argument('--resume', action='store_true'); a=p.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', a.run_id)
    run=ROOT / 'outputs/full_videoqa' / a.run_id
    if a.prepare_only: prepare(run)
    else:
        assert (run / 'protocol.json').exists()
        if (run / 'calls').exists() and not a.resume: raise ValueError('Existing run requires --resume')
        execute(run)
