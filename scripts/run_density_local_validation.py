#!/usr/bin/env python3
"""局部选帧验证入口：CPU重放与最多一张空闲GPU提取扩窗视觉特征，不生成答案。"""
import argparse
import os
from pathlib import Path
import sys
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_audit.density_local_validation import execute
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--gpu',default='0');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();print(execute(args.run_id,args.gpu,args.resume),flush=True)
