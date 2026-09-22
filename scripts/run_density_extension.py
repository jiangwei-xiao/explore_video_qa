#!/usr/bin/env python3
"""开发200题扩展入口；准备阶段不占GPU，正式阶段可后台恢复且不重复生成。"""
import argparse
import os
from pathlib import Path
import sys
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_audit.density_extension import execute
from videoqa_audit.density_extension_report import source_inventory,summarize

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--gpus',default='0,1,2,3,4,5,6,7');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();run=execute(args.run_id,args.gpus.split(','),args.resume,args.prepare_only)
    if args.prepare_only:source_inventory(run);print('Prepared200 questions; 500 reused results; 300 new answers pending')
    else:print(summarize(run)['accuracy']['all200'])
