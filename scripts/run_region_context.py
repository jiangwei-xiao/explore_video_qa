"""R实验入口：离线验证与正式问答分开，正式阶段必须已有通过的证据门槛。"""
import argparse
import os
from pathlib import Path
import sys
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True);parser.add_argument('--phase',choices=['offline','formal'],required=True)
    parser.add_argument('--gpus',default='0,1,2,3,4,5,6,7');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();gpus=args.gpus.split(',')
    if len(set(gpus))!=len(gpus) or not all(g.isdigit() for g in gpus):parser.error('Distinct numeric GPU IDs required')
    from videoqa_methods.region_run import execute
    execute(args.run_id,args.phase,gpus,args.resume)
