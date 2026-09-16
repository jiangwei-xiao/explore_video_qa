"""首轮改进批量入口：独立运行编号、后台执行及安全恢复。"""
import argparse
import os
from pathlib import Path
import sys
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

if __name__=='__main__':
    # 步骤1：只暴露已冻结的阶段与GPU参数，不提供未授权的算法调参入口。
    parser=argparse.ArgumentParser(description='C-local 50题 + 原词受限查询离线验证')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    parser.add_argument('--phase',choices=['clocal','query','all'],default='all')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--replay-only',action='store_true')
    args=parser.parse_args()
    gpus=args.gpus.split(',')
    if len(gpus)!=len(set(gpus)) or not all(g.isdigit() for g in gpus): parser.error('Distinct numeric GPU IDs required')
    # 步骤2：全部生成调用由持久化账本保护；不自动重试、提交或推送。
    from videoqa_methods.followups import execute
    execute(args.run_id,gpus,args.phase,args.resume,args.replay_only)
