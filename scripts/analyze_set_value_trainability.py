"""集合价值学习诊断独立入口：只用CPU小头和历史资产，完整运行或恢复审计。"""
import argparse
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    """步骤1核对运行ID；步骤2仅调用独立CPU诊断，不启动原训练/问答入口。"""
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if not args.run_id.replace('_','').isalnum():parser.error('Invalid analysis ID')
    from videoqa_audit.set_value_trainability import run
    result=run(ROOT/'outputs/analysis'/args.run_id)
    print({k:result[k] for k in ('status','decision','total_optimizer_updates','trajectories','new_backbone_forwards','new_generations')})


if __name__=='__main__':main()
