"""exp33阶段入口：先完成9748项零模型离线检查，再依据证据门槛接续小测。"""
import argparse
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_audit.adjacent_context_experiment import offline,render_review,verify_offline


def main():
    """接口：运行编号绑定独立目录；阶段恢复只消费已冻结资产。"""
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',required=True)
    p.add_argument('--stage',choices=('offline','review','verify'),required=True)
    args=p.parse_args()
    if not args.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    root=ROOT/'outputs/methods'/args.run_id
    # 步骤1离线冻结，步骤2按固定面板渲染，步骤3完整恢复只验指纹。
    if args.stage=='offline':offline(root)
    elif args.stage=='review':render_review(root)
    else:verify_offline(root)


if __name__=='__main__':main()
