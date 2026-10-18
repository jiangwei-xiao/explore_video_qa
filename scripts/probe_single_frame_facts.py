"""单帧事实接口验证入口；48辅助调用与正式视频问答预算隔离。"""
import argparse
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    """接口：准备、评价冻结、后台生成和审计分阶段；物理GPU由协调器固定。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',required=True,choices=['prepare','freeze','run','worker','audit'])
    parser.add_argument('--gpu')
    args=parser.parse_args()
    if not args.run_id.replace('_','').isalnum():parser.error('Invalid diagnostic run ID')
    from videoqa_audit.single_frame_fact_probe import prepare,freeze_reference,run,worker,audit
    root=ROOT/'outputs/diagnostics'/args.run_id
    # 步骤1：只有run/worker才可能加载模型；准备与冻结明确分离。
    if args.stage=='prepare':prepare(root)
    elif args.stage=='freeze':freeze_reference(root)
    elif args.stage=='run':run(root)
    elif args.stage=='audit':audit(root)
    else:
        if args.gpu not in map(str,range(8)) or os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu:
            parser.error('Worker must see exactly the assigned physical GPU')
        worker(root,args.gpu)


if __name__=='__main__':main()
