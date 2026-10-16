"""帧组事实观察能力的独立入口；原选择算法和正式QA入口不修改。"""
import argparse
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    """接口：先prepare与事前标注freeze，再run；后台worker必须绑定唯一物理卡。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=['prepare','freeze','run','worker','audit','report'],required=True)
    parser.add_argument('--gpu')
    args=parser.parse_args()
    if not args.run_id.replace('_','').isalnum():parser.error('Invalid diagnostic run ID')
    from videoqa_audit.evidence_observation_probe import prepare,freeze_reference,run,worker,audit,technical_report
    root=ROOT/'outputs/diagnostics'/args.run_id
    # 步骤1：CPU准备、评价冻结和GPU生成是显式阶段，不因缺少标注自动调用模型。
    if args.stage=='prepare':prepare(root)
    elif args.stage=='freeze':freeze_reference(root)
    elif args.stage=='run':run(root)
    elif args.stage=='audit':audit(root)
    elif args.stage=='report':technical_report(root)
    else:
        if args.gpu not in map(str,range(8)) or os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu:
            parser.error('Observer worker must see exactly its assigned physical GPU')
        worker(root,args.gpu)


if __name__=='__main__':main()
