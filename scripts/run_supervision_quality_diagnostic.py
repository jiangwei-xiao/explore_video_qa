"""训练监督画面敏感性：CPU准备、原卡后台问答、只读完成审计。"""
import argparse
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    """接口：独立运行编号隔离历史；worker必须只暴露其历史物理GPU。"""
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=['prepare','run','worker','audit'],required=True)
    parser.add_argument('--gpu');args=parser.parse_args()
    if not args.run_id.replace('_','').isalnum():parser.error('Invalid diagnostic ID')
    from videoqa_audit import supervision_quality as d
    root=ROOT/'outputs/diagnostics'/args.run_id
    if args.stage=='worker':
        if args.gpu not in map(str,range(8)) or os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu:
            parser.error('Expose assigned physical GPU only')
        d.worker(root,args.gpu)
    elif args.stage=='prepare':print(d.prepare(root))
    elif args.stage=='audit':print(d.audit(root))
    else:
        result=d.run(root);print({k:result[k] for k in ('status','questions') if k in result})


if __name__=='__main__':main()
