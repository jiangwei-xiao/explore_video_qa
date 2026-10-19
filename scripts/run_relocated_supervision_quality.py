"""已授权GPU0—3接续：原输入不变，旧55题引用、未开始65题一次生成。"""
import argparse
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    """接口：换卡接续使用独立运行目录，不覆盖第一批协议或结果。"""
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=['prepare','run','worker','audit'],required=True);parser.add_argument('--gpu')
    args=parser.parse_args()
    if not args.run_id.replace('_','').isalnum():parser.error('Invalid run ID')
    from videoqa_audit import supervision_quality_relocation as r
    root=ROOT/'outputs/diagnostics'/args.run_id
    if args.stage=='worker':
        if args.gpu not in r.GPUS or os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu:parser.error('Expose assigned GPU0—3 only')
        r.d.worker(root/'relocated',args.gpu)
    elif args.stage=='prepare':print(r.prepare(root))
    elif args.stage=='audit':print(r.audit(root))
    else:
        result=r.run(root);print({k:result[k] for k in ('status','questions') if k in result})


if __name__=='__main__':main()
