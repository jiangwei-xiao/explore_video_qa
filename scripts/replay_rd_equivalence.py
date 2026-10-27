"""RD共同选择链的CPU配对重放；无GPU/解码/评分/问答执行阶段。"""
import argparse
import os
from pathlib import Path
import re
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    """接口：只开放CPU并发、运行编号、准备/重放/验收及显式恢复，不开放方法参数搜索。"""
    from videoqa_audit.rd_equivalence import prepare,execute,audit_completed
    from videoqa_full.state import durable,utc
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--stage',choices=['prepare','run','audit'],default='run')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id) or not 1<=args.workers<=8:
        parser.error('Invalid run ID or worker count')
    root=ROOT/'outputs/analysis'/args.run_id
    if (root/'protocol.json').exists() and not args.resume:raise ValueError('Existing frozen run requires --resume')
    # 步骤1准备绑定旧来源与当前实现；步骤2重放或完成验证均不实例化模型。
    try:
        if args.stage=='prepare':result=prepare(root)
        elif args.stage=='audit':result=audit_completed(root)
        else:result=execute(root,args.workers)
        print(result.get('status','prepared'),flush=True)
    except BaseException as error:
        root.mkdir(parents=True,exist_ok=True)
        durable(root/'entry_failure.json',dict(error=repr(error),stage=args.stage,utc=utc()));raise


if __name__=='__main__':main()
