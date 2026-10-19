"""独立数据CPU准备入口：参考源指纹、候选索引与800题冻结。"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    """接口：reference可与网络下载并行；manifest须在采集进程结束后执行。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=['reference','manifest'],required=True)
    args=p.parse_args()
    if not args.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    from videoqa_set_value.data import build_reference_bank,prepare_manifest
    root=ROOT/'outputs/set_value'/args.run_id
    if args.stage=='reference':build_reference_bank(root)
    else:prepare_manifest(root)


if __name__=='__main__':main()
