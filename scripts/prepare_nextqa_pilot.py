"""NExT-QA小试的数据资产入口；只下载固定身份候选，不运行模型。"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    """接口：首次预取320训练/120验证视频，后续资格替补沿用同一散列顺序。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--train-videos',type=int,default=320);p.add_argument('--val-videos',type=int,default=120)
    args=p.parse_args()
    if not args.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    from videoqa_set_value.data_acquisition import acquire
    acquire(ROOT/'outputs/set_value'/args.run_id,args.train_videos,args.val_videos)


if __name__=='__main__':main()
