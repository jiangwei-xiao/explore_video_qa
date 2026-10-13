"""BLIP前端对照固定复核页入口，只读已完成结果，不生成新问答。"""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse
from pathlib import Path
import sys
import re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_audit.scorer_comparison_review import render


def main():
    """接口：结果完整后按固定规则生成页面，不能在中途分数上选择优化案例。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id):raise ValueError('Invalid run ID')
    value=render(ROOT/'outputs/full_videoqa'/a.run_id)
    print(dict(question_ids=value['question_ids'],boards=len(value['boards']),status=value['status']),flush=True)


if __name__=='__main__':main()
