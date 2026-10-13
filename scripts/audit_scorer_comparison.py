"""评分器对照独立只读审计入口，不修改冻结运行协议或生成任何新答案。"""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse
from pathlib import Path
import sys
import re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_audit.scorer_comparison_audit import audit_coarse,audit_selection


def main():
    """接口：粗池已完成项可先核验；完整模式不允许把部分产物标为全量完成。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--stage',choices=['interim','coarse','qwen32','llava8'],default='interim');a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id):raise ValueError('Invalid run ID')
    root=ROOT/'outputs/full_videoqa'/a.run_id
    if a.stage in ('interim','coarse'):result=audit_coarse(root,a.stage=='coarse')
    else:result=audit_selection(root,a.stage)
    print({k:v for k,v in result.items() if k not in ('sources','source')},flush=True)


if __name__=='__main__':main()
