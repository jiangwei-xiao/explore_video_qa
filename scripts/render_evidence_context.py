"""精确原片窗口核查入口：输入JSON规格，不生成模型答案。"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_audit.core import read
from videoqa_audit.context import render_windows

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True);parser.add_argument('--spec',required=True)
    args=parser.parse_args();render_windows(args.out,read(args.spec))
