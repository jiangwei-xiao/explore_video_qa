"""首轮改进统计入口，只读实验结果并生成独立分析产物。"""
import sys
import os
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
if __name__=='__main__':
    from videoqa_methods.followups_report import summarize
    summarize(Path(sys.argv[1]))
