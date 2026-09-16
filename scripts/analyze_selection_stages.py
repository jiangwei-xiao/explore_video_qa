"""原选择过程的离线阶段定位入口，不运行任何模型或参数优化。"""
import argparse
import os
import sys
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_audit.stage_attribution import analyze

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--audit',required=True)
    a=p.parse_args();summary=analyze(a.source,a.audit);print({k:v for k,v in summary.items() if k!='details'})
