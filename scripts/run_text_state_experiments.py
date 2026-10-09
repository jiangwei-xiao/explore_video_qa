"""exp29固定OCR文字状态重复度入口；试点后再决定开发，不搜索参数。"""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_audit.text_state_experiments import prepare,synthetic,extract,select_population
from videoqa_full.state import durable,utc,exclusive


def main():
    """接口：固定阶段、独立目录与恢复，异常不降分辨率或更换OCR阈值。"""
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    p.add_argument('--stage',required=True,choices=('prepare','pilot','development-assets','select'))
    p.add_argument('--population',choices=('pilot','development'),default='pilot');p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if not a.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    root=ROOT/'outputs/methods'/a.run_id
    try:
        with exclusive(root/'pipeline.lock'):
            if a.stage=='prepare':prepare(root)
            elif a.stage=='pilot':
                prepare(root);synthetic(root);extract(root,'pilot',a.workers);select_population(root,'pilot')
            elif a.stage=='development-assets':
                if not (root/'pilot_review_gate.json').exists() or __import__('json').loads((root/'pilot_review_gate.json').read_text())['status']!='passed':raise ValueError('Actual competition review not passed')
                extract(root,'development',a.workers);select_population(root,'development')
            else:select_population(root,a.population)
    except BaseException as e:durable(root/('failure_'+a.stage+'.json'),dict(error=repr(e),utc=utc()));raise


if __name__=='__main__':main()
