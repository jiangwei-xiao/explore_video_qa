#!/usr/bin/env python3
"""冻结200题Qwen输入的阶段入口；正式运行必须nohup＋setsid后台执行。"""
import argparse
import os
from pathlib import Path
import re
import sys
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT
from videoqa_budget32 import development

def main():
    """接口：显式运行编号和恢复标记，旧调用存在时不得误开启新一轮生成。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=['prepare','qa','report','audit'],required=True)
    parser.add_argument('--input-dir')
    parser.add_argument('--gpus',nargs='+',default=['0','1','2','3'])
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id)
    run=ROOT/'outputs/diagnostics'/args.run_id
    # 步骤1：账本与运行目录绑定；恢复只检查/复用已有生成，绝不增加调用额度。
    if (run/'calls').exists() and not args.resume:raise ValueError('Existing calls require --resume')
    if args.stage=='prepare':
        assert args.input_dir;development.prepare(run,args.input_dir,args.gpus)
    elif args.stage=='qa':
        development.execute(run);development.audit(run)
    elif args.stage=='report':development.summarize(run)
    else:development.audit(run)

if __name__=='__main__':main()
