#!/usr/bin/env python3
"""9题定位文本探针入口；长任务以nohup＋setsid后台运行，最终问答始终为0。"""
import argparse
import os
from pathlib import Path
import re
import sys
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_audit import query_input_probe as probe


def main():
    """接口：准备/评分/汇总分离；内部worker只能执行已冻结的题目与GPU。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=['prepare','run','worker','report'],required=True)
    p.add_argument('--gpus',nargs='+',default=['0','1']);p.add_argument('--gpu');p.add_argument('--questions',nargs='+');p.add_argument('--resume',action='store_true')
    a=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id)
    run=ROOT/'outputs/analysis'/a.run_id
    # 步骤1：外部恢复需显式标记；内部worker按协调器的固定分配执行。
    if a.stage=='run' and (run/'calls').exists() and not a.resume:raise ValueError('Existing scoring calls require --resume')
    if a.stage=='prepare':probe.prepare(run,a.gpus)
    elif a.stage=='run':probe.execute(run)
    elif a.stage=='report':probe.summarize(run)
    else:
        from videoqa_runtime.common import read_json
        spec=read_json(run/'protocol.json');assignment=read_json(run/'assignment.json')
        assert a.gpu in spec['allowed_gpus'] and a.questions==assignment[a.gpu]
        probe.worker(run,a.gpu,a.questions)


if __name__=='__main__':main()
