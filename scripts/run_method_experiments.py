import argparse
import os
import sys
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from videoqa_methods.run import execute

if __name__ == '__main__':
    # 步骤1：校验命令参数，以独立运行编号冻结本轮实验。
    p = argparse.ArgumentParser(description='方法首版四组实验：Video-MME冻结开发50题')
    p.add_argument('--run-id', required=True)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    gpus = args.gpus.split(',')
    if len(set(gpus)) != len(gpus) or not all(x.isdigit() for x in gpus): p.error('Distinct numeric GPU IDs required')
    # 步骤2：执行/恢复后台调度；旧基线仅作为已冻结对照，不重新答题。
    run = execute(args.run_id, gpus, args.resume)
    from videoqa_methods.report import summarize
    # 步骤3：模型任务完成后自动统计并渲染案例，不追加问答调用。
    summarize(run)
    from videoqa_methods.visualize import render_cases
    render_cases(run)
