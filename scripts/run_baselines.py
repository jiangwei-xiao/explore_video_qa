import argparse
import os
import sys
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from videoqa_runtime.baseline_run import execute, DATASETS
from videoqa_runtime.baseline_report import summarize
from videoqa_runtime.common import ROOT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frozen Video-MME baseline evaluation (uniform/topk)')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dataset', default='development50', choices=sorted(DATASETS),
                        help='development50=冻结开发50题（默认）；full2700=全量公开测试集2700题')
    args = parser.parse_args()
    gpus = args.gpus.split(',')
    if not all(g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus):
        parser.error('GPU IDs must be distinct numeric indices')
    execute(args.run_id, gpus, args.resume, args.dataset)
    report = summarize(ROOT / 'outputs/baselines' / args.run_id)
    print('Final accuracy:', {m: report['methods'][m]['by_stratum']['all']['accuracy'] for m in report['methods']}, flush=True)
