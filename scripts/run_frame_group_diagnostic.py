"""五组三条件邻接诊断入口：准备无GPU，正式阶段独立后台执行。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from videoqa_audit.frame_group_diagnostic import cases, prepare_case, coordinate, worker, summarize, PYTHONS
from videoqa_full.state import durable, sha256, utc


def main():
    """接口：各组CPU/推理使用历史正式环境；所有输入发布后才可运行真实生成。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--stage', choices=('all', 'prepare', 'prepare-case', 'qa', 'worker', 'audit'), default='all')
    parser.add_argument('--case', choices=tuple(c['id'] for c in cases()))
    args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum(): parser.error('Invalid run ID')
    root = ROOT/'outputs/diagnostics'/args.run_id; root.mkdir(parents=True, exist_ok=True)
    # 步骤1：每组独立任务实例；MLVU任务重绑定不泄漏到Video-MME。
    if args.stage in ('all', 'prepare'):
        for case in cases():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
            subprocess.run([PYTHONS[case['profile']], str(Path(__file__).resolve()), '--run-id', args.run_id,
                '--stage', 'prepare-case', '--case', case['id']], cwd=ROOT, env=env, check=True)
        durable(root/'prepared.json', dict(protocols={c['id']:sha256(root/c['id']/'protocol.json') for c in cases()},
            max_real_calls=15, max_synthetic_calls=5, utc=utc()))
        if args.stage == 'prepare': return
    if args.stage in ('all', 'qa'): coordinate(root)
    elif args.stage == 'audit': summarize(root)
    else:
        if args.case is None: parser.error('Case is required')
        case = next(c for c in cases() if c['id'] == args.case)
        if args.stage == 'prepare-case': prepare_case(root/case['id'], case)
        else:
            if os.environ.get('CUDA_VISIBLE_DEVICES') != case['gpu']: parser.error('Worker must use fixed historical physical GPU')
            worker(root/case['id'])


if __name__ == '__main__': main()
