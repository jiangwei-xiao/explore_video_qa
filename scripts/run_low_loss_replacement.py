"""exp34独立执行入口：离线、完整输入冻结、后台问答与安全恢复。"""
import argparse
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    """接口：每个子进程以同一运行编号读取冻结配置，问答阶段不再执行选帧。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--stage', choices=(
        'all', 'offline', 'prepare', 'prepare-unit', 'qa', 'run-unit',
        'worker', 'audit', 'report'), default='all')
    parser.add_argument('--dataset', choices=('videomme', 'mlvu'))
    parser.add_argument('--profile', choices=('llava8', 'qwen32'))
    parser.add_argument('--gpu')
    parser.add_argument('--gpus')
    args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum():
        parser.error('Invalid run ID')
    os.environ['RD_EXP34_RUN_ID'] = args.run_id
    # 步骤1：运行编号必须在加载接口之前绑定，后台子进程继承同一资产目录。
    from videoqa_audit.low_loss_experiment import prepare_offline, verify_offline
    from videoqa_audit.low_loss_development import (
        preflight, prepare_all, prepare_unit, execute_all, run_unit,
        worker, audit_all, report,
    )
    root = ROOT / 'outputs/methods' / args.run_id
    if args.stage in ('all', 'offline'):
        prepare_offline(root)
    if args.stage == 'offline':
        return
    if args.stage in ('all', 'prepare'):
        verify_offline(root)
        preflight(root)
        prepare_all(root)
    if args.stage in ('all', 'qa'):
        execute_all(root)
    elif args.stage == 'prepare':
        return
    elif args.stage == 'audit':
        audit_all(root)
    elif args.stage == 'report':
        report(root)
    else:
        if not args.dataset or not args.profile:
            parser.error('Cell requires dataset and profile')
        if args.stage == 'prepare-unit':
            prepare_unit(root, args.dataset, args.profile)
        elif args.stage == 'run-unit':
            gpus = (args.gpus or '').split(',')
            if not 1 <= len(gpus) <= 2 or len(set(gpus)) != len(gpus) or any(
                    g not in map(str, range(8)) for g in gpus):
                parser.error('One or two unique physical GPUs required')
            run_unit(root, args.dataset, args.profile, gpus)
        else:
            if args.gpu not in map(str, range(8)) or os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu:
                parser.error('Worker must see its assigned physical GPU')
            worker(root, args.dataset, args.profile, args.gpu)


if __name__ == '__main__':
    main()
