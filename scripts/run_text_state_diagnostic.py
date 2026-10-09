"""固定022-1计时状态两输入诊断，真实生成上限2，合成上限1。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from videoqa_audit.text_state_diagnostic import prepare, execute, worker, audit, PYTHON


def main():
    """接口：CPU冻结后后台等卡；不加入未通过画面门槛的exp31诊断。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--stage', choices=('all', 'prepare', 'qa', 'worker', 'audit'), default='all')
    parser.add_argument('--gpu', choices=tuple(map(str, range(8))))
    args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum(): parser.error('Invalid run ID')
    root = ROOT/'outputs/diagnostics'/args.run_id
    # 步骤1：按固定正式环境执行，准备阶段明确无GPU可见，不产生模型前向。
    if args.stage == 'all':
        for stage in ('prepare', 'qa'):
            env = dict(os.environ)
            if stage == 'prepare': env['CUDA_VISIBLE_DEVICES'] = ''
            subprocess.run([PYTHON, str(Path(__file__).resolve()), '--run-id', args.run_id, '--stage', stage], cwd=ROOT, env=env, check=True)
    elif args.stage == 'prepare': prepare(root)
    elif args.stage == 'qa': execute(root)
    elif args.stage == 'audit': audit(root)
    else:
        if args.gpu is None: parser.error('Worker requires one physical GPU')
        worker(root, args.gpu)


if __name__ == '__main__':
    main()
