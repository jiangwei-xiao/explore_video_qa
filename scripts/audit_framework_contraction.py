"""单预算接口与内部结构审计入口：只读冻结资产，CPU执行，新增模型调用为0。"""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
from pathlib import Path
import sys
import argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_audit.framework_contraction import prepare, execute
from videoqa_full.state import durable, utc


def main():
    """接口：运行编号与CPU并发控制；预算固定覆盖已冻结K8/K16/K32，不提供M选项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--stage', choices=('prepare', 'run'), default='run')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not args.run_id.replace('_', '').replace('-', '').isalnum() or not 1 <= args.workers <= 16:
        parser.error('Invalid run-id or CPU worker count')
    root = ROOT / 'outputs/analysis' / args.run_id
    # 步骤1：先绑定协议，再执行；恢复采用同一运行编号与输入指纹。
    try:
        result = prepare(root) if args.stage == 'prepare' else execute(root, args.workers)
        print(result.get('status', 'prepared'), flush=True)
    except BaseException as error:
        # 步骤2：异常原子留痕，不改采样、选择公式或历史结果来追求完成。
        durable(root / 'failure.json', dict(error=repr(error), utc=utc()))
        raise


if __name__ == '__main__':
    main()
