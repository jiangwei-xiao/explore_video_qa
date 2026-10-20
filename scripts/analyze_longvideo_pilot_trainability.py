"""长视频先导CPU诊断入口：禁用GPU和联网，固定五折训练或只读完成恢复。"""
import argparse
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                  OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    """接口：只接收运行ID；不暴露训练参数、GPU、数据集或教师调用开关。"""
    # 步骤1验证目录名；步骤2执行既有CPU训练器上的固定诊断。
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum():
        parser.error('Invalid analysis run ID')
    from videoqa_audit.longvideo_pilot_trainability import run
    result = run(ROOT / 'outputs/analysis' / args.run_id)
    print(dict(status=result['status'], decision=result['recommendation']['decision'],
               correct_197=result['primary']['correct'], correct_194=result['sensitivity']['correct'],
               total_optimizer_updates=result['total_optimizer_updates'],
               new_generations=result['new_generations']))


if __name__ == '__main__':
    main()
