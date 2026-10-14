"""MLVU新任务的合成GPU入口验收；独立目录、不产生真实题答案。"""
from pathlib import Path
import argparse
import os
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))


def main():
    """接口：只使用指定空闲卡，记录实际原生框架生成、任务模板和可恢复原文。"""
    from videoqa_runtime.baseline_run import available_gpus
    from videoqa_full.state import durable, utc
    p = argparse.ArgumentParser(); p.add_argument('--profile', choices=['llava8','qwen32'], required=True)
    p.add_argument('--gpu', required=True); a = p.parse_args()
    while True:
        free, _ = available_gpus([a.gpu])
        if a.gpu in free: break
        time.sleep(15)
    # 步骤1：CUDA可见范围先于加载；验收目录与正式QA账本完全隔离。
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    run = ROOT / 'outputs/analysis/mlvu_backend_synthetic_20261014_r1' / a.profile
    if (run / 'acceptance.json').exists(): raise ValueError('Synthetic stage already completed; do not repeat')
    durable(run / 'protocol.json', dict(profile=a.profile, scope='synthetic_only', real_QA=0))
    durable(run / 'launcher.json', dict(pid=os.getpid(), sid=os.getsid(0), gpu=a.gpu, utc=utc()))
    from videoqa_mlvu.runtime import qa_initialize
    backend, adapter, observer = qa_initialize(run, a.gpu)
    # 步骤2：qa_initialize已核验真实Tensor、原生generate及原始返回；不根据答对多少决定通过。
    durable(run / 'acceptance.json', dict(status='passed', gpu=a.gpu, real_QA=0,
        synthetic_generations=4 if a.profile=='qwen32' else 3, model_load_report=backend.load_report, utc=utc()))
    print(dict(profile=a.profile, status='passed', real_QA=0), flush=True)


if __name__ == '__main__': main()
