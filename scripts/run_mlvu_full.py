"""MLVU独立全量入口：显式跨环境、持久状态、后台自动推进与安全恢复。"""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
from pathlib import Path
import argparse
import re
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json
from videoqa_full.state import durable, exclusive, utc
from videoqa_mlvu.data import PYTHONS


def main():
    """接口：零问答数据准备可先运行；模型阶段必须有冻结协议与完整输入额度。"""
    p = argparse.ArgumentParser(); p.add_argument('--run-id', required=True)
    p.add_argument('--gpus', nargs='+', default=list(map(str, range(8))))
    p.add_argument('--stage', choices=['data', 'controls', 'preflight', 'coarse', 'refine',
        'prepare-llava8', 'llava8', 'prepare-qwen32', 'qwen32', 'report', 'audit', 'all'], default='all')
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', a.run_id) or not a.gpus or len(set(a.gpus)) != len(a.gpus) or not set(a.gpus) <= set(map(str, range(8))):
        raise ValueError('Invalid run/GPU scope')
    root = ROOT / 'outputs/full_videoqa' / a.run_id; root.mkdir(parents=True, exist_ok=True)
    if list(root.glob('qa/*/calls/*/*.json')) and not a.resume: raise ValueError('Existing generation needs explicit resume')
    if a.stage == 'data':
        from videoqa_mlvu.data import prepare_manifest, prepare_indices
        with exclusive(root / 'data.lock'):
            durable(root / 'data_launcher.json', dict(pid=os.getpid(), sid=os.getsid(0), utc=utc()))
            prepare_manifest(root, root / 'sources/mlvu_official.parquet'); prepare_indices(root)
    elif a.stage == 'controls':
        from videoqa_scorer_comparison.preparation import controls
        controls(root)
    elif a.stage == 'all':
        with exclusive(root / 'pipeline.lock'):
            durable(root / 'launcher.json', dict(pid=os.getpid(), sid=os.getsid(0), utc=utc()))
            stages = ['data', 'controls', 'preflight', 'coarse', 'refine', 'prepare-llava8',
                      'llava8', 'prepare-qwen32', 'qwen32', 'report', 'audit']
            for stage in stages:
                if stage == 'controls' and (root / 'controls_summary.json').exists():
                    # 已独立完成的10800项控制按原文件摘要核验，不重复计算或改写完成记录。
                    from videoqa_runtime.common import sha256
                    control = read_json(root / 'controls_summary.json')
                    if control['status'] != 'passed' or control['selection_replays'] != 10800: raise ValueError('Control gate differs')
                    for relative, expected in control['files'].items():
                        if sha256(root / relative) != expected: raise ValueError('Control output fingerprint differs')
                    continue
                if stage == 'data' and (root / 'data_launcher.json').exists() and not (root / 'indices_complete.json').exists():
                    # 步骤1：先行CPU准备是具体已知进程；观察超时不重复启动索引。
                    pid = read_json(root / 'data_launcher.json')['pid']
                    while not (root / 'indices_complete.json').exists():
                        os.kill(pid, 0)
                        durable(root / 'pipeline_status.json', dict(status='waiting_data_prepare', stage=stage,
                            data_pid=pid, pid=os.getpid(), sid=os.getsid(0), utc=utc()))
                        time.sleep(20)
                profile = stage.removeprefix('prepare-')
                python = PYTHONS.get(profile, PYTHONS['prepare'])
                if stage in ('data', 'controls', 'preflight', 'report', 'audit'): python = PYTHONS['qwen32']
                durable(root / 'pipeline_status.json', dict(status='running', stage=stage,
                    pid=os.getpid(), sid=os.getsid(0), utc=utc()))
                try:
                    subprocess.run([python, str(Path(__file__).resolve()), '--run-id', a.run_id,
                        '--stage', stage, '--gpus', *a.gpus, '--resume'], cwd=ROOT, check=True)
                except BaseException as exc:
                    durable(root / 'pipeline_status.json', dict(status='paused', stage=stage, error=repr(exc), utc=utc())); raise
            durable(root / 'pipeline_status.json', dict(status='completed', utc=utc()))
    else:
        from videoqa_mlvu import preparation, runtime
        if a.stage == 'preflight': preparation.preflight(root)
        elif a.stage.startswith('prepare-'): runtime.prepare_inputs(root, a.stage.removeprefix('prepare-'))
        elif a.stage in ('coarse', 'refine', 'llava8', 'qwen32'):
            from videoqa_mlvu.coordinator import run_stage
            run_stage(root, a.stage, a.gpus)
        elif a.stage == 'report':
            from videoqa_mlvu.report import summarize
            summarize(root)
        else:
            from videoqa_mlvu.report import final_audit
            final_audit(root)


if __name__ == '__main__': main()
