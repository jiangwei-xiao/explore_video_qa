#!/usr/bin/env python3
"""固定两题帧库的原选项检索对照：仅移除问题前缀，零问答、不选择正确分支。"""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256, offline_environment
from videoqa_full.state import durable, exclusive, utc
from videoqa_full.prepare import environment
from run_framegroup_blip_probe import score_once, MODEL, PYTHON


def prepare(run, parent, gpu):
    """接口：冻结原选项、既有像素与对照输出；只改变是否带原问题前缀。"""
    assets = read_json(parent / 'assets_manifest_r3.json')
    previous = read_json(parent / 'blip_protocol.json')
    # 步骤1：先核对全部原始来源和已完成对照，不允许输出后改动帧组。
    for p, h in {**previous['sources'], **previous['model_files'],
                 **read_json(parent / 'blip_summary.json')['artifacts']}.items():
        assert sha256(p) == h, p
    jobs = []
    for q, case in assets['cases'].items():
        assert len(case['options']) == 4
        for letter, option in zip('ABCD', case['options']):
            ids = sorted(case['frames'], key=int)
            for start in range(0, len(ids), 16):
                jobs.append(dict(id=f'{q}__option_{letter}__{start}', question_id=q,
                                 view='option_' + letter, text=option, pts=ids[start:start+16]))
    assert len(jobs) == 12 and sum(len(j['pts']) for j in jobs) == 128
    sources = {str(p): sha256(p) for p in [parent / 'assets_manifest_r3.json',
        parent / 'blip_summary.json', Path(__file__), ROOT / 'scripts/run_framegroup_blip_probe.py']}
    for case in assets['cases'].values():
        for row in case['frames'].values():
            assert sha256(row['file']) == row['png_sha256']
            sources[row['file']] = row['png_sha256']
    spec = dict(parent=str(parent), sources=sources, model_files=previous['model_files'],
        environment=environment(PYTHON), gpu=gpu, jobs=jobs, real_frame_text_pairs=128,
        real_batches=12, synthetic_batches=1, final_qa_calls=0,
        contrast='Original option including letter, versus frozen question plus same option; all branches symmetric')
    assert spec['environment'] == previous['environment']
    path = run / 'blip_protocol.json'
    if path.exists():
        assert read_json(path) == spec, 'Frozen input changed'
    else:
        durable(path, spec)
    return assets, spec


def execute(run, parent, gpu, stage):
    """接口：调用前检查点、失败不重试；完成恢复不加载模型、不新增前向。"""
    import subprocess
    run.mkdir(parents=True, exist_ok=True)
    with exclusive(run / 'run.lock'):
        assets, spec = prepare(run, parent, gpu)
        if stage == 'prepare':
            print('Frozen:128 frame-text pairs,12 batches,QA0', flush=True)
            return
        if (run / 'summary.json').exists():
            for p, h in read_json(run / 'summary.json')['artifacts'].items():
                assert sha256(p) == h
            print('Completed recovery:0 extra forwards', flush=True)
            return
        for p in (run / 'blip_calls').glob('*.json'):
            assert read_json(p)['status'] == 'completed', 'Uncertain forward; stop'
        memory, utilization = map(int, subprocess.check_output(['nvidia-smi', f'--id={gpu}',
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(','))
        if memory >= 256 or utilization:
            raise RuntimeError('Requested GPU is occupied')
        os.environ.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        offline_environment()
        import torch
        from PIL import Image
        from videoqa_runtime.baseline_selection import BlipItmScorer
        from videoqa_budget32.selection import configure_visual_precision
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(2027)
        precision = configure_visual_precision(torch)
        started = time.perf_counter()
        scorer = BlipItmScorer(str(MODEL))
        assert scorer.model.dtype == torch.float32 and not scorer.model.training
        durable(run / 'load.json', dict(pid=os.getpid(), sid=os.getsid(0), gpu=gpu,
            precision=precision, seconds=time.perf_counter()-started, utc=utc()))
        # 步骤2：保持官方384、FP32、无截断；新旧条件图像及顺序相同。
        images = [Image.new('RGB', (1280, 720), 'gray') for _ in range(16)]
        score_once(run, scorer, 'Which synthetic color is visible?', images, list(range(16)), 'synthetic', gpu)
        for im in images:
            im.close()
        scores = {}
        for job in spec['jobs']:
            images = []
            for pts in job['pts']:
                with Image.open(assets['cases'][job['question_id']]['frames'][pts]['file']) as im:
                    images.append(im.convert('RGB'))
            result = score_once(run, scorer, job['text'], images,
                [job['question_id']+'__'+p for p in job['pts']], job['id'], gpu)
            scores.setdefault(job['question_id'], {}).setdefault(job['view'], {}).update(zip(job['pts'], result['scores']))
            for im in images:
                im.close()
        # 步骤3：输出所有选项及所有帧，不用正确答案筛选分支、不计算答题成绩。
        calls = list((run / 'blip_calls').glob('*.json'))
        assert len(calls) == 13 and all(read_json(p)['status'] == 'completed' for p in calls)
        durable(run / 'summary.json', dict(status='completed', scores=scores,
            real_frame_text_pairs=128, real_batches=12, synthetic_batches=1, final_qa_calls=0,
            artifacts={str(p):sha256(p) for p in calls}, seconds=time.perf_counter()-started, utc=utc()))
        print('Complete:128 pairs,12 real batches,QA0', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--parent', default='framegroup_embedding_pilot2_20261002_r1')
    parser.add_argument('--gpu', choices=list(map(str, range(8))), default='0')
    parser.add_argument('--stage', choices=['prepare', 'run'], default='prepare')
    args = parser.parse_args()
    import re
    assert all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', v) for v in [args.run_id, args.parent])
    target = ROOT / 'outputs/analysis' / args.run_id
    try:
        execute(target, ROOT / 'outputs/analysis' / args.parent, args.gpu, args.stage)
    except BaseException as exc:
        durable(target / ('failure_'+str(time.time_ns())+'.json'), dict(error=repr(exc), utc=utc()))
        raise
