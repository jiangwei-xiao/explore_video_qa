#!/usr/bin/env python3
"""十一题完整粗池的原选项检索验证；复用已验收评分事务，禁止调用问答。"""
import argparse
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive, utc, rows
from videoqa_full.prepare import environment
from videoqa_audit import query_input_probe as base
from videoqa_audit.option_rank_selection import select_option_ranks
from run_option_rank_development import worker

PARENT = ROOT/'outputs/analysis/query_input_pilot9_20261001_r1'
QUESTIONS = list(base.WITNESSES) + ['395-2', '409-3']


def verify(run):
    """接口：实际运行和恢复均核对全部代码、数据、模型、环境，不重算原问题。"""
    spec = read_json(run/'protocol.json')
    assert environment(base.PYTHON) == spec['environment']
    for p, h in spec['sources'].items():
        assert sha256(p) == h, p
    return spec


def prepare(run):
    """接口：固定原九题控制加两题机制案例，冻结四路原选项及批次数，无答案字段。"""
    from transformers import BlipProcessor
    run.mkdir(parents=True, exist_ok=True)
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():
            return verify(run)
        parent = base.verify(PARENT)
        data = {r['question_id']:r for r in rows()}
        tokenizer = BlipProcessor.from_pretrained(base.MODEL, local_files_only=True).tokenizer
        sources = dict(parent['model_files'])
        sources[str(PARENT/'protocol.json')] = sha256(PARENT/'protocol.json')
        for name in ['scripts/run_option_only_pool_probe.py', 'scripts/run_option_rank_development.py',
                     'src/videoqa_audit/option_rank_selection.py', *parent['code_sha256']]:
            sources[str(ROOT/name)] = sha256(ROOT/name)
        frames = batches = 0
        # 步骤1：全部原选项对称处理；影片与粗帧特征身份沿用冻结检查点。
        for q in QUESTIONS:
            row = data[q]
            pp = base.BASE/'pools'/(q+'.json')
            fp = base.BASE/'features'/(q+'.npy')
            cp = base.BASE/'selection_checkpoints'/(q+'.json')
            check = read_json(cp)
            for p in [pp, fp]:
                assert sha256(p) == check['files'][str(p.relative_to(base.BASE))]
            pool = read_json(pp)
            n = pool['initial_count']
            assert n == len(pool['initial_scores'])
            queries = row['options']
            assert len(queries) == 4
            lengths = [len(tokenizer(t, truncation=False)['input_ids']) for t in queries]
            assert max(lengths) <= 512
            alternatives = ['option_rank_dev200_20261001_r1', 'option_routes_controls7_20261001_r1', 'option_routes_pilot2_20261001_r1']
            old = next(ROOT/'outputs/analysis'/a/'results'/(q+'.json') for a in alternatives
                       if (ROOT/'outputs/analysis'/a/'results'/(q+'.json')).exists())
            item = dict(question_id=q, video_id=row['video_id'], queries=queries, query_tokens=lengths,
                pool_path=str(pp), feature_path=str(fp), initial_count=n, cache=None, previous_scores=str(old))
            ip = run/'inputs'/(q+'.json')
            durable(ip, item)
            for p in [pp, fp, cp, ip, old, Path(pool['video']['path'])]:
                sources[str(p)] = sha256(p)
            # 旧全量准备协议提供视频身份，不以本次新算哈希自行证明来源。
            assert sources[pool['video']['path']] == read_json(base.SOURCE/'protocol.json')['video_hashes'][row['video_id']]
            frames += 4*n
            batches += 4*((n+15)//16)
        spec = dict(question_ids=QUESTIONS, sources=sources, environment=environment(base.PYTHON),
            gpus=['0','1'], assignment={'0':QUESTIONS[::2], '1':QUESTIONS[1::2]},
            frame_text_scores=frames, batch_calls=batches, synthetic_batches=4, qa_calls=0,
            selection='Frozen four-route minimum-rank Top32 diagnostic only; not an RD method', utc=utc())
        durable(run/'protocol.json', spec)
        print('Prepared', frames, 'frame-text pairs;', batches, 'batches; QA0', flush=True)
        return spec


def summarize(run, spec):
    """接口：核对完整候选与纯视觉重放，并保留旧新四路排名，不附用历史答案。"""
    calls = list((run/'calls').glob('*/*/*.json'))
    assert len(calls) == spec['batch_calls']
    assert all(read_json(p)['status'] == 'completed' for p in calls)
    assert sum(read_json(p)['identity']['frames'] for p in calls) == spec['frame_text_scores']
    comparison = {}
    for q in spec['question_ids']:
        item = read_json(run/'inputs'/(q+'.json'))
        pool = read_json(item['pool_path'])
        coarse = pool['candidates'][:item['initial_count']]
        result = read_json(run/'results'/(q+'.json'))
        assert result['max_feature_error'] <= 1e-6
        original = read_json(item['previous_scores'])['scores']
        old_ids, old_trace = select_option_ranks(coarse, original)
        new_ids, new_trace = select_option_ranks(coarse, result['scores'])
        assert new_ids == result['selected_indices'] and new_trace == result['trace']
        comparison[q] = dict(old_indices=old_ids, new_indices=new_ids,
            old_trace=old_trace, new_trace=new_trace, max_feature_error=result['max_feature_error'])
    artifacts = {str(p):sha256(p) for folder in ['calls', 'results', 'synthetic'] for p in (run/folder).rglob('*') if p.is_file()}
    durable(run/'summary.json', dict(status='completed', comparisons=comparison,
        frame_text_scores=spec['frame_text_scores'], batch_calls=len(calls), qa_calls=0,
        artifacts=artifacts, utc=utc()))


def execute(run):
    """接口：两卡后台协调；断线不影响运行，任何未知前向停止，不自动重试。"""
    from videoqa_runtime.baseline_run import available_gpus
    with exclusive(run/'coordinator.lock'):
        spec = verify(run)
        if (run/'summary.json').exists():
            for p,h in read_json(run/'summary.json')['artifacts'].items():
                assert sha256(p) == h
            print('Completed recovery:0 forwards', flush=True)
            return
        assert not (run/'STOP.json').exists()
        free, _ = available_gpus(spec['gpus'])
        assert set(spec['gpus']) <= set(free), 'GPU occupied'
        # 步骤2：复用事务式评分和精确RGB核验，不调用旧入口的QA导出或执行阶段。
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn')) as ex:
            pending = {ex.submit(worker, run, gpu, qs) for gpu,qs in spec['assignment'].items()}
            while pending:
                done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for f in done:
                    f.result()
                durable(run/'status.json', dict(status='running', pid=os.getpid(), sid=os.getsid(0),
                    completed=len(list((run/'results').glob('*.json'))), total=11, utc=utc()))
        # 步骤3：候选及账本闭合后才汇总；报告绝不继承历史预测作为新成绩。
        verify(run)
        summarize(run, spec)
        durable(run/'status.json', dict(status='completed', pid=os.getpid(), sid=os.getsid(0), completed=11, utc=utc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--stage', choices=['prepare', 'run'], required=True)
    args = parser.parse_args()
    import re
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id)
    run = ROOT/'outputs/analysis'/args.run_id
    if args.stage == 'prepare':
        prepare(run)
    else:
        execute(run)
