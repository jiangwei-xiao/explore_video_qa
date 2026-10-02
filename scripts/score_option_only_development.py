#!/usr/bin/env python3
"""开发200题原选项评分准备：固定四路文本，复用控制资产，零最终问答。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
import os
from pathlib import Path
import sys
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive, utc, rows
from videoqa_full.prepare import environment
from videoqa_audit import query_input_probe as base
from videoqa_audit.joint_query_views import select_joint_views
from run_option_rank_development import worker, verify as verify_original

SOURCE = ROOT/'outputs/analysis/option_rank_dev200_20261001_r1'
PILOT = ROOT/'outputs/analysis/option_only_pool11_20261002_r1'
DATA = ROOT/'data/manifests/video_mme_development_200_20260920.json'


def verify(run):
    """接口：恢复仅接受固定代码、环境和源资产；缓存来源与新计算分别记账。"""
    spec = read_json(run/'protocol.json')
    assert environment(base.PYTHON) == spec['environment']
    for p,h in spec['sources'].items():
        assert sha256(p) == h, p
    return spec


def prepare(run):
    """接口：冻结200题全量评分，不根据已知答案挑题，不改变窗口或选择公式。"""
    from transformers import BlipProcessor
    run.mkdir(parents=True, exist_ok=True)
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():
            return verify(run)
        old = verify_original(SOURCE)
        ids = [r['question_id'] for r in read_json(DATA)['rows']]
        assert ids == old['question_ids'] and len(set(ids)) == 200
        data = {r['question_id']:r for r in rows()}
        pilot = read_json(PILOT/'summary.json')
        assert pilot['status'] == 'completed'
        for p,h in pilot['artifacts'].items():
            assert sha256(p) == h
        sources = dict(old['sources'])
        for name in ['scripts/score_option_only_development.py','scripts/run_option_rank_development.py',
                     'src/videoqa_audit/joint_query_views.py']:
            sources[str(ROOT/name)] = sha256(ROOT/name)
        for p,h in read_json(ROOT/'outputs/analysis/query_input_pilot9_20261001_r1/protocol.json')['model_files'].items():
            assert sha256(p) == h
            sources[p] = h
        tokenizer = BlipProcessor.from_pretrained(base.MODEL, local_files_only=True).tokenizer
        cached = {}
        frames = batches = 0
        # 步骤1：数据集原选项全文作为四路查询；不产生改写、不读取答案用于规则。
        for q in ids:
            item = read_json(SOURCE/'inputs'/(q+'.json'))
            texts = data[q]['options']
            assert len(texts) == 4
            lengths = [len(tokenizer(t, truncation=False)['input_ids']) for t in texts]
            assert max(lengths) <= 512
            new = dict(item, queries=texts, query_tokens=lengths, cache=None)
            path = run/'inputs'/(q+'.json')
            durable(path, new)
            sources[str(path)] = sha256(path)
            rp = PILOT/'results'/(q+'.json')
            if rp.exists():
                previous = read_json(PILOT/'inputs'/(q+'.json'))
                assert previous['queries'] == texts and previous['pool_path'] == new['pool_path']
                cached[q] = str(rp)
                sources[str(rp)] = sha256(rp)
                sources[str(PILOT/'protocol.json')] = sha256(PILOT/'protocol.json')
            else:
                frames += 4*item['initial_count']
                batches += 4*((item['initial_count']+15)//16)
        assignment = {str(g):[q for index,q in enumerate(ids) if index % 8 == g and q not in cached] for g in range(8)}
        spec = dict(question_ids=ids, sources=sources, environment=environment(base.PYTHON),
            cached=cached, assignment=assignment, frame_text_scores=frames, batch_calls=batches,
            synthetic_batches=2*sum(bool(v) for v in assignment.values()), qa_calls=0,
            rule='Original question plus four original option-only views; frozen minimum-rank fusion; offline stage only', utc=utc())
        durable(run/'protocol.json', spec)
        # 步骤2：旧评分仅引用并重绑本轮协议，不伪造本轮模型调用。
        for q,p in cached.items():
            durable(run/'results'/(q+'.json'), dict(read_json(p), protocol_sha256=sha256(run/'protocol.json'), cached_from=p))
        print(dict(new_pairs=frames,new_batches=batches,cached=len(cached),qa_calls=0), flush=True)
        return spec


def execute(run):
    """接口：八卡空闲时后台固定分配；运行失败暂停，已完成评分不重复执行。"""
    from videoqa_runtime.baseline_run import available_gpus
    with exclusive(run/'coordinator.lock'):
        spec = verify(run)
        if (run/'summary.json').exists():
            for p,h in read_json(run/'summary.json')['artifacts'].items():
                assert sha256(p) == h
            print('Completed recovery:0 forwards', flush=True)
            return
        assert not (run/'STOP.json').exists()
        active = {g:qs for g,qs in spec['assignment'].items() if any(not (run/'results'/(q+'.json')).exists() for q in qs)}
        if active:
            free,_ = available_gpus(list(active))
            assert set(active) <= set(free), 'GPU occupied; no preemption'
            # 步骤3：仅调用旧入口中的BLIP worker，不调用导出或QA函数。
            with ProcessPoolExecutor(max_workers=len(active), mp_context=mp.get_context('spawn')) as ex:
                pending = {ex.submit(worker,run,g,qs) for g,qs in active.items()}
                while pending:
                    done,pending = wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result()
                    durable(run/'status.json', dict(status='running',pid=os.getpid(),sid=os.getsid(0),
                        completed=len(list((run/'results').glob('*.json'))),total=200,utc=utc()))
        verify(run)
        calls = list((run/'calls').glob('*/*/*.json'))
        assert len(calls) == spec['batch_calls'] and all(read_json(p)['status']=='completed' for p in calls)
        assert sum(read_json(p)['identity']['frames'] for p in calls) == spec['frame_text_scores']
        # 步骤4：只冻结五路种子提议；查询名称修正为原选项，不改旧函数或其快照。
        for q in spec['question_ids']:
            result = read_json(run/'results'/(q+'.json'))
            item = read_json(run/'inputs'/(q+'.json'))
            pool = read_json(item['pool_path'])
            ids,trace = select_joint_views(pool['candidates'][:pool['initial_count']],pool['initial_scores'],result['scores'])
            trace['view_names'] = ['question_only', 'option_A','option_B','option_C','option_D']
            durable(run/'seeds'/(q+'.json'),dict(question_id=q,indices=ids,trace=trace))
        artifacts = {str(p):sha256(p) for folder in ['calls','results','synthetic','seeds'] for p in (run/folder).rglob('*') if p.is_file()}
        durable(run/'summary.json',dict(status='completed',questions=200,qa_calls=0,artifacts=artifacts,utc=utc()))
        durable(run/'status.json',dict(status='completed',completed=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=['prepare','run'],required=True)
    args = parser.parse_args()
    import re
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id)
    run = ROOT/'outputs/analysis'/args.run_id
    if args.stage == 'prepare':
        prepare(run)
    else:
        execute(run)
