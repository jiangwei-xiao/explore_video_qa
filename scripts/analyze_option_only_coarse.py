#!/usr/bin/env python3
"""原选项五路入口的同粗池重放，隔离查询变化与新增补查，零模型调用。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import multiprocessing as mp
import os
from pathlib import Path
import sys
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive, utc
from videoqa_audit.query_input_probe import BASE
from videoqa_audit.joint_query_views import select_joint_views
from videoqa_audit.option_coarse_replay import replay_coarse
from videoqa_methods.density_selection import source_time
from videoqa_budget32.selection import config

SOURCE = ROOT/'outputs/analysis/option_only_dev200_20261003_r1'
OLD = ROOT/'outputs/analysis/joint_query_views200_20261001_r1'
FACTS = ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'


def prepare(run):
    """接口：完整评分闭合后冻结200题、粗资产和旧同池结果，不依赖问答标签。"""
    scored = read_json(SOURCE/'summary.json')
    assert scored['status'] == 'completed' and scored['questions'] == 200 and scored['qa_calls'] == 0
    ids = read_json(SOURCE/'protocol.json')['question_ids']
    sources = {str(SOURCE/'summary.json'):sha256(SOURCE/'summary.json'), str(FACTS):sha256(FACTS)}
    sources.update(scored['artifacts'])
    for name in ['scripts/analyze_option_only_coarse.py','src/videoqa_audit/joint_query_views.py',
                 'src/videoqa_audit/option_coarse_replay.py','src/videoqa_methods/prefix_coverage_budget.py']:
        sources[str(ROOT/name)] = sha256(ROOT/name)
    for q in ids:
        for p in [BASE/'pools'/(q+'.json'), BASE/'features'/(q+'.npy'), OLD/'questions'/(q+'.json')]:
            sources[str(p)] = sha256(p)
    spec = dict(question_ids=ids,sources=sources,configuration=config(32,32),new_model_calls=0)
    run.mkdir(parents=True,exist_ok=True)
    pp = run/'protocol.json'
    if pp.exists():
        assert read_json(pp) == spec
    else:
        durable(pp,spec)
    for p,h in sources.items():
        assert sha256(p) == h
    return spec


def worker(job):
    """接口：保持原问题支持、锚点、局部顺序公式及DP，只替换补充查询来源。"""
    run,q = Path(job[0]),job[1]
    spec = read_json(run/'protocol.json')
    dest = run/'questions'/(q+'.json')
    if dest.exists():
        result = read_json(dest)
        assert result['protocol_sha256'] == sha256(run/'protocol.json')
        return
    pool = read_json(BASE/'pools'/(q+'.json'))
    n = pool['initial_count']
    candidates = pool['candidates'][:n]
    features = np.load(BASE/'features'/(q+'.npy'),allow_pickle=False)[:n].copy()
    scores = read_json(SOURCE/'results'/(q+'.json'))['scores']
    seeds,ranking = select_joint_views(candidates,pool['initial_scores'],scores)
    ranking['view_names'] = ['question_only','option_A','option_B','option_C','option_D']
    # 步骤1：仅粗池重放不会触发几何计划中的refine标记，不伪装正式完整输入。
    after = replay_coarse(candidates,features,pool['initial_scores'],seeds,pool['video'],spec['configuration'])
    before = read_json(OLD/'questions'/(q+'.json'))['conditions']['five_views']
    if set(before['plan']['seed_indices']) == set(seeds):
        assert before['selected_indices'] == after['selected_indices']
    compact = dict(plan=after['plan'],selected_indices=after['selected_indices'],
        selected_frames=[candidates[i] for i in after['selected_indices']],region_budgets=after['trace']['region_budgets'])
    durable(dest,dict(question_id=q,protocol_sha256=sha256(run/'protocol.json'),
        conditions=dict(question_options=before,options_only=compact),ranking=ranking,qa_calls=0))


def summarize(run,spec):
    """接口：固定42题75事实，种子、窗口、最终粗帧三级独立计数，不继承旧答案。"""
    records = {q:read_json(run/'questions'/(q+'.json')) for q in spec['question_ids']}
    facts = []
    for fact in read_json(FACTS)['entries']:
        q = fact['question_id']
        pool = read_json(BASE/'pools'/(q+'.json'))
        pts = set(fact['witness_pts'])
        known = [c for c in pool['candidates'][:pool['initial_count']] if c['source_pts'] in pts]
        assert known
        hits = {}
        for name,cond in records[q]['conditions'].items():
            hits[name] = dict(seed=bool(pts & {pool['candidates'][i]['source_pts'] for i in cond['plan']['seed_indices']}),
                selected=bool(pts & {c['source_pts'] for c in cond['selected_frames']}),
                window=any(Fraction(r['left_fraction']) <= source_time(c,pool['video']) <= Fraction(r['right_fraction'])
                    for r in cond['plan']['regions'] for c in known))
        facts.append(dict(question_id=q,fact_id=fact['fact_id'],hits=hits))
    qids = {f['question_id'] for f in facts}
    assert len(facts)==75 and len(qids)==42
    metrics = {name:{stage:dict(count=sum(f['hits'][name][stage] for f in facts),
        macro=sum(sum(f['hits'][name][stage] for f in facts if f['question_id']==q)/sum(f['question_id']==q for f in facts) for q in qids)/42)
        for stage in ['seed','window','selected']} for name in ['question_options','options_only']}
    changes = {stage:{kind:[(f['question_id'],f['fact_id']) for f in facts
        if (f['hits']['question_options'][stage],f['hits']['options_only'][stage]) == pattern]
        for kind,pattern in [('gained',(False,True)),('lost',(True,False))]} for stage in ['seed','window','selected']}
    durable(run/'summary.json',dict(status='completed',metrics=metrics,changes=changes,facts=facts,
        changed_seeds=[q for q,r in records.items() if set(r['conditions']['question_options']['plan']['seed_indices'])!=set(r['conditions']['options_only']['plan']['seed_indices'])],
        changed_selection=[q for q,r in records.items() if r['conditions']['question_options']['selected_indices']!=r['conditions']['options_only']['selected_indices']],
        qa_calls=0,new_model_calls=0,artifacts={str(run/'questions'/(q+'.json')):sha256(run/'questions'/(q+'.json')) for q in records},utc=utc()))
    print(metrics,changes,flush=True)


def execute(run):
    """接口：四进程CPU离线验证，来源异常停止，恢复不启动模型。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'coordinator.lock'):
        spec=prepare(run)
        if (run/'summary.json').exists():
            for p,h in read_json(run/'summary.json')['artifacts'].items():
                assert sha256(p)==h
            print('Completed recovery:0 model calls',flush=True)
            return
        # 步骤2：全部200题都执行同一公式，不以代理得失过滤输入。
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as ex:
            futures=[ex.submit(worker,(str(run),q)) for q in spec['question_ids']]
            for i,f in enumerate(as_completed(futures),1):
                f.result()
                durable(run/'status.json',dict(status='running',completed=i,total=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤3：离线差异报告不包含准确率字段。
        summarize(run,spec)
        durable(run/'status.json',dict(status='completed',completed=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    import re
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id)
    execute(ROOT/'outputs/analysis'/args.run_id)
