#!/usr/bin/env python3
"""固定十一题的查询信号传递诊断：同种子、同粗池，只改变局部种子支持权重。"""
import argparse
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.query_input_probe import WITNESSES
from videoqa_audit.joint_query_views import select_joint_views
from videoqa_audit.option_coarse_replay import replay_coarse
from videoqa_audit.query_seed_support import query_supported_seed_weights
from videoqa_methods.prefix_coverage_budget import select_prefix_coverage
from videoqa_budget32.selection import config
SOURCE=ROOT/'outputs/analysis/option_only_pool11_20261002_r1'


def execute(run):
    """接口：冻结已有11题与分数，不评分、不补查、不回答，不改变正在运行的200题。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'coordinator.lock'):
        prior=read_json(SOURCE/'summary.json')
        assert prior['status']=='completed'
        ids=read_json(SOURCE/'protocol.json')['question_ids']
        sources=dict(prior['artifacts'])
        for q in ids:
            item=read_json(SOURCE/'inputs'/(q+'.json'))
            for p in [Path(item['pool_path']),Path(item['feature_path']),SOURCE/'inputs'/(q+'.json')]:
                sources[str(p)]=sha256(p)
        for name in ['scripts/analyze_query_seed_support.py','src/videoqa_audit/query_seed_support.py',
                     'src/videoqa_audit/option_coarse_replay.py','src/videoqa_methods/prefix_coverage_budget.py',
                     'src/videoqa_methods/weighted_seed_support.py']:
            sources[str(ROOT/name)]=sha256(ROOT/name)
        spec=dict(question_ids=ids,sources=sources,new_model_calls=0,configuration=config(32,32),
            hypothesis='Same option-only five-view seeds; replace question-only relative seed support by max of within-query regional relative support; no new coefficient')
        path=run/'protocol.json'
        if path.exists():assert read_json(path)==spec
        else:durable(path,spec)
        for p,h in sources.items():assert sha256(p)==h,p
        if (run/'summary.json').exists():
            for p,h in read_json(run/'summary.json')['artifacts'].items():assert sha256(p)==h
            print('Completed recovery:0 model calls',flush=True)
            return
        facts=[]
        for q in ids:
            dest=run/'questions'/(q+'.json')
            item=read_json(SOURCE/'inputs'/(q+'.json'))
            pool=read_json(item['pool_path'])
            n=pool['initial_count']
            candidates=pool['candidates'][:n]
            if dest.exists():
                result=read_json(dest)
                assert result['protocol_sha256']==sha256(path)
            else:
                # 步骤1：五路种子保持不变，所有后续比较只使用同一初始粗帧池。
                features=np.load(item['feature_path'],allow_pickle=False)[:n].copy()
                options=read_json(SOURCE/'results'/(q+'.json'))['scores']
                seeds,_=select_joint_views(candidates,pool['initial_scores'],options)
                before=replay_coarse(candidates,features,pool['initial_scores'],seeds,pool['video'],spec['configuration'])
                weights,weight_trace=query_supported_seed_weights(before['plan'],[pool['initial_scores'],*options])
                # 步骤2：原函数中initial_scores只决定L与粗前缀长度；锚点、D、覆盖目标不变。
                selected,trace=select_prefix_coverage(candidates,features,before['plan'],pool['video'],spec['configuration'],
                    before['trace']['region_budgets'],weights)
                assert trace['protected_indices']==before['trace']['protected_indices']
                assert trace['eligible_indices']==before['trace']['eligible_indices']
                assert len(selected)==len(set(selected))==32
                # 相同查询退化对照必须逐帧一致，不仅检查代理目标近似相同。
                identical,_=query_supported_seed_weights(before['plan'],[pool['initial_scores']]*5)
                replay,_=select_prefix_coverage(candidates,features,before['plan'],pool['video'],spec['configuration'],
                    before['trace']['region_budgets'],identical)
                assert replay==before['selected_indices']
                result=dict(question_id=q,protocol_sha256=sha256(path),plan=before['plan'],
                    before_indices=before['selected_indices'],after_indices=selected,
                    before_trace=before['trace'],after_trace=trace,weight_trace=weight_trace,
                    identical_query_replay=True,new_model_calls=0)
                durable(dest,result)
            # 步骤3：仅复用已冻结部分见证，不以答案对错解释选帧好坏。
            for label,indices,meaning in WITNESSES.get(q,[]):
                facts.append(dict(question_id=q,fact=label,meaning=meaning,
                    before=bool(set(indices)&set(result['before_indices'])),after=bool(set(indices)&set(result['after_indices']))))
            durable(run/'status.json',dict(status='running',completed=ids.index(q)+1,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        summary=dict(status='completed',facts=facts,before=sum(f['before'] for f in facts),after=sum(f['after'] for f in facts),
            gains=[f for f in facts if f['after'] and not f['before']],losses=[f for f in facts if f['before'] and not f['after']],
            new_model_calls=0,artifacts={str(run/'questions'/(q+'.json')):sha256(run/'questions'/(q+'.json')) for q in ids},utc=utc())
        durable(run/'summary.json',summary)
        print({k:v for k,v in summary.items() if k not in ['artifacts','facts']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    import re
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id)
    execute(ROOT/'outputs/analysis'/args.run_id)
