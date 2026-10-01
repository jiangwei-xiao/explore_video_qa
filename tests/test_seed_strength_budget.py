"""区域检索强度的等价边界和预算保护测试。"""
import numpy as np
import pytest
from videoqa_methods.density_selection import build_regions
from videoqa_methods.option_seed_entry import select_option_pool
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_methods.seed_strength_budget import select_seed_strength_budget


def fixture(scores):
    c=[dict(candidate_index=i,source_frame_index=i,source_pts=4*i+1,timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(15)]
    v=dict(start_pts=0,time_base='1/4',duration_fraction='16');cfg=dict(seed_budget=3,frame_budget=7,core_gap_seconds='2',window_radius_seconds='1',refinement_min_core_seeds=3)
    f=np.random.default_rng(1).random((15,8)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True)
    p=dict(candidates=c,video=v,initial_count=15,initial_scores=scores,plan=build_regions(c,scores,v,cfg))
    _,trace=select_option_pool(p,f,cfg);return p,f,trace


@pytest.mark.parametrize('value',[0.,.2,1.])
def test_equal_strength_is_exact_parent(value):
    p,f,t=fixture([value]*15);parent,_=select_relevance_coverage(p,f,t);new,trace=select_seed_strength_budget(p,f,t)
    assert parent==new and all(r['query_strength_factor']==1 for r in trace['regions'])


def test_strength_scale_and_original_prefixes():
    scores=[.01]*15;scores[1]=.9;scores[7]=.5;scores[13]=.2
    p,f,t=fixture(scores);ids,trace=select_seed_strength_budget(p,f,t)
    assert len(ids)==len(set(ids))==7
    assert {r['anchor_index'] for r in p['plan']['regions']}<=set(ids)
    assert trace['objective']+1e-12>=trace['parent_allocation_objective']
    factors=[r['query_strength_factor'] for r in trace['regions']]
    assert max(factors)==1 and min(factors)<1
    for r,old in zip(trace['regions'],t['regions']):assert r['prefix']==old['prefix']
