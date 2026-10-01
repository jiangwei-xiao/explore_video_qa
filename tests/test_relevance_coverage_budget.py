"""原问题加权覆盖预算的退化等价、预算和特征身份测试。"""
import numpy as np
import pytest
from videoqa_methods.density_selection import build_regions
from videoqa_methods.option_seed_entry import select_option_pool
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage


def fixture(scores):
    c=[dict(candidate_index=i,source_frame_index=i,source_pts=4*i+1,timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(15)]
    v=dict(start_pts=0,time_base='1/4',duration_fraction='16');cfg=dict(seed_budget=3,frame_budget=7,core_gap_seconds='2',window_radius_seconds='1',refinement_min_core_seeds=3)
    f=np.random.default_rng(2027).random((15,8)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True)
    p=dict(video=v,candidates=c,initial_count=15,initial_scores=scores,plan=build_regions(c,scores,v,cfg))
    ids,trace=select_option_pool(p,f,cfg);return p,f,ids,trace


@pytest.mark.parametrize('level',[0.,.2,1.])
def test_equal_scores_exactly_reproduce_original(level):
    p,f,old,trace=fixture([level]*15);new,detail=select_relevance_coverage(p,f,trace)
    assert new==old and detail['region_budgets']==trace['region_budgets']
    assert detail['objective']==trace['objective']


def test_prefixes_and_total_fixed():
    scores=[.1]*15
    for i in [1,7,13]:scores[i]=.9
    p,f,old,trace=fixture(scores);new,detail=select_relevance_coverage(p,f,trace)
    assert len(new)==len(set(new))==7
    for before,after in zip(trace['regions'],detail['regions']):
        assert before['prefix']==after['prefix'] and before['reference_coarse_indices']==after['reference_coarse_indices']
        assert abs(sum(after['reference_weights'])-1)<1e-12
    assert detail['objective']+1e-12>=detail['old_allocation_weighted_objective']


def test_mismatched_feature_order_stops():
    p,f,old,trace=fixture([.2]*15)
    with pytest.raises(ValueError,match='coverage mismatch'):select_relevance_coverage(p,f[::-1],trace)
