"""原核心保护与带下界精确DP测试。"""
import itertools
import numpy as np
import pytest
from videoqa_methods.density_selection import build_regions
from videoqa_methods.option_seed_entry import select_option_pool
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_methods.core_anchor_prefix import allocate_with_minima,select_core_anchor_prefix


def fixture(radius,frame_budget=7):
    c=[dict(candidate_index=i,source_frame_index=i,source_pts=4*i+1,timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(15)]
    v=dict(start_pts=0,time_base='1/4',duration_fraction='16');cfg=dict(seed_budget=3,frame_budget=frame_budget,core_gap_seconds='2',window_radius_seconds=str(radius),refinement_min_core_seeds=3)
    f=np.random.default_rng(5).random((15,8)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True)
    scores=[.1]*15;scores[1]=.9;scores[7]=.5;scores[13]=.2
    p=dict(candidates=c,video=v,initial_count=15,initial_scores=scores,plan=build_regions(c,scores,v,cfg))
    _,trace=select_option_pool(p,f,cfg);return p,f,trace


def test_minima_dp_matches_exhaustive():
    curves=[[0,.2,.4,.8],[0,.4,.5,.6],[0,.1,.7,.9]];weights=[2.,1.,3.];mins=[2,1,2]
    q,value=allocate_with_minima(curves,weights,mins,7)
    choices=[(sum(w*c[n] for w,c,n in zip(weights,curves,k)),k) for k in itertools.product(*[range(m,len(c)) for m,c in zip(mins,curves)]) if sum(k)==7]
    expected=max(choices);assert (value,tuple(q))==expected


def test_single_core_regions_exact_parent():
    p,f,t=fixture(1);expected,_=select_relevance_coverage(p,f,t);actual,new=select_core_anchor_prefix(p,f,t)
    assert actual==expected and new['minimum_quotas']==[1,1,1]


def test_merged_window_preserves_all_core_peaks():
    p,f,t=fixture(4);selected,new=select_core_anchor_prefix(p,f,t)
    assert len(p['plan']['regions'])==1 and new['minimum_quotas']==[3]
    assert {1,7,13}<=set(selected) and len(selected)==7


def test_all_budget_protected_degenerates_to_original_seeds():
    p,f,t=fixture(4,3);selected,_=select_core_anchor_prefix(p,f,t)
    assert selected==[1,7,13]


def test_impossible_minimum_is_rejected():
    with pytest.raises(ValueError):allocate_with_minima([[0,.5]], [1], [2], 2)
