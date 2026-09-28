"""前缀覆盖预算的DP最优性、基准可行性与局部规则一致性。"""
import itertools
import numpy as np
import pytest
from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget,select_prefix_coverage
from videoqa_methods.coarse_eligibility import select_global_coarse

def test_dp_matches_bruteforce_with_nonconcave_curves():
    curves=[[0,.5,.51,.95],[0,.4,.6,.7],[0,.6,.7,.71]];weights=[3,2,1];budget=6
    choices=[(sum(w*c[n] for w,c,n in zip(weights,curves,ns)),ns) for ns in itertools.product(range(1,4),repeat=3) if sum(ns)==budget]
    expected=max(choices);allocation,value=allocate_prefix_budget(curves,weights,budget)
    assert (value,tuple(allocation))==expected

def test_dp_tie_prefers_earlier_region():
    assert allocate_prefix_budget([[0,1,1,1],[0,1,1,1]],[1,1],4)[0]==[3,1]

@pytest.mark.parametrize('curves,weights,budget',[([[0,.5,.4]],[1],2),([[0,.5]],[0],1),([[0,.5]],[1],2),([[0,float('nan')]],[1],1)])
def test_bad_curves_or_budget_rejected(curves,weights,budget):
    with pytest.raises(ValueError):allocate_prefix_budget(curves,weights,budget)

def fixture():
    f=np.random.default_rng(2027).normal(size=(12,8)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True)
    c=[dict(candidate_index=i,source_pts=i,source_frame_index=i,timestamp_seconds=float(i)) for i in range(12)]
    p=dict(regions=[dict(region_id=0,left_fraction='0',right_fraction='5',anchor_index=0,seed_indices=[0,1,2],weight=3),
                    dict(region_id=1,left_fraction='6',right_fraction='11',anchor_index=6,seed_indices=[6,7,8],weight=3)])
    return c,f,p,dict(start_pts=0,time_base='1',duration_fraction='12'),dict(frame_budget=8),{'0':4,'1':4},np.linspace(.1,.9,12)

def test_final_prefixes_match_unchanged_local_selector():
    a=fixture();ids,t=select_prefix_coverage(*a)
    assert len(ids)==len(set(ids))==8 and {0,6}<=set(ids)
    assert t['reference_prefix_selection']==select_global_coarse(*a)[0]
    args=a[:5]+(t['region_budgets'],a[-1])
    assert ids==select_global_coarse(*args)[0]
    assert t['objective']>=t['reference_objective']-1e-12

def test_coverage_curves_are_bounded_and_monotone():
    _,t=select_prefix_coverage(*fixture())
    for r in t['regions']:
        assert all(0<=x<=1 for x in r['coverage_curve'])
        assert all(y>=x for x,y in zip(r['coverage_curve'],r['coverage_curve'][1:]))

def test_only_coarse_frames_are_reference_points():
    a=list(fixture());a[-1]=[.5]*9
    _,t=select_prefix_coverage(*a)
    assert set(i for r in t['regions'] for i in r['reference_coarse_indices'])==set(range(9))
