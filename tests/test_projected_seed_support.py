"""支持表征单因素的退化、去重边界和参考不变性测试。"""
import numpy as np
import pytest
from videoqa_methods.density_selection import build_regions
from videoqa_methods.option_seed_entry import select_option_pool
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_methods.projected_seed_support import select_projected_support


def fixture():
    c=[dict(candidate_index=i,source_frame_index=i,source_pts=4*i+1,timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(15)]
    v=dict(start_pts=0,time_base='1/4',duration_fraction='16');cfg=dict(seed_budget=3,frame_budget=7,core_gap_seconds='2',window_radius_seconds='4',refinement_min_core_seeds=3)
    f=np.random.default_rng(6).random((15,8)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True);scores=[.1]*15;scores[1]=.9;scores[7]=.5;scores[13]=.2
    p=dict(candidates=c,video=v,initial_count=15,initial_scores=scores,plan=build_regions(c,scores,v,cfg));_,t=select_option_pool(p,f,cfg);return p,f,t


def test_same_representation_exactly_reproduces_parent():
    p,f,t=fixture();old,_=select_relevance_coverage(p,f,t);new,_=select_projected_support(p,f,f,t);assert old==new


def test_constant_support_features_do_not_replace_raw_novelty():
    p,f,t=fixture();g=np.ones((15,4),dtype=np.float32)/2;selected,trace=select_projected_support(p,f,g,t)
    assert len(selected)==len(set(selected))==7
    for r in trace['regions']:
        chosen=[r['prefix'][0]]
        for step in r['local_steps']:
            expected=float(np.max(np.clip(f[chosen]@f[step['candidate_index']],0,1)))
            assert abs(expected-step['raw_redundancy'])<1e-6
            assert abs(step['projected_support']-1)<1e-6
            chosen.append(step['candidate_index'])
        weights=np.array(r['reference_weights']);refs=r['reference_coarse_indices']
        expected=float(weights@np.max(np.clip(f[refs]@f[r['prefix']].T,0,1),axis=1))
        assert abs(expected-r['coverage_curve'][-1])<1e-6


def test_bad_projection_norm_rejected():
    p,f,t=fixture()
    with pytest.raises(ValueError):select_projected_support(p,f,2*f,t)
