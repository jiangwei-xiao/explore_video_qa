"""核心内竞争的边界、质量守恒及退化一致性测试。"""
import numpy as np
from videoqa_methods.density_selection import build_regions
from videoqa_methods.option_seed_entry import select_option_pool
from videoqa_methods.core_anchor_prefix import select_core_anchor_prefix
from videoqa_methods.core_partition_selection import nearest_core,select_core_partition


def fixture(radius):
    c=[dict(candidate_index=i,source_frame_index=i,source_pts=4*i+1,timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(15)]
    v=dict(start_pts=0,time_base='1/4',duration_fraction='16');cfg=dict(seed_budget=3,frame_budget=7,core_gap_seconds='2',window_radius_seconds=str(radius),refinement_min_core_seeds=3)
    f=np.random.default_rng(7).random((15,8)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True)
    scores=[.1]*15;scores[1]=.9;scores[7]=.5;scores[13]=.2
    p=dict(candidates=c,video=v,initial_count=15,initial_scores=scores,plan=build_regions(c,scores,v,cfg))
    _,trace=select_option_pool(p,f,cfg);return p,f,trace


def test_exact_pts_midpoint_tie_prefers_earlier_core():
    intervals={0:(1,3),1:(7,9)}
    assert nearest_core(5,intervals)==0 and nearest_core(6,intervals)==1
    assert nearest_core(-100,intervals)==0 and nearest_core(100,intervals)==1


def test_single_core_regions_exact_parent():
    p,f,t=fixture(1);old,_=select_core_anchor_prefix(p,f,t);new,trace=select_core_partition(p,f,t)
    assert old==new and all(abs(r['reference_mass']-1)<1e-12 for r in trace['regions'])


def test_merged_partition_conserves_candidates_mass_and_anchors():
    p,f,t=fixture(4);selected,trace=select_core_partition(p,f,t)
    assert len(selected)==len(set(selected))==7 and {1,7,13}<=set(selected)
    assert len(trace['regions'])==3
    members=[i for r in trace['regions'] for i in r['members']];assert sorted(members)==list(range(15))
    assert abs(sum(r['reference_mass'] for r in trace['regions'])-1)<1e-12
    assert sum(trace['aggregated_parent_budgets'].values())==7
    assert all(r['support_seed_indices']==p['plan']['regions'][0]['seed_indices'] for r in trace['regions'])
