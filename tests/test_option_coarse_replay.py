"""显式种子适配只改变集合；原分数仍决定锚点，不把排序编码当置信度。"""
import numpy as np
import pytest
from videoqa_audit.option_coarse_replay import plan_from_seeds,replay_coarse
from videoqa_methods.density_selection import build_regions


def fixture():
    video=dict(start_pts=0,time_base='1/4',duration_fraction='12')
    candidates=[dict(candidate_index=i,source_frame_index=i,source_pts=4*i+1,
                     timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(10)]
    cfg=dict(seed_budget=3,frame_budget=4,core_gap_seconds='2',window_radius_seconds='1',refinement_min_core_seeds=3)
    return candidates,video,cfg


@pytest.mark.parametrize('scores',[[0.5]*10,list(np.linspace(0,1,10)),[0.0]*10])
def test_old_seed_plan_exact(scores):
    c,v,cfg=fixture();old=build_regions(c,scores,v,cfg)
    assert plan_from_seeds(c,scores,old['seed_indices'],v,cfg)==old


def test_new_seed_membership_keeps_original_anchor_priority():
    c,v,cfg=fixture();scores=[0.1]*10;scores[5]=0.9
    plan=plan_from_seeds(c,scores,[4,5,6],v,cfg)
    assert set(plan['seed_indices'])=={4,5,6}
    assert plan['regions'][0]['anchor_index']==5


def test_invalid_seed_count_or_duplicates_stops():
    c,v,cfg=fixture()
    for seeds in ([1,2],[1,1,2],[1,2,10]):
        with pytest.raises(ValueError):plan_from_seeds(c,[.2]*10,seeds,v,cfg)


def test_same_inputs_replay_deterministically_without_fine_candidates():
    c,v,cfg=fixture();scores=list(np.linspace(0,1,10));f=np.eye(10,dtype=np.float32)
    a=replay_coarse(c,f,scores,[1,5,9],v,cfg);b=replay_coarse(c,f,scores,[1,5,9],v,cfg)
    assert a==b and len(a['selected_indices'])==4
    assert a['new_model_calls']==0 and a['refinement_performed'] is False
