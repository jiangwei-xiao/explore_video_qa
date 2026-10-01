"""窗内资格诊断的配额、保护及边界检查。"""
import numpy as np
import pytest
from videoqa_methods.density_selection import build_regions
from videoqa_audit.window_competition import select_inside_windows


def fixture():
    v=dict(start_pts=0,time_base='1/4',duration_fraction='26')
    c=[dict(candidate_index=i,source_frame_index=i,source_pts=4*i+1,timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(25)]
    s=[.1]*25;s[2]=.9;s[20]=.8
    cfg=dict(seed_budget=2,frame_budget=4,core_gap_seconds='2',window_radius_seconds='1',refinement_min_core_seeds=3)
    return dict(candidates=c,initial_scores=s,video=v,plan=build_regions(c,s,v,cfg)),np.eye(25,dtype=np.float32),cfg


def test_only_window_candidates_with_original_quota():
    p,f,cfg=fixture();ids,trace=select_inside_windows(p,f,{'0':2,'1':2},cfg)
    assert set(ids)<={1,2,3,19,20,21} and {2,20}<=set(ids)
    assert len(ids)==4 and trace['region_budgets']=={'0':2,'1':2}


def test_insufficient_capacity_does_not_expand_or_borrow():
    p,f,cfg=fixture();cfg['frame_budget']=5
    with pytest.raises(ValueError,match='capacity'):select_inside_windows(p,f,{'0':4,'1':1},cfg)
