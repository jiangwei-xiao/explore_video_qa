"""逐选项对称入口与窗口几何检查；无模型调用。"""
from pathlib import Path
import runpy
import numpy as np
from videoqa_methods.density_selection import build_regions
MODULE=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/run_option_route_probe.py'))


def fixture():
    """接口：纯元数据构造，用于验证去重和时间边界。"""
    c=[dict(candidate_index=i,source_pts=i,source_frame_index=i,timestamp_seconds=float(i),requested_seconds=float(i)) for i in range(40)]
    return c,dict(start_pts=0,time_base='1',duration_fraction='40')


def test_all_branches_equal_and_order_invariant():
    c,_=fixture();scores=[np.roll(np.linspace(0,1,40),k*10).tolist() for k in range(4)]
    branches,ids=MODULE['seed_union'](c,scores)
    assert all(len(b)==8 for b in branches) and len(ids)==32
    assert ids==MODULE['seed_union'](c,list(reversed(scores)))[1]


def test_duplicates_not_filled_with_extra_candidates():
    c,_=fixture();s=np.linspace(0,1,40).tolist();branches,ids=MODULE['seed_union'](c,[s]*4)
    assert len(ids)==8 and len(set(ids))==8


def test_windows_match_original_geometry():
    c,v=fixture();scores=np.random.default_rng(2027).random(40).tolist()
    cfg=dict(seed_budget=16,frame_budget=32,core_gap_seconds='2',window_radius_seconds='4',refinement_min_core_seeds=3)
    old=build_regions(c,scores,v,cfg)
    assert MODULE['seed_windows'](c,old['seed_indices'],v)==[{k:r[k] for k in ('left_fraction','right_fraction')} for r in old['regions']]
