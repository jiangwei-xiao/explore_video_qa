"""更大种子数几何预检与冻结区域实现的对照。"""
import importlib.util
from pathlib import Path
import pytest
from videoqa_methods.density_selection import build_regions
from videoqa_runtime.baseline_selection import ProtocolError

spec=importlib.util.spec_from_file_location('seed_geometry',Path(__file__).parents[1]/'scripts/analyze_seed64_geometry.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def inputs(times):
    """构造四分之一秒PTS，保留源序号与请求网格的区别。"""
    video=dict(start_pts=0,time_base='1/4',duration_fraction=str(times[-1]+1))
    candidates=[dict(candidate_index=i,source_pts=int(t*4),source_frame_index=i,
        timestamp_seconds=t,requested_seconds=t) for i,t in enumerate(times)]
    return candidates,[.5]*len(times),video


def test_overlapping_windows_match_reference():
    """窗口共享端点也合并，但不误称同一个原核心。"""
    c,s,v=inputs([.25,8.25,18.25])
    cfg=dict(seed_budget=64,frame_budget=64,core_gap_seconds=2,window_radius_seconds=4,refinement_min_core_seeds=3)
    assert module.geometry(c,s,v,64)==module.signature(build_regions(c,s,v,cfg))


def test_capacity_failure_reported_not_silently_merged():
    """几何层可报告64个区域；生产K32规则仍拒绝，不绕过其保护。"""
    c,s,v=inputs([20*i+.25 for i in range(64)])
    assert len(module.geometry(c,s,v,64)['regions'])==64
    with pytest.raises(ProtocolError):
        build_regions(c,s,v,dict(seed_budget=64,frame_budget=32,core_gap_seconds=2,window_radius_seconds=4,refinement_min_core_seeds=3))


def test_short_pool_and_refinement_trigger():
    """短池不重复种子，三个相连种子仍触发原条件补查。"""
    c,s,v=inputs([.25,2.25,4.25])
    plan=module.geometry(c,s,v,64)
    assert len(plan['seed_indices'])==3 and plan['regions'][0]['refine']
    assert plan['regions'][0]['left_fraction']=='0'
