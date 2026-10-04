"""可选区域生产候选的容量、锚点与候选身份边界。"""
from itertools import product
import numpy as np
import pytest
from videoqa_methods.optional_region_selection import build_optional_regions,allocate_optional,select_optional_regions


def make_pool(times,budget):
    """构造真实四分之一秒时钟的合成池，不加载视觉模型。"""
    video=dict(start_pts=0,time_base='1/4',duration_fraction=str(times[-1]+1))
    candidates=[dict(candidate_index=i,source_pts=int(t*4),source_frame_index=int(t*4),
        requested_seconds=t,timestamp_seconds=t) for i,t in enumerate(times)]
    scores=[.5]*len(times)
    cfg=dict(seed_budget=64,frame_budget=budget,core_gap_seconds=2,window_radius_seconds=4,refinement_min_core_seeds=3)
    plan=build_optional_regions(candidates,scores,video,cfg)
    return dict(video=video,candidates=candidates,initial_scores=scores,initial_count=len(times),plan=plan)


def test_more_regions_than_budget_is_supported():
    """40个定位提议只分配2帧，不能把规划容量64误作最终帧预算。"""
    pool=make_pool([i*20+.25 for i in range(40)],2)
    selected,trace=select_optional_regions(pool,np.eye(40,dtype=np.float32),2)
    assert selected==[0,1] and len(trace['zero_quota_regions'])==38
    assert trace['active_anchors']==selected and trace['effective_frame_budget']==2


def test_allocation_matches_exhaustive():
    """小规模穷举核对目标与同分规则。"""
    curves=[[0,.2,.4],[0,.9,1],[0,.4,.8]];weights=[1,2,1]
    expected=max((sum(w*c[n] for w,c,n in zip(weights,curves,ns)),ns)
                 for ns in product(range(3),repeat=3) if sum(ns)==2)
    counts,value=allocate_optional(curves,weights,2)
    assert (value,tuple(counts))==expected


def test_short_pool_retains_all_unique_candidates():
    """预算是上限，容量不足时不重复补齐。"""
    pool=make_pool([.25,1.25,2.25],32)
    selected,trace=select_optional_regions(pool,np.eye(3,dtype=np.float32))
    assert selected==[0,1,2] and trace['effective_frame_budget']==3


def test_fine_frame_cannot_use_expanded_selection_cell():
    """全局粗资格不能被误用成新增细帧的解码窗口。"""
    pool=make_pool([.25,20.25],2)
    pool['candidates'].append(dict(candidate_index=2,source_pts=43,source_frame_index=43,
                                  requested_seconds=10.75,timestamp_seconds=10.75))
    with pytest.raises(ValueError,match='细帧'):
        select_optional_regions(pool,np.eye(3,dtype=np.float32),2)


def test_nonmonotone_or_nonfinite_curve_rejected():
    """不接受损坏的覆盖资产。"""
    for curve in [[0,.8,.7],[0,float('nan')]]:
        with pytest.raises(ValueError):allocate_optional([curve],[1],1)
