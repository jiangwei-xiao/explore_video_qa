"""完整源PTS映射的变帧率、起点、去重及末端边界测试。"""
import numpy as np
import pytest
from videoqa_methods.optional_region_pool import map_optional_pool


def example(origin=0):
    """构造不能按平均FPS恢复帧号的非均匀PTS序列。"""
    pts=np.asarray([0,3,7,8,13,17,23,27],dtype=np.int64)+origin
    frames=[]
    for cid,index in enumerate([1,4,6]):
        p=int(pts[index]);t=(p-origin)/10
        frames.append(dict(candidate_index=cid,source_frame_index=index,source_pts=p,
            source_seconds=p/10,timestamp_seconds=t,requested_seconds=cid+.25))
    old=dict(initial_count=3,candidates=frames,initial_scores=[.9,.8,.7],
             video=dict(time_base='1/10',start_pts=origin,duration_fraction='3'))
    cfg=dict(seed_budget=64,frame_budget=32,core_gap_seconds=2,window_radius_seconds=4,
        refinement_min_core_seeds=3,fine_fps=2,coarse_fps=1,phase_fraction='1/4')
    return old,cfg,pts


def test_vfr_mapping_dedup_and_last_target():
    """0.75映射到0.8；1.75映射到已有2.3并去重；2.75后无源帧。"""
    pool=map_optional_pool(*example())
    assert pool['new_candidate_count']==1
    frame=pool['candidates'][-1]
    assert frame['source_frame_index']==3 and frame['source_pts']==8
    assert frame['requested_fraction']=='3/4'
    assert [x['status'] for x in pool['refinement'][0]['mappings']]==['added','duplicate_source','no_source_in_window']


def test_nonzero_start_pts():
    """相对目标与绝对PTS起点分开处理。"""
    pool=map_optional_pool(*example(100))
    frame=pool['candidates'][-1]
    assert frame['source_pts']==108 and frame['timestamp_seconds']==.8


def test_mismatched_source_index_rejected():
    """完整索引必须与每一个冻结粗帧身份一致。"""
    old,cfg,pts=example();pts[1]=4
    with pytest.raises(ValueError):map_optional_pool(old,cfg,pts)


def test_nonmonotone_index_rejected():
    """重复PTS不得造成候选或帧号歧义。"""
    old,cfg,pts=example();pts[2]=pts[1]
    with pytest.raises(ValueError):map_optional_pool(old,cfg,pts)
