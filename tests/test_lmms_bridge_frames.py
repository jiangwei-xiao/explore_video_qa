"""均匀源索引与真实PTS规则验收，不运行模型。"""
import numpy as np
import pytest
from videoqa_lmms_bridge.frames import native_indices,rows_at_indices


@pytest.mark.parametrize('n',[1,2,11,14,16,17,103])
def test_native_cap_unique_endpoints(n):
    ids=native_indices(n)
    assert len(ids)==len(set(ids))==min(n,16)
    assert ids==sorted(ids) and ids[0]==0 and ids[-1]==n-1


def test_reference_floor_not_old_round():
    assert native_indices(18)==np.linspace(0,17,16,dtype=int).tolist()
    assert native_indices(18)!=np.rint(np.linspace(0,17,16)).astype(int).tolist()


def test_true_vfr_pts_and_origin():
    pts=[1000,1033,1080,1200]
    v=dict(start_pts=1000,time_base='1/1000',duration_seconds=.3)
    rs=rows_at_indices(pts,v,[0,2,3])
    assert [r['timestamp_seconds'] for r in rs]==[0.,.08,.2]
    assert [r['source_frame_index'] for r in rs]==[0,2,3]


def test_empty_fails():
    with pytest.raises(ValueError):native_indices(0)
