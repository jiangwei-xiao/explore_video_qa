"""固定前缀归因必须扣除其他区域锚点，不将候选资格与最终可达性混淆。"""
import numpy as np
import pytest
from videoqa_audit.option_entry_attribution import prefix_gate,nearest_similarities


def trace():
    return dict(regions=[dict(region_id=0,members=[0,1,2,3,4],prefix=[0,1,2,3]),
                         dict(region_id=1,members=[5,6],prefix=[5,6])],region_budgets={'0':2,'1':2})


def test_selected_and_quota_reachable():
    assert prefix_gate(1,trace(),4)['status']=='selected'
    gate=prefix_gate(2,trace(),4)
    assert gate['status']=='quota_reachable' and gate['maximum_feasible_quota']==3


def test_other_anchor_blocks_fourth_prefix_frame():
    assert prefix_gate(3,trace(),4)['status']=='local_prefix_blocked'
    assert prefix_gate(4,trace(),4)['prefix_position'] is None


def test_bad_membership_rejected():
    with pytest.raises(ValueError):prefix_gate(8,trace(),4)
    t=trace();t['regions'][1]['members'].append(0)
    with pytest.raises(ValueError):prefix_gate(0,t,4)


def test_similarity_excludes_self_and_keeps_visual_twins():
    f=np.array([[1,0],[1,0],[0,1]],dtype=np.float32)
    assert nearest_similarities(f,[0,1,2])==[1.,1.,0.]
    assert nearest_similarities(f,[0])==[0.]
    with pytest.raises(ValueError):nearest_similarities(f,[0,0])
