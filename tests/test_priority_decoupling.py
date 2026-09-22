"""P1纯选择测试：唯一差异是跨区w*u；不运行模型。"""
import math
import numpy as np
import pytest
from videoqa_audit.priority_decoupling import select_decoupled
from videoqa_audit.density_local_validation import supported_selection


def fixture(extra=False):
    """构造密集近重复区域与稀疏补充区域，验证两层效用不是同一个目标。"""
    f=np.array([[1,0,0],[.94,math.sqrt(1-.94**2),0],[1,0,0],[1,0,0],[1,0,0],[0,0,1],[math.sqrt(.75),0,.5]]+([[math.sqrt(.96),0,.2]] if extra else []),dtype=np.float32)
    rows=[dict(candidate_index=i,source_pts=i,source_frame_index=i,timestamp_seconds=float(i)) for i in range(len(f))]
    video=dict(start_pts=0,time_base='1',duration_fraction='10')
    plan={'regions':[dict(region_id=0,left_fraction='0',right_fraction='4',anchor_index=0,seed_indices=[0,1,2,3,4],weight=5),
                     dict(region_id=1,left_fraction='5',right_fraction='9',anchor_index=5,seed_indices=[5],weight=1)]}
    return rows,f,plan,video,{'frame_budget':3}


def test_only_global_priority_changes():
    args=fixture();old,_=supported_selection(*args);new,trace=select_decoupled(*args)
    assert old==[0,1,5] and new==[0,5,6]
    w=trace['rounds'][0]['winner'];assert w['utility']==pytest.approx(.25) and w['priority']==pytest.approx(.5)


def test_local_proposal_remains_soft_not_max_novelty():
    args=fixture(extra=True);indices,_=select_decoupled(*args)
    assert 6 in indices and 7 not in indices


def test_one_region_matches_original_soft_order():
    rows,f,_,video,cfg=fixture();plan={'regions':[dict(region_id=0,left_fraction='0',right_fraction='9',anchor_index=0,seed_indices=[0,1,5],weight=3)]}
    assert select_decoupled(rows,f,plan,video,cfg)[0]==supported_selection(rows,f,plan,video,cfg)[0]


def test_zero_gain_fills_unique_budget_in_time_order():
    rows,f,plan,video,cfg=fixture();f[:]=[1,0,0];cfg['frame_budget']=7
    ids,t=select_decoupled(rows,f,plan,video,cfg)
    assert ids==list(range(7));assert [x['winner']['candidate_index'] for x in t['rounds']]==[1,2,3,4,6]


def test_invalid_features_are_rejected():
    rows,f,plan,video,cfg=fixture();f[1]=[0,0,0]
    with pytest.raises(ValueError):select_decoupled(rows,f,plan,video,cfg)


def test_budget_cap_and_protection():
    rows,f,plan,video,cfg=fixture();cfg['frame_budget']=20
    ids,t=select_decoupled(rows,f,plan,video,cfg)
    assert len(ids)==7 and set(t['protected_indices'])<=set(ids)
