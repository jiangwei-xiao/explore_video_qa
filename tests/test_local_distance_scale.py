"""固定配额局部距离尺度的纯逻辑测试，不调用任何模型。"""
import math
import numpy as np
import pytest
from videoqa_audit.local_distance_scale import select_local_distance
from videoqa_audit.density_local_validation import supported_selection


def fixture():
    """归一化特征中，近种子状态与大差异背景之间的有限对照。"""
    f=np.array([[1,0,0],[.85,math.sqrt(1-.85**2),0],[.5,-.5,math.sqrt(.5)]],dtype=np.float32)
    rows=[dict(candidate_index=i,source_pts=i,source_frame_index=i,timestamp_seconds=float(i)) for i in range(3)]
    p={'regions':[dict(region_id=0,left_fraction='0',right_fraction='3',anchor_index=0,seed_indices=[0,1],weight=2)]}
    return rows,f,p,dict(start_pts=0,time_base='1',duration_fraction='4'),dict(frame_budget=2),{'0':2}


def test_changes_scale_without_budget_change():
    a=fixture();new,t=select_local_distance(*a);old,_=supported_selection(*a)
    assert old==[0,2] and new==[0,1] and t['region_budgets']=={'0':2}


def test_normalized_euclidean_identity():
    a=fixture();f=a[1];u=1-float(f[0]@f[1])
    assert math.sqrt(u)==pytest.approx(float(np.linalg.norm(f[0]-f[1]))/math.sqrt(2),abs=1e-6)


def test_single_seed_preference_moves_not_fixed_half():
    a=fixture();a[2]['regions'][0]['seed_indices']=[0]
    a[1][1]=[.5,math.sqrt(.75),0];a[1][2]=[2/3,math.sqrt(5)/3,0]
    assert select_local_distance(*a)[0]==[0,2]
    assert supported_selection(*a)[0]==[0,1]


def test_anchor_only_unchanged():
    a=fixture();a[-2]['frame_budget']=1;a[-1]['0']=1
    assert select_local_distance(*a)[0]==[0]


def test_identical_features_tie_by_time():
    a=fixture();a[1][:]=[1,0,0]
    assert select_local_distance(*a)[0]==[0,1]


def test_fixed_quota_mismatch_refused():
    a=fixture();a[-1]['0']=3
    with pytest.raises(ValueError):select_local_distance(*a)


def test_non_normalized_features_refused():
    a=fixture();a[1][1]*=2
    with pytest.raises(ValueError):select_local_distance(*a)
