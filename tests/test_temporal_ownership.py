"""时间归属支持的几何定义与exp04单因素退化测试。"""
from fractions import Fraction
import copy
import numpy as np
import pytest
from videoqa_methods.temporal_ownership import temporal_ownership,select_temporal_ownership
from videoqa_methods.coarse_eligibility import select_global_coarse_two_pass

def fixture(full=False):
    rng=np.random.default_rng(2027);f=rng.normal(size=(12,8)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True)
    c=[dict(candidate_index=i,source_pts=i,source_frame_index=i,timestamp_seconds=float(i)) for i in range(12)]
    p=dict(regions=[dict(region_id=0,left_fraction='0',right_fraction='5' if full else '3',anchor_index=1,seed_indices=[1,2],weight=2),
                    dict(region_id=1,left_fraction='6' if full else '7',right_fraction='11',anchor_index=8,seed_indices=[8,9],weight=2)])
    return c,f,p,dict(start_pts=0,time_base='1',duration_fraction='12'),dict(frame_budget=8),{'0':4,'1':4},np.linspace(.1,.9,12)

def test_barycentric_gap_and_tie():
    p=fixture()[2]['regions']
    assert temporal_ownership(3,0,p)==1
    assert temporal_ownership(4,0,p)==.75
    assert temporal_ownership(5,0,p)==.5
    assert temporal_ownership(5,1,p)==.5
    assert temporal_ownership(6,1,p)==.75
    assert temporal_ownership(7,1,p)==1

def test_exterior_and_single_region_no_new_penalty():
    p=fixture()[2]['regions']
    assert temporal_ownership(-1,0,p)==1
    assert temporal_ownership(12,1,p)==1
    assert temporal_ownership(8,0,p[:1])==1

def test_wrong_temporal_owner_rejected():
    with pytest.raises(ValueError):temporal_ownership(6,0,fixture()[2]['regions'])

def test_fractional_clock_exact():
    p=[dict(left_fraction='0',right_fraction='1/3'),dict(left_fraction='2/3',right_fraction='1')]
    assert temporal_ownership(Fraction(1,2),0,p)==.5
    assert temporal_ownership(Fraction(5,12),0,p)==.75

def test_full_coverage_reduces_to_exp04():
    a=fixture(True)
    assert select_temporal_ownership(*a)[0]==select_global_coarse_two_pass(*a)[0]

def test_budget_anchors_and_no_mutation():
    a=fixture();p=copy.deepcopy(a[2]);f=a[1].copy();ids,t=select_temporal_ownership(*a)
    assert len(ids)==len(set(ids))==8 and {1,8}<=set(ids)
    assert t['eligible_indices']==list(range(12)) and a[2]==p and np.array_equal(f,a[1])
    assert all(.5<=v<=1 for r in t['supports'].values() for v in r['ownership'].values())

def test_fine_frame_in_gap_is_not_allowed():
    a=list(fixture());a[-1]=[.5]*4;a[2]['regions']=a[2]['regions'][:1];a[-2].clear();a[-2]['0']=8
    with pytest.raises(ValueError,match='Fine frame escaped'):select_temporal_ownership(*a)

def test_zero_diversity_stable_fill():
    a=fixture();a[1][:]=0;a[1][:,0]=1
    ids,t=select_temporal_ownership(*a)
    assert ids==[0,1,2,3,4,5,6,8]
