"""相关性加权支持的代数退化、来源一致性与预算约束测试。"""
import numpy as np
import pytest
from videoqa_methods.weighted_seed_support import select_weighted_seed_support,weighted_support
from videoqa_audit.local_distance_scale import select_local_distance

def fixture():
    """接口：两区域、多个种子及模拟细帧，全部特征归一化。"""
    f=np.random.default_rng(2027).normal(size=(12,8)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True)
    c=[dict(candidate_index=i,source_pts=i,source_frame_index=i,timestamp_seconds=float(i)) for i in range(12)]
    p=dict(regions=[dict(region_id=0,left_fraction='0',right_fraction='5',anchor_index=0,seed_indices=[0,1,2]),
                    dict(region_id=1,left_fraction='6',right_fraction='11',anchor_index=6,seed_indices=[6,7])])
    return c,f,p,dict(start_pts=0,time_base='1',duration_fraction='12'),dict(frame_budget=8),{'0':4,'1':4}

def test_equal_scores_recover_original_rule():
    a=fixture();old=select_local_distance(*a)[0]
    assert select_weighted_seed_support(*a,[.7]*8)[0]==old

def test_zero_scores_explicit_fallback():
    a=fixture();ids,t=select_weighted_seed_support(*a,[0]*8)
    assert ids==select_local_distance(*a)[0]
    assert all(v['all_zero_fallback'] for v in t['supports'].values())

def test_score_scaling_leaves_selection_unchanged():
    a=fixture();scores=np.linspace(.2,.8,8)
    assert select_weighted_seed_support(*a,scores)[0]==select_weighted_seed_support(*a,scores*.5)[0]

def test_single_seed_region_is_unchanged():
    a=fixture()
    for r in a[2]['regions']:r['seed_indices']=[r['anchor_index']]
    assert select_weighted_seed_support(*a,np.linspace(.1,.8,8))[0]==select_local_distance(*a)[0]

def test_new_frames_need_no_itm_score_and_share_support_rule():
    a=fixture();ids,t=select_weighted_seed_support(*a,np.linspace(.8,.1,8))
    assert len(ids)==len(set(ids))==8 and {0,6}<=set(ids)
    assert t['region_budgets']=={'0':4,'1':4} and len(t['supports']['1']['values'])==6

def test_weighted_support_matches_definition():
    f=np.array([[1,0],[0,1],[.6,.8]],dtype=np.float32)
    support,w,_=weighted_support(f,[0,1,2],[0,1],[.8,.2])
    assert w=={0:1.,1:.25}
    assert support==pytest.approx([1,.25,.6])

@pytest.mark.parametrize('scores',[[float('nan')]*8,[-.1]*8,[1.1]*8,[.5]*2])
def test_invalid_scores_or_missing_seed_scores_stop(scores):
    with pytest.raises(ValueError):select_weighted_seed_support(*fixture(),scores)

def test_wrong_total_quota_stops():
    a=fixture();a[-1]['0']=3
    with pytest.raises(ValueError):select_weighted_seed_support(*a,[.5]*8)
