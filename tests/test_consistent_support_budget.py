"""预算支持统一的单因素约束、退化关系与稳定性测试。"""
import copy
import numpy as np
import pytest
from videoqa_methods.consistent_support_budget import allocate_weighted_support,select_consistent_support
from videoqa_methods.coarse_eligibility import coarse_selection_cells,select_global_coarse_two_pass
from videoqa_audit.density_reward_decay import select_density_decay

def fixture():
    """接口：固定两区域、多种子与归一化视觉特征，不需要真实视频/模型。"""
    rng=np.random.default_rng(2027);f=rng.normal(size=(12,8)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True)
    c=[dict(candidate_index=i,source_pts=i,source_frame_index=i,timestamp_seconds=float(i)) for i in range(12)]
    p=dict(regions=[dict(region_id=0,left_fraction='0',right_fraction='5',anchor_index=0,seed_indices=[0,1,2],weight=3),
                    dict(region_id=1,left_fraction='6',right_fraction='11',anchor_index=6,seed_indices=[6,7,8],weight=3)])
    return c,f,p,dict(start_pts=0,time_base='1',duration_fraction='12'),dict(frame_budget=8)

def test_equal_weights_recover_original_allocation():
    a=fixture();old,t=select_density_decay(*a);new,u=allocate_weighted_support(*a,[.5]*12)
    assert old==new and t['region_budgets']==u['region_budgets'] and t['rounds']==u['rounds']

def test_single_seed_regions_recover_original():
    a=fixture()
    for r in a[2]['regions']:r['seed_indices']=[r['anchor_index']]
    assert allocate_weighted_support(*a,np.linspace(.1,.9,12))[0]==select_density_decay(*a)[0]

def test_two_pass_support_values_identical():
    a=fixture();_,old=select_density_decay(*a)
    ids,t=select_consistent_support(*a,old['region_budgets'],np.linspace(.1,.9,12))
    assert len(ids)==8 and {0,6}<=set(ids)
    for rid,v in t['supports'].items():assert v['values']==t['budget_trace']['supports'][rid]

def test_equal_weights_full_method_matches_exp04():
    a=fixture();_,t=select_density_decay(*a)
    assert select_consistent_support(*a,t['region_budgets'],[.5]*12)[0]==select_global_coarse_two_pass(*a,t['region_budgets'],[.5]*12)[0]

def test_no_input_mutation_and_score_scale_invariance():
    a=fixture();p=copy.deepcopy(a[2]);f=a[1].copy();scores=np.linspace(.1,.9,12)
    assert allocate_weighted_support(*a,scores)[0]==allocate_weighted_support(*a,scores/2)[0]
    assert p==a[2] and np.array_equal(f,a[1])

def test_zero_diversity_stable_fill_and_unique_budget():
    a=fixture();a[1][:]=0;a[1][:,0]=1
    ids,t=allocate_weighted_support(*a,[.5]*12)
    assert len(ids)==len(set(ids))==8
    assert [r['winner']['candidate_index'] for r in t['rounds']]==[1,2,3,4,5,7]

@pytest.mark.parametrize('scores',[[float('nan')]*12,[-1]*12,[2]*12,[.5]])
def test_bad_scores_rejected(scores):
    with pytest.raises(ValueError):allocate_weighted_support(*fixture(),scores)
