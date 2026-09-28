"""粗选择资格与细采样范围解耦的精确PTS、配额和非变更约束。"""
import copy
import numpy as np
import pytest
from videoqa_methods.coarse_eligibility import coarse_selection_cells,select_global_coarse,select_global_coarse_two_pass
from videoqa_methods.weighted_seed_support import select_weighted_seed_support

def fixture():
    """接口：两个原窗口和中间/首尾粗候选，全部特征归一化。"""
    rng=np.random.default_rng(2027);f=rng.normal(size=(11,8)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True)
    c=[dict(candidate_index=i,source_pts=i,source_frame_index=i,timestamp_seconds=float(i)) for i in range(11)]
    p=dict(regions=[dict(region_id=0,left_fraction='1',right_fraction='3',anchor_index=1,seed_indices=[1,2],weight=2),
                    dict(region_id=1,left_fraction='7',right_fraction='9',anchor_index=8,seed_indices=[8,9],weight=2)])
    return c,f,p,dict(start_pts=0,time_base='1',duration_fraction='11'),dict(frame_budget=4),{'0':2,'1':2},[.5]*11

def test_exact_gap_midpoint_earlier_tie_and_no_mutation():
    a=fixture();saved=copy.deepcopy(a[2]);p=coarse_selection_cells(a[2],a[3])
    assert [(r['left_fraction'],r['right_fraction']) for r in p['regions']]==[('0','5'),('6','11')]
    assert a[2]==saved

def test_all_coarse_eligible_and_original_anchors_preserved():
    a=fixture();ids,t=select_global_coarse(*a)
    assert t['eligible_indices']==list(range(11))
    assert t['newly_eligible_coarse']==[0,4,5,6,10]
    assert len(ids)==4 and {1,8}<=set(ids) and t['region_budgets']=={'0':2,'1':2}

def test_fine_candidates_cannot_be_newly_added_outside_old_windows():
    a=fixture();a[-1].pop()
    with pytest.raises(ValueError,match='Fine candidate outside'):select_global_coarse(*a)

def test_full_original_coverage_is_equivalent():
    a=fixture();a[2]['regions'][0].update(left_fraction='0',right_fraction='5');a[2]['regions'][1].update(left_fraction='6',right_fraction='11')
    assert select_global_coarse(*a)[0]==select_weighted_seed_support(*a)[0]

def test_single_region_and_cap():
    a=fixture();a[2]['regions']=a[2]['regions'][:1];a[-2].clear();a[-2]['0']=4
    ids,t=select_global_coarse(*a)
    assert len(ids)==4 and t['eligible_indices']==list(range(11))

def test_fractional_vfr_clock_not_average_fps():
    a=fixture();pts=[0,3,9,10,14,16,20,21,25,31,40]
    for i,p in enumerate(pts):a[0][i].update(source_pts=p,source_frame_index=i,timestamp_seconds=p/10)
    a[3].update(time_base='1/10',duration_fraction='41/10')
    a[2]['regions'][0].update(left_fraction='1/5',right_fraction='11/10')
    a[2]['regions'][1].update(left_fraction='12/5',right_fraction='16/5')
    ids,t=select_global_coarse(*a)
    assert len(ids)==4 and len(t['eligible_indices'])==11
    assert t['selection_cells'][0]['right_fraction']=='17/10'
    assert t['selection_cells'][1]['left_fraction']=='9/5'

def test_touching_original_windows_rejected():
    a=fixture();a[2]['regions'][0]['right_fraction']='7'
    with pytest.raises(ValueError):select_global_coarse(*a)

def test_rebudget_uses_unmodified_first_pass_on_new_pool():
    from videoqa_audit.density_reward_decay import select_density_decay
    a=fixture();cells=coarse_selection_cells(a[2],a[3])
    _,expected=select_density_decay(a[0],a[1],cells,a[3],a[4])
    ids,t=select_global_coarse_two_pass(*a)
    assert t['budget_trace']==expected and t['region_budgets']==expected['region_budgets']
    assert len(ids)==4 and {1,8}<=set(ids)

def test_rebudget_matches_exp03_when_original_domain_is_full():
    from videoqa_audit.density_reward_decay import select_density_decay
    a=fixture();a[2]['regions'][0].update(left_fraction='0',right_fraction='5');a[2]['regions'][1].update(left_fraction='6',right_fraction='11')
    _,allocation=select_density_decay(a[0],a[1],a[2],a[3],a[4]);a[-2].update(allocation['region_budgets'])
    assert select_global_coarse_two_pass(*a)[0]==select_weighted_seed_support(*a)[0]
