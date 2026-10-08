"""条件合并、唯一归属和预算边界：合成测试不调用模型。"""
from fractions import Fraction
import numpy as np
from test_region_refinement_scope import make_pool
from videoqa_methods.region_refinement_scope import restrict_refinement_pool
from videoqa_methods.conditional_window_merge import conditional_groups,core_ownership,select_conditional_merge
from videoqa_methods.budget_contract import select_exp07


def test_dense_sparse_overlap_is_split_and_does_not_duplicate_source():
    p,f=make_pool();new,nf,_=restrict_refinement_pool(p,f,4)
    groups=conditional_groups(p['plan'],4)
    assert len(p['plan']['regions'])==1 and len(groups)==2
    selected,trace=select_conditional_merge(new,nf,4)
    assert len(selected)==len(set(selected))==4
    assert sorted(i for r in trace['regions'] for i in r['members'])==list(range(len(nf)))
    assert len(trace['protected_indices'])==2 and sum(trace['region_budgets'].values())==4


def test_two_overlapping_dense_cores_merge_and_match_exp07():
    p,f=make_pool(seed_times=(0.25,1.25,2.25,10.25,11.25,12.25),fine=())
    new,nf,_=restrict_refinement_pool(p,f,6)
    assert len(conditional_groups(p['plan'],6))==1
    assert select_conditional_merge(new,nf,6)[0]==select_exp07(new,nf,6)[0]


def test_sparse_cores_never_merge_even_if_windows_overlap():
    p,f=make_pool(seed_times=(2.25,3.25,10.25,11.25),fine=())
    assert len(p['plan']['regions'])==1 and len(conditional_groups(p['plan'],4))==2
    selected,trace=select_conditional_merge(p,f,4)
    assert len(selected)==4 and all(r['refine'] is False for r in trace['regions'])


def test_sparse_core_does_not_bridge_two_dense_scopes():
    cores=[dict(core_id=0,seed_count=3,seed_indices=[0,1,2],left_fraction='0',right_fraction='6'),
           dict(core_id=1,seed_count=1,seed_indices=[3],left_fraction='5',right_fraction='12'),
           dict(core_id=2,seed_count=3,seed_indices=[4,5,6],left_fraction='11',right_fraction='18')]
    assert [g['core_ids'] for g in conditional_groups(dict(cores=cores),7)]==[[0],[1],[2]]


def test_touching_dense_scopes_merge_without_absorbing_interleaved_sparse_core():
    cores=[dict(core_id=0,seed_count=3,seed_indices=[0,1,2],left_fraction='0',right_fraction='10'),
           dict(core_id=1,seed_count=1,seed_indices=[3],left_fraction='7',right_fraction='15'),
           dict(core_id=2,seed_count=3,seed_indices=[4,5,6],left_fraction='10',right_fraction='20')]
    assert [g['core_ids'] for g in conditional_groups(dict(cores=cores),7)]==[[0,2],[1]]


def test_source_time_ownership_not_average_fps_and_tie_goes_earlier():
    rows=[dict(source_pts=1),dict(source_pts=9),dict(source_pts=5)]
    plan=dict(cores=[dict(core_id=0,seed_indices=[0]),dict(core_id=1,seed_indices=[1])])
    assert core_ownership(rows,plan)==[0,1,0]


def test_short_pool_fills_available_unique_sources_and_no_fine_priority():
    p,f=make_pool(seed_times=tuple(0.25+i for i in range(19)),fine=(0.75,))
    new,nf,_=restrict_refinement_pool(p,f,32);selected,trace=select_conditional_merge(new,nf,32)
    assert len(selected)==20 and len(set(selected))==20
    assert trace['selection_group_count']==1


def test_identical_features_zero_gain_fill_and_stable_time_order():
    p,f=make_pool();new,nf,_=restrict_refinement_pool(p,f,4)
    nf=np.ones((len(nf),1),dtype=np.float32)
    a,trace=select_conditional_merge(new,nf,4);b,_=select_conditional_merge(new,nf,4)
    assert a==b and len(a)==4 and sum(trace['region_budgets'].values())==4
    assert all(x['local_score']==0 for r in trace['regions'] for x in r['local_steps'])
