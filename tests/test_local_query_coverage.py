"""局部描述覆盖的同分、回退、预算、区域约束与选项置换测试。"""
import numpy as np
from videoqa_methods.density_selection import build_regions
from videoqa_methods.option_seed_entry import select_option_pool
from videoqa_methods.local_query_coverage import strict_rank_counts,select_local_query_coverage


def fixture():
    candidates=[dict(candidate_index=i,source_frame_index=i,source_pts=4*i+1,timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(15)]
    video=dict(start_pts=0,time_base='1/4',duration_fraction='16')
    cfg=dict(seed_budget=3,frame_budget=7,core_gap_seconds='2',window_radius_seconds='1',refinement_min_core_seeds=3)
    original=[.1]*15
    for i in [1,7,13]:original[i]=.9
    features=np.random.default_rng(2027).random((15,8)).astype(np.float32)
    features/=np.linalg.norm(features,axis=1,keepdims=True)
    pool=dict(video=video,candidates=candidates,initial_count=15,initial_scores=original,plan=build_regions(candidates,original,video,cfg))
    selected,trace=select_option_pool(pool,features,cfg)
    return pool,features,cfg,selected,trace


def test_ties_do_not_create_arbitrary_query_preference():
    assert strict_rank_counts([[.2,.2,.7,.1]]).tolist()==[[1,1,3,0]]
    assert strict_rank_counts([[.2]*4]).tolist()==[[0]*4]


def test_constant_queries_exactly_replay_original():
    pool,f,cfg,old,trace=fixture()
    selected,new=select_local_query_coverage(pool,f,cfg,trace['region_budgets'],np.full((4,15),.2))
    assert selected==old and new['region_budgets']==trace['region_budgets']
    assert new['objective']==trace['objective']
    assert all(r['query_steps']==0 for r in new['regions'])


def test_permutation_budget_and_window_constraints():
    pool,f,cfg,old,trace=fixture()
    scores=np.random.default_rng(42).random((4,15))
    selected,new=select_local_query_coverage(pool,f,cfg,trace['region_budgets'],scores)
    permuted,other=select_local_query_coverage(pool,f,cfg,trace['region_budgets'],scores[[3,1,0,2]])
    assert selected==permuted and new['region_budgets']==other['region_budgets']
    assert len(selected)==len(set(selected))==7
    assert set(trace['protected_indices'])<=set(selected)
    for region in new['regions']:
        assert region['query_steps']<=4
        assert all(s['candidate_index'] in region['original_window_indices'] for s in region['local_steps'] if s['mode']=='description_coverage')


def test_disabled_query_module_replays_original_without_scores():
    pool,f,cfg,old,trace=fixture()
    selected,new=select_local_query_coverage(pool,f,cfg,trace['region_budgets'],None,enabled=False)
    assert selected==old and not new['local_query_coverage_enabled']
