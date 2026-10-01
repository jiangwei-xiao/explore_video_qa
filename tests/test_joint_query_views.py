"""五路融合的固定预算与原题视角保留测试，不依赖答案。"""
import numpy as np
from videoqa_audit.joint_query_views import select_joint_views


def candidates(n):return [dict(candidate_index=i,source_pts=i*4+1) for i in range(n)]


def test_identical_views_reduce_to_original_top32():
    s=np.linspace(1,0,60).tolist();ids,_=select_joint_views(candidates(60),s,[s]*4)
    assert ids==list(range(32))


def test_each_view_top6_is_preserved_at_budget32():
    scores=np.random.default_rng(2027).random((5,100)).tolist();ids,trace=select_joint_views(candidates(100),scores[0],scores[1:])
    assert len(ids)==len(set(ids))==32
    assert {i for order in trace['per_view_orders'] for i in order[:6]}<=set(ids)


def test_short_pool_keeps_all_unique_frames():
    ids,_=select_joint_views(candidates(11),[.3]*11,[[.3]*11]*4)
    assert ids==list(range(11))


def test_joint_union_not_forced_superset_of_old_four_views():
    c=candidates(40);options=[[1-i/40 for i in range(40)]]*4;question=list(reversed(options[0]))
    ids,_=select_joint_views(c,question,options)
    assert 39 in ids and len(ids)==32
