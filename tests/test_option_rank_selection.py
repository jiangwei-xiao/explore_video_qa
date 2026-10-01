"""验证对称检索名次融合，不借助问答正确性选择规则。"""
import itertools
import numpy as np
import pytest
from videoqa_audit.option_rank_selection import select_option_ranks


def candidates(n):
    return [dict(candidate_index=i, source_pts=10*i) for i in range(n)]


def test_identical_routes_reduce_to_top32():
    c = candidates(50); s = np.linspace(1,0,50).tolist()
    selected, trace = select_option_ranks(c, [s]*4)
    assert selected == list(range(32))
    assert trace['actual_frames'] == 32


def test_route_permutation_does_not_change_selection():
    c = candidates(64)
    scores = np.random.default_rng(2027).random((4,64)).tolist()
    expected, _ = select_option_ranks(c, scores)
    for perm in itertools.permutations(range(4)):
        actual, _ = select_option_ranks(c, [scores[i] for i in perm])
        assert actual == expected


def test_contains_all_top8_routes_and_fills_unique_budget():
    c = candidates(100); scores = np.random.default_rng(3).random((4,100)).tolist()
    selected, trace = select_option_ranks(c, scores)
    union = set(i for order in trace['per_route_orders'] for i in order[:8])
    assert union <= set(selected)
    assert len(selected) == len(set(selected)) == 32


def test_short_pool_no_duplicates():
    c = candidates(11); scores = [[0.4]*11]*4
    selected, _ = select_option_ranks(c, scores)
    assert selected == list(range(11))


def test_stable_ties_use_source_time_then_index():
    c = [dict(source_pts=30,candidate_index=2),dict(source_pts=10,candidate_index=1),dict(source_pts=20,candidate_index=0)]
    selected, trace = select_option_ranks(c, [[0.5]*3]*4, 2)
    assert selected == [1,2] and trace['fused_order'] == [1,2,0]


@pytest.mark.parametrize('scores,budget', [([[0.5]]*3,32), ([[float('nan')]]*4,32), ([[1.1]]*4,32), ([[0.5]]*4,0)])
def test_invalid_input_stops(scores,budget):
    with pytest.raises((ValueError,RuntimeError)):
        select_option_ranks(candidates(1),scores,budget)
