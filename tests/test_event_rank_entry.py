"""双排序入口的同池、单调性、同分和回退验收，不运行模型。"""
import math
import pytest
from videoqa_audit.event_rank_entry import joint_rank_scores


def candidates(n):
    return [dict(candidate_index=i, source_pts=i * 30) for i in range(n)]


def test_same_query_preserves_original_stable_order():
    c = candidates(4); values = [.6, .9, .9, .2]
    combined, trace = joint_rank_scores(c, values, values)
    assert sorted(range(4), key=lambda i: -combined[i]) == [1, 2, 0, 3]
    assert trace['question_ranks'] == trace['event_ranks']


def test_absolute_query_scale_does_not_change_joint_entry():
    c = candidates(4)
    a, _ = joint_rank_scores(c, [.1, .2, .3, .4], [.8, .6, .4, .2])
    b, _ = joint_rank_scores(c, [.01, .02, .03, .04], [.4, .3, .2, .1])
    assert a == b


def test_symmetric_queries_and_log_cost_have_same_order():
    c = candidates(5); q = [.9, .8, .7, .6, .5]; e = [.5, .6, .7, .8, .9]
    scores, trace = joint_rank_scores(c, q, e)
    reverse, _ = joint_rank_scores(c, e, q)
    assert scores == reverse
    assert [math.log(a) + math.log(b) for a, b in zip(trace['question_ranks'], trace['event_ranks'])] == pytest.approx(
        [math.log(p) for p in trace['rank_products']])


@pytest.mark.parametrize('q,e', [([], []), ([.2], [.3, .4]), ([float('nan')], [.2]), ([1.1], [.2])])
def test_invalid_population_or_numeric_values_stop(q, e):
    with pytest.raises(ValueError): joint_rank_scores(candidates(len(q)), q, e)
