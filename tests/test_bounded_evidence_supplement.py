"""有界补充的原输入保护、四路对称性和精确删除成本验收。"""
import numpy as np
from videoqa_methods.bounded_evidence_supplement import supplement, coverage_and_removal_cost


def fixture():
    candidates = [dict(candidate_index=i, source_frame_index=i, source_pts=4*i+1,
                       timestamp_seconds=i+.25, requested_seconds=i+.25) for i in range(40)]
    f = np.random.default_rng(2027).random((40, 12)).astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    pool = dict(candidates=candidates, video=dict(start_pts=0, time_base='1/4', duration_fraction='41'),
                initial_count=40, initial_scores=[.3]*40)
    trace = dict(protected_indices=[0], regions=[dict(region_id=0, members=list(range(40)),
                      reference_coarse_indices=list(range(40)), density_weight=2)])
    return pool, f, list(range(32)), trace


def test_no_positive_increment_retains_original():
    pool, f, old, trace = fixture()
    selected, changes = supplement(pool, f, old, trace, np.full((4, 40), .3))
    assert selected == old and changes['steps'] == []


def test_budget_anchors_swap_cap_and_option_permutation():
    pool, f, old, trace = fixture()
    scores = np.random.default_rng(11).random((4, 40))
    selected, changes = supplement(pool, f, old, trace, scores)
    permuted, _ = supplement(pool, f, old, trace, scores[[3, 1, 0, 2]])
    assert selected == permuted and len(selected) == len(set(selected)) == 32
    assert set(trace['protected_indices']) <= set(selected)
    assert len(changes['added_indices']) == len(changes['removed_indices']) == 4
    assert not set(changes['added_indices']).intersection(old)


def test_removal_cost_matches_brute_force():
    _, f, old, trace = fixture()
    value, costs = coverage_and_removal_cost(f, trace['regions'], set(old))
    for i in old:
        reduced, _ = coverage_and_removal_cost(f, trace['regions'], set(old)-{i})
        assert abs(value-reduced-costs[i]) < 1e-6


def test_all_original_frames_protected_causes_fallback():
    pool, f, old, trace = fixture()
    trace['protected_indices'] = old
    selected, changes = supplement(pool, f, old, trace, np.full((4, 40), .9))
    assert selected == old and changes['steps'] == []
