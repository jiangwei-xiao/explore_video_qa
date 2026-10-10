"""完整候选评分闭合、无信号回退与预算/锚点边界的CPU回归。"""
import numpy as np
import pytest
from videoqa_methods.option_evidence_selection import select_option_evidence


def case():
    video = dict(time_base='1/1', start_pts=0, duration_seconds=4., duration_fraction='4')
    candidates = [dict(candidate_index=i, source_pts=i, source_frame_index=i,
        timestamp_seconds=float(i), requested_seconds=float(i)) for i in range(4)]
    features = np.asarray([[1, 0], [.8, .6], [0, 1], [.6, .8]], dtype=np.float32)
    pool = dict(video=video, candidates=candidates, initial_scores=[.9, .8, .2, .4],
        plan=dict(regions=[dict(region_id=0, anchor_index=0, seed_indices=[0])]))
    parent = dict(regions=[dict(region_id=0, members=[0, 1, 2, 3])], region_budgets={'0': 2})
    return pool, features, parent


def test_zero_evidence_exactly_replays_original_local_rule():
    pool, features, parent = case()
    zero, trace = select_option_evidence(pool, features, parent, {1: 0, 2: 0, 3: 0}, 2)
    one, _ = select_option_evidence(pool, features, parent, {1: 1, 2: 1, 3: 1}, 2)
    assert zero == one and trace['regions'][0]['no_signal_fallback']


def test_signal_can_change_only_supplement_without_removing_anchor():
    pool, features, parent = case()
    selected, trace = select_option_evidence(pool, features, parent, {1: 1, 2: 0, 3: 0}, 2)
    assert selected == [0, 1] and not trace['regions'][0]['no_signal_fallback']
    assert trace['region_budgets'] == parent['region_budgets']


def test_missing_legal_score_and_nonfinite_score_pause():
    pool, features, parent = case()
    with pytest.raises(ValueError, match='Unscored legal'):
        select_option_evidence(pool, features, parent, {1: 1}, 2)
    with pytest.raises(ValueError, match='Invalid evidence'):
        select_option_evidence(pool, features, parent, {1: 1, 2: 0, 3: float('nan')}, 2)


def test_single_anchor_region_does_not_require_unusable_scores():
    pool, features, parent = case(); parent['region_budgets']['0'] = 1
    selected, _ = select_option_evidence(pool, features, parent, {}, 1)
    assert selected == [0]
