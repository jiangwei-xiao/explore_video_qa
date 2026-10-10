"""完整选项词组、重复/顺序歧义以及无关文字不获得新名奖励的CPU测试。"""
import numpy as np
import pytest
from videoqa_methods.literal_option_coverage import option_word_groups, literal_matches, select_literal_coverage


def test_complete_option_words_required_without_partial_or_number_reward():
    groups = option_word_groups('Which communication method?', ['A. Telephone.', 'B. Homing pigeon.', 'C. Telegraph.', 'D. Pony express.'])
    matches = literal_matches([['the', 'pony', 'express'], ['pony'], ['00', '41'], ['telephone']], groups)
    assert matches == [{3}, set(), set(), {0}]


def test_duplicate_and_reordered_option_content_is_one_witness_not_order_answer():
    groups = option_word_groups('What order?', ['A. Horse then boat.', 'B. Boat then horse.', 'C. Horse then boat.', 'D. Six.'])
    assert len(groups) == 2
    assert option_word_groups('How many?', ['A. 1', 'B. 2', 'C. 3', 'D. 4']) == []


def test_question_already_contains_group_and_wrong_option_order_are_rejected():
    groups = option_word_groups('Which B7 chord?', ['A. B7', 'B. C7', 'C. D7', 'D. E7'])
    assert ('b7',) not in groups
    with pytest.raises(ValueError):
        option_word_groups('Original?', ['B. B7', 'A. C7', 'C. D7', 'D. E7'])


def test_named_state_gains_once_and_preserves_anchor_and_quota():
    candidates = [dict(candidate_index=i, source_frame_index=i, source_pts=i,
        timestamp_seconds=float(i), requested_seconds=float(i)) for i in range(4)]
    pool = dict(video=dict(start_pts=0, time_base='1/1', duration_fraction='4', duration_seconds=4.),
        candidates=candidates, initial_scores=[.9, .9, .9, .9],
        plan=dict(regions=[dict(region_id=0, anchor_index=0, seed_indices=[0])]))
    parent = dict(regions=[dict(region_id=0, members=[0, 1, 2, 3])], region_budgets={'0': 3})
    f = np.asarray([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=np.float32)
    selected, trace = select_literal_coverage(pool, f, parent, [set(), {0}, {0}, {1}], 3)
    assert selected == [0, 1, 3]
    assert trace['regions'][0]['final_covered_groups'] == [0, 1]
