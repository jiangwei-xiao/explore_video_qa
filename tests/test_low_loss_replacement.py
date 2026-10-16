"""exp34删除损失：检验冻结提议与覆盖性质，使用无模型小矩阵。"""
from fractions import Fraction
import copy
import numpy as np
import pytest

from videoqa_methods.adjacent_context_repair import select_adjacent_repair
from videoqa_methods.low_loss_replacement import select_low_loss_replacement


def fixture():
    """接口：粗帧时间有序，细帧混在数组末尾；b是独有状态，另一帧可被代表。"""
    points = [25, 225, 425, 625, 75, 275, 475]
    frames = [dict(source_pts=p, source_frame_index=p, candidate_index=i,
                   requested_fraction=str(Fraction(p, 100)), timestamp_seconds=p / 100)
              for i, p in enumerate(points)]
    pool = dict(candidates=frames, initial_count=4,
                initial_scores=[.9, .8, .7, .6],
                plan=dict(seed_indices=[0, 3], regions=[dict(anchor_index=0)]))
    features = np.array([[1,0,0], [0,1,0], [1,0,0], [0,0,1],
                         [1,0,0], [0,1,0], [1,0,0]], dtype=np.float32)
    return pool, features, [0, 1, 2, 3]


def run(pool, features, selected):
    _, exp33 = select_adjacent_repair(pool, features, selected, len(selected))
    return select_low_loss_replacement(pool, features, selected, exp33, len(selected))


def cover(features, coarse, indices):
    sim = np.clip(features[:coarse] @ features[indices].T, 0, 1)
    return np.max(sim, axis=1).mean(dtype=np.float64)


def test_only_donor_changes_and_coverage_is_not_below_exp33():
    pool, features, selected = fixture()
    untouched = copy.deepcopy(pool)
    output, trace = run(pool, features, selected)
    assert output == [0, 4, 1, 3]
    assert trace['added_index'] == 4
    assert trace['previous_donor_index'] == 1
    assert trace['chosen_donor_index'] == 2
    assert trace['donor_changed']
    assert trace['coverage_final'] == pytest.approx(cover(features, 4, output))
    assert trace['coverage_final'] >= trace['coverage_exp33']
    assert pool == untouched and selected == [0, 1, 2, 3]


def test_equal_loss_prefers_exp33_original_donor():
    pool, features, selected = fixture()
    features[2] = [0, 1, 0]
    output, trace = run(pool, features, selected)
    assert trace['chosen_donor_index'] == trace['previous_donor_index'] == 1
    assert output == trace['exp33_indices']


def test_all_losses_match_direct_enumeration():
    pool, features, selected = fixture()
    _, trace = run(pool, features, selected)
    augmented = selected + [trace['added_index']]
    for entry in trace['donor_losses']:
        other = [x for x in augmented if x != entry['index']]
        expected = cover(features, 4, augmented) - cover(features, 4, other)
        assert entry['coverage_loss'] == pytest.approx(expected, abs=1e-7)


def test_no_exp33_proposal_keeps_parent():
    pool, features, _ = fixture()
    selected = sorted(range(len(pool['candidates'])),
                      key=lambda i: pool['candidates'][i]['source_pts'])
    output, trace = run(pool, features, selected)
    assert output == selected and not trace['changed']


def test_frozen_proposal_identity_rejects_mutation():
    pool, features, selected = fixture()
    _, old = select_adjacent_repair(pool, features, selected, 4)
    old['winner']['added_index'] = 5
    with pytest.raises(ValueError, match='proposal differs'):
        select_low_loss_replacement(pool, features, selected, old, 4)


def test_numeric_tie_keeps_old_donor_but_larger_loss_switches():
    pool, features, selected = fixture()
    _, trace = run(pool, features, selected)
    assert trace['chosen_donor_loss'] + trace['tie_tolerance'] >= trace['minimum_coverage_loss']
    assert trace['original_donor_loss'] > trace['chosen_donor_loss'] + trace['tie_tolerance']


def test_no_answer_or_question_interface():
    import inspect
    assert list(inspect.signature(select_low_loss_replacement).parameters) == [
        'pool', 'features', 'selected_indices', 'exp33_trace', 'K']
