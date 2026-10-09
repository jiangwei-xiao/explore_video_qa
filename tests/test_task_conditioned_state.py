"""问题条件状态的零向量、集合重复度及固定竞争上下文回归。"""
from pathlib import Path
import sys
import numpy as np
import pytest

from videoqa_methods.task_conditioned_state import image_residual, conditional_redundancy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_task_conditioned_state import evaluation_pairs, verify_completed, feature_call
from videoqa_full.state import durable
from videoqa_runtime.common import sha256


def test_visual_residual_uses_same_baseline_without_epsilon_noise():
    vector, valid, norm = image_residual([3, 5], [3, 1])
    assert valid and norm == 4 and np.array_equal(vector, [0, 1])
    vector, valid, norm = image_residual([3, 5], [3, 5])
    assert not valid and norm == 0 and np.array_equal(vector, [0, 0])


def test_nonfinite_or_mismatched_states_pause():
    with pytest.raises(ValueError):
        image_residual([float('nan')], [1])
    with pytest.raises(ValueError):
        image_residual([1, 2], [1])


def test_valid_task_similarity_replaces_only_redundancy():
    old = np.array([1., 0.]); context = np.array([[1., 0.]])
    assert conditional_redundancy(old, context, [0, 1], [[1, 0]], True, [True]) == 0
    assert conditional_redundancy(old, context, [0, 1], [[1, 0]], False, [True]) == 1
    assert conditional_redundancy(old, context, [0, 1], [[1, 0]], True, [False]) == 1


def test_cumulative_max_and_negative_clipping_match_parent_semantics():
    result = conditional_redundancy([1, 0], [[1, 0], [1, 0]],
        [1, 0], [[-1, 0], [.8, .6]], True, [True, True])
    assert result == pytest.approx(.8)
    with pytest.raises(ValueError):
        conditional_redundancy([1, 0], [[1, 0]], [1, 0], [[1, 0]], True, [True, True])


def test_protection_uses_original_insertion_context_and_fixed_challenger():
    pool = dict(candidates=[dict(timestamp_seconds=i) for i in range(30)])
    parent = dict(regions=[dict(region_id=0, members=list(range(30)), prefix=[0, 27, 2, 3])],
        region_budgets={'0': 3})
    failed = dict(regions=[dict(region_id=0, prefix=[0, 1, 3, 2])])
    pair = evaluation_pairs(pool, parent, failed, '007-2')[0]
    assert pair['selected_before'] == [0] and pair['winner'] == 1
    assert pair['role'] == 'protected_state' and pair['parent_selected']


def test_missing_witness_uses_actual_parent_final_slot_not_favorable_context():
    pool = dict(candidates=[dict(timestamp_seconds=i) for i in range(30)])
    parent = dict(regions=[dict(region_id=0, members=list(range(30)), prefix=[0, 1, 2, 3])],
        region_budgets={'0': 3})
    failed = dict(regions=[dict(region_id=0, prefix=[0, 1, 3, 2])])
    pair = evaluation_pairs(pool, parent, failed, '007-2')[0]
    assert pair['selected_before'] == [0, 1] and pair['winner'] == 2
    assert pair['role'] == 'missing_state_opportunity' and not pair['parent_selected']


def test_completed_restore_verifies_bytes_without_encoder_or_forward(tmp_path):
    raw = tmp_path / 'return.json'; durable(raw, {'state': [1., 2.]})
    durable(tmp_path / 'summary.json', {'artifacts': {str(raw): sha256(raw)}})
    assert verify_completed(tmp_path)['artifacts'][str(raw)] == sha256(raw)
    assert __import__('json').loads((tmp_path / 'completed_restore_audit.json').read_text())['new_model_forwards'] == 0
    durable(raw, {'state': [1., 3.]})
    with pytest.raises(ValueError, match='artifact changed'):
        verify_completed(tmp_path)


def test_uncertain_prefill_cannot_be_repeated_as_another_method(tmp_path):
    from PIL import Image
    durable(tmp_path / 'calls/real/a.json', {'status': 'started', 'question': 'Original?', 'category': 'real_frame'})
    with Image.new('RGB', (28, 28)) as image:
        with pytest.raises(ValueError, match='automatic retry forbidden'):
            feature_call(tmp_path, 'real/a.json', image, 'Original?', None, 'real_frame')
