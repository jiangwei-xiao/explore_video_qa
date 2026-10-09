"""实际分量诊断用例，不为原选择器增加参数。"""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('text_utility_audit', Path(__file__).resolve().parents[1]/'scripts/audit_text_state_utilities.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_nonempty_false_character_can_remove_visual_penalty():
    f = np.array([[1,0], [1,0]], dtype=np.float32)
    result = m.utility_parts(f,[['7'],[]],0,[1],0.8)
    assert result['visual_redundancy'] == 1
    assert result['text_redundancy'] == 0
    assert result['visual_utility'] == 0
    assert result['text_utility'] == 0.8


def test_no_detected_text_keeps_original_penalty():
    f = np.array([[1,0], [1,0]], dtype=np.float32)
    result = m.utility_parts(f,[[],['7']],0,[1],0.8)
    assert result['visual_redundancy'] == result['text_redundancy'] == 1


def test_multiply_then_max_is_not_product_of_separate_maxima():
    f = np.array([[1,0], [1,0], [0.5,np.sqrt(0.75)]], dtype=np.float32)
    result = m.utility_parts(f,[['a'],['b'],['a']],0,[1,2],1)
    assert result['visual_redundancy'] == 1
    assert result['text_redundancy'] == 0.5


def test_empty_reference_is_not_silently_a_free_slot():
    with pytest.raises(ValueError):
        m.utility_parts(np.array([[1,0]]),[['a']],0,[],1)


def test_soft_companion_keeps_visual_lower_bound_and_is_symmetric():
    f = np.array([[1,0], [0.9,np.sqrt(0.19)]], dtype=np.float32)
    for states in ([[],[]], [['a'],[]], [['a'],['b']], [['a'],['a']]):
        value = m.companion_similarity(f,states,0,1)
        assert value == m.companion_similarity(f,states,1,0)
        assert value >= float(f[0]@f[1])/2
    assert m.companion_similarity(f,[[],[]],0,1) == float(f[0]@f[1])
