"""五类概率、格式质量与最小任务输入测试，不加载模型或读取标准答案。"""
import numpy as np
import pytest
from videoqa_methods.option_evidence_support import normalized_js, evidence_support, reference_question_text


def test_js_identity_symmetry_and_bit_bound():
    a = [1, 0, 0, 0, 0]; b = [0, 1, 0, 0, 0]
    assert normalized_js(a, a) == 0
    assert normalized_js(a, b) == normalized_js(b, a) == pytest.approx(1)


def test_other_quality_is_not_renormalized_away():
    weak = [0.001, 0, 0, 0, 0.999]; baseline = [0, 1, 0, 0, 0]
    value = evidence_support(weak, baseline)
    assert value['letter_mass'] == .001
    assert 0 <= value['value'] <= .001
    assert evidence_support([0, 0, 0, 0, 1], baseline)['value'] == 0


def test_correcting_prior_can_change_choice_without_entropy_threshold():
    baseline = [.99, .01, 0, 0, 0]; shifted = [.05, .90, 0, 0, .05]
    assert evidence_support(shifted, baseline)['value'] > .5


@pytest.mark.parametrize('value', [[.1, .1, .1, .1], [-.1, .3, .3, .3, .2], [0, 0, 0, 0, 0], [float('nan'), 0, 0, 0, 1]])
def test_invalid_distribution_pauses(value):
    with pytest.raises(ValueError):
        normalized_js(value, [0, 0, 0, 0, 1])


def test_reference_prompt_only_receives_public_question_and_original_options(monkeypatch):
    import videoqa_lmms_bridge.common as common
    class Task:
        def videomme_doc_to_text(self, document, kwargs):
            assert set(document) == {'question', 'options'}
            assert document['options'] == ['A. One', 'B. Two', 'C. Three', 'D. Four']
            return document['question'] + kwargs['post_prompt']
    monkeypatch.setattr(common, 'load_task', lambda: Task())
    assert reference_question_text('Original?', ['A. One', 'B. Two', 'C. Three', 'D. Four']).startswith('Original?')
    with pytest.raises(TypeError):
        reference_question_text('Original?', ['A. One', 'B. Two', 'C. Three', 'D. Four'], answer='B')
    with pytest.raises(ValueError):
        reference_question_text('Original?', ['B. Two', 'A. One', 'C. Three', 'D. Four'])
