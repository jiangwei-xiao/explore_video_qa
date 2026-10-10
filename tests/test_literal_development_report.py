"""完整人口与空子集回归，不读取模型或生成真实答案。"""
import pytest
from videoqa_audit.literal_development_report import evaluate
from videoqa_audit.literal_development_runtime import METHOD


def test_empty_changes_are_not_reported_as_zero_accuracy():
    """接口：无变化时保留总体，空子集不产生虚构准确率或配对比较。"""
    data = [dict(question_id='q', video_id='v', stratum='short')]
    records = {'q': dict(correct=True)}
    result = evaluate(data, records, {'parent': records}, set(), set())
    assert result['accuracy'][METHOD]['all']['all']['correct'] == 1
    assert result['accuracy'][METHOD]['changed'] == {}
    assert 'changed' not in result['comparisons']['parent']
    assert result['comparisons']['parent']['all']['difference_pp'] == 0


def test_incomplete_reference_cannot_replace_population():
    """接口：不能把有结果或文字命中的题当作完整开发分母。"""
    data = [dict(question_id='q', video_id='v', stratum='short')]
    with pytest.raises(ValueError, match='paired population'):
        evaluate(data, {'q': dict(correct=True)}, {'parent': {}}, set(), set())


def test_gains_and_regressions_are_both_preserved():
    """接口：配对得失全保留，不只输出纠错题。"""
    data = [dict(question_id=str(i), video_id=str(i), stratum='short') for i in range(2)]
    parent = {'0': dict(correct=True), '1': dict(correct=False)}
    new = {'0': dict(correct=False), '1': dict(correct=True)}
    result = evaluate(data, new, {'parent': parent}, {'0', '1'}, {'0', '1'})
    pair = result['comparisons']['parent']['all']
    assert pair['cells']['left_only'] == ['1']
    assert pair['cells']['right_only'] == ['0']
    assert pair['difference_pp'] == 0
