"""事件输入契约测试，无模型或答案依赖。"""
import pytest
from videoqa_audit.event_query_contract import checked_event_clause


@pytest.mark.parametrize('prefix', ['In the last model gathering, ', 'At the end of the video, ', 'Before sunrise, '])
def test_nontrivial_leading_condition_falls_back(prefix):
    question = prefix + 'at what time does the green model appear in the video?'
    result = checked_event_clause(question)
    assert result['event_query'] == question and not result['supported']
    assert result['guard_reason'] == 'nonneutral_leading_condition'
    assert result['source_span'] == [0, len(question)]


@pytest.mark.parametrize('prefix', ['', 'In the video, ', 'According to the video, '])
def test_neutral_prefix_does_not_change_prior_event_expression(prefix):
    question = prefix + 'what time does the hero in the video get up in real life?'
    result = checked_event_clause(question)
    assert result['event_query'] == 'get up in real life'
    assert result['supported'] and result['actor_information_ablation']


def test_named_actor_and_tail_qualifier_stay_in_source():
    result = checked_event_clause('At what time does Alice first visit the gym after school?')
    assert result['event_query'] == 'Alice first visit the gym after school'
    assert not result['actor_information_ablation']


def test_unsupported_question_keeps_whole_original():
    question = 'How long should one wait before cracking the knuckle?'
    result = checked_event_clause(question)
    assert result['event_query'] == question and not result['supported']


@pytest.mark.parametrize('question', ['', '   '])
def test_blank_question_is_invalid(question):
    with pytest.raises(ValueError): checked_event_clause(question)
