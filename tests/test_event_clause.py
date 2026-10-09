"""事件子句原文约束，不给定位器注入答案或改写后的事实。"""
import pytest
from videoqa_audit.event_clause import event_clause


@pytest.mark.parametrize('question,event',[
    ('What time does the hero in the video get up in real life?','get up in real life'),
    ('What time does the male protagonist in the video go out to relax?','go out to relax'),
    ('What time did the heroine in the video turn off her alarm clock?','turn off her alarm clock'),
    ("In the video, at what time do classes begin at the boys' school?","classes begin at the boys' school"),
    ('At what time does the protagonist in the video typically visit the gym for their workout?','typically visit the gym for their workout'),
])
def test_extracts_exact_source_with_temporal_and_relation_words(question,event):
    result=event_clause(question)
    assert result['event_query']==event
    a,b=result['source_span']; assert question[a:b]==event


def test_named_actor_is_not_deleted_or_new_fact_inferred():
    result=event_clause('What time does Alice leave after the first bell?')
    assert result['event_query']=='Alice leave after the first bell'
    assert not result['actor_information_ablation']


def test_unsupported_question_stays_original():
    question='How many athletes cross the finish line?'
    assert event_clause(question)['event_query']==question


def test_nonempty_original_required():
    with pytest.raises(ValueError):event_clause('')
