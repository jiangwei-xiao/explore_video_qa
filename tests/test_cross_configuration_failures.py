"""共同失败分类须保留分母，基线互补不等于证据因果解释。"""
import pytest
from videoqa_audit.cross_configuration_failures import PROFILES,summarize_failures


def test_common_failure_and_category_denominator():
    data=[dict(question_id=str(i),video_id='v'+str(i),stratum='short',task_type='count' if i<2 else 'order',domain='d') for i in range(3)]
    records={str(i):dict(conditions={p:dict(correct=i==2) for p in PROFILES},
             baselines={p:{'uniform':dict(correct=i==1),'topk':dict(correct=False)} for p in PROFILES}) for i in range(3)}
    result=summarize_failures(data,records)
    assert result['common_wrong_ids']==['0','1'] and result['baseline_can_answer_ids']==['1']
    assert result['all_eight_baselines_wrong_ids']==['0']
    assert result['categories']['task_type']['count']['total']==2
    assert result['categories']['task_type']['count']['common_wrong_rate']==1
    assert result['categories']['task_type']['order']['common_wrong_rate']==0


def test_missing_question_refuses_partial_claim():
    with pytest.raises(ValueError,match='paired'):summarize_failures([dict(question_id='q')],{})
