"""验证事实保留统计使用准确PTS且不自动给出充分性结论。"""
import pytest
from videoqa_audit.evidence_witnesses import map_question


def fixture():
    """构造B/C替换不同帧但同一事实仍存在的16帧组合。"""
    q={'question_id':'q','question':'Q','frames':[{'source_pts':i,'timestamp_seconds':float(i)} for i in range(20)],'groups':{'gB':list(range(16)),'gC':list(range(1,17))}}
    o={'gB':{'method':'B'},'gC':{'method':'C'}}
    s={'reasoning_limits':'test','facts':[{'id':'same','description':'same','union_indices':[1,2]},{'id':'unique','description':'unique','union_indices':[1]}]}
    return q,o,s


def test_alternative_witness_is_retained():
    q,o,s=fixture(); r=map_question(q,o,s)
    assert r['BC_fact_transitions'][0]['state']=='retained'
    assert r['BC_fact_transitions'][1]['state']=='listed_witness_lost'
    assert r['groups']['gC']['sufficiency']=='not_inferred_from_witness_counts'


@pytest.mark.parametrize('bad',[0,21,1.0,True])
def test_reject_invalid_union_index(bad):
    q,o,s=fixture();s['facts'][0]['union_indices']=[bad]
    with pytest.raises(ValueError):map_question(q,o,s)


def test_missing_witness_does_not_imply_absence():
    q,o,s=fixture();s['facts'][0]['union_indices']=[]
    r=map_question(q,o,s)
    assert r['groups']['gB']['facts']['same']['state']=='no_listed_witness_in_input'


def test_reject_duplicate_group_pts():
    q,o,s=fixture();q['groups']['gB']=[0]*16
    with pytest.raises(ValueError):map_question(q,o,s)
