"""事实观察诊断边界：源组构建、严格JSON、评价前冻结和未知返回恢复。"""
import copy
import inspect
from pathlib import Path

import pytest

from videoqa_audit.evidence_observation_backend import observation_text,parse_observations
from videoqa_audit import evidence_observation_probe as probe
from videoqa_full.state import durable


def example():
    """接口：可见事实的最小有效结构，不读取真实问题、答案或图像。"""
    return '{"observations":[{"frame_ids":[3],"text":"A gray strip is visible."}],"relations":[],"unverified_constraints":[]}'


def test_single_frame_ids_preserved_not_renumbered():
    assert parse_observations(example(),[3])['valid']
    assert not parse_observations(example(),[1])['valid']


@pytest.mark.parametrize('raw',[
    '```json\n'+example()+'\n```',
    example()+' explanation',
    example().replace('[3]','[3,3]'),
    example().replace('[3]','[true]'),
    example().replace('"text":"A gray strip is visible."','"text":""'),
    example().replace('"unverified_constraints":[]','"unverified_constraints":"unknown"'),
    example().replace('"relations":[]','"relations":[{"frame_ids":[3],"text":"x"},{"frame_ids":[3],"text":"y"}]')
])
def test_invalid_json_references_and_shapes_are_retained_as_failure(raw):
    assert not parse_observations(raw,[3])['valid']


def test_truncated_valid_looking_json_is_still_a_failure():
    assert not parse_observations(example(),[3],True)['valid']


def test_empty_json_has_no_implicit_fact_or_answer():
    value=parse_observations('{"observations":[],"relations":[],"unverified_constraints":[]}',[1])
    assert value['valid'] and not value['value']['observations']


def test_question_interface_has_no_option_or_answer_argument():
    assert list(inspect.signature(observation_text).parameters)==['question','frame_ids','times']
    prompt=observation_text('What is visible?',[3],[12.25])
    assert 'Frame ID 3: source time 12.250000000' in prompt
    with pytest.raises(ValueError):observation_text('q',[1,2],[2.,1.])


def test_groups_use_true_pts_and_parent_coarse_score_not_array_order():
    points=[0,10,20,30,40,50,60,15,25]
    cs=[dict(source_pts=p,source_frame_index=p) for p in points]
    pool=dict(candidates=cs,initial_count=7,initial_scores=[.3,.4,.9,.8,.7,.6,.5])
    parent=dict(selected_frames=[cs[i] for i in range(7)])
    old=copy.deepcopy(pool)
    result=probe.choose_groups(pool,parent,[15,25])
    assert result['witness']==[1,7,2,8]
    assert result['comparison']==[3,4,5,6]
    assert pool==old


def test_missing_witness_does_not_approximate_a_nearby_frame():
    pool=dict(candidates=[dict(source_pts=i,source_frame_index=i) for i in range(8)],
              initial_count=8,initial_scores=[.5]*8)
    with pytest.raises(ValueError,match='Exact frozen witness'):
        probe.choose_groups(pool,dict(selected_frames=pool['candidates']),[99])


def test_uncertain_return_never_reissues_generation(tmp_path):
    durable(tmp_path/'protocol.json',{})
    durable(tmp_path/'calls/facts/job.json',dict(status='started_may_generate',gpu='7',
        protocol_sha256=probe.sha256(tmp_path/'protocol.json')))
    with pytest.raises(RuntimeError,match='no retry'):
        probe.restore_raw(tmp_path,'job','facts')


def test_complete_raw_return_recovers_only_parser(tmp_path):
    durable(tmp_path/'protocol.json',{})
    fp=probe.sha256(tmp_path/'protocol.json')
    durable(tmp_path/'calls/facts/job.json',dict(status='failed_or_uncertain',gpu='7',protocol_sha256=fp))
    durable(tmp_path/'raw_generations/facts/job.json',dict(
        identity=dict(job_id='job',method='facts',protocol_sha256=fp,gpu='7'),
        generation_state=dict(started=True,returned=True),raw_output=example(),
        metadata=dict(frame_ids=[3],truncated=False)))
    value=probe.restore_raw(tmp_path,'job','facts')
    assert value['parsed']['valid']
    assert probe.read_json(tmp_path/'calls/facts/job.json')['status']=='completed'


def test_reference_must_precede_any_started_call(tmp_path,monkeypatch):
    group=dict(case_id='one')
    monkeypatch.setattr(probe,'verify_assets',lambda root:dict(groups=[group],code={}))
    durable(tmp_path/'assets.json',{})
    durable(tmp_path/'evaluation_reference.json',dict(stage='before_model_generation',cases={'one':dict(
        facts=[dict(frames=dict(witness=[1],comparison=[]))],relations=[])}))
    durable(tmp_path/'calls/facts/job.json',{})
    with pytest.raises(ValueError,match='after a model invocation'):
        probe.freeze_reference(tmp_path)
