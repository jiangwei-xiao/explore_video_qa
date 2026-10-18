"""最小观察接口的格式边界、来源隔离及未知生成恢复测试。"""
import inspect
from pathlib import Path
import pytest

from videoqa_audit import single_frame_fact_probe as probe
from videoqa_full.state import durable

VALID='{"visible_text":["FIRST COMPUTER TOUCHSCREEN"],"observations":["A beige computer is pictured."]}'


@pytest.mark.parametrize('wrapper',['{}','```json\n{}\n```','```\n{}\n```'])
def test_whole_response_only_normalization(wrapper):
    """只放行事前允许的完整单层包装，不依赖真实模型输出调规则。"""
    result=probe.parse_facts(wrapper.format(VALID))
    assert result['valid']
    assert result['strict_valid']==(wrapper=='{}')


@pytest.mark.parametrize('raw',[
    'Here is JSON: '+VALID,
    '```json\n'+VALID+'\n``` explanation',
    '```json\n```json\n'+VALID+'\n```\n```',
    '{"visible_text":[],"observations":[],"relations":[]}',
    '{"observations":[]}',
    '{"visible_text":[],"observations":[1]}',
    '{"visible_text":[],"observations":[""]}',
    '{"visible_text":[],"observations":["a","b","c","d"]}',
])
def test_no_body_extraction_or_schema_repair(raw):
    assert not probe.parse_facts(raw)['valid']


def test_truncated_output_is_not_rescued_by_valid_looking_json():
    assert not probe.parse_facts(VALID,True)['valid']


def test_empty_output_remains_empty_not_implicit_evidence():
    result=probe.parse_facts('{"visible_text":[],"observations":[]}')
    assert result['valid'] and result['value']==dict(visible_text=[],observations=[])


def test_prompt_interface_has_no_source_answer_or_reference_input():
    assert list(inspect.signature(probe.fact_text).parameters)==['question']
    text=probe.fact_text('What color is the strip?')
    assert text.endswith('Question:\nWhat color is the strip?')
    assert 'Frame ID' not in text and 'source time' not in text
    assert 'specific question constraint' not in text and 'visibly supported relation' not in text


def test_started_without_raw_return_never_calls_model(tmp_path):
    durable(tmp_path/'protocol.json',{})
    durable(tmp_path/'calls/facts/j.json',dict(status='started_may_generate',gpu='6',
        protocol_sha256=probe.sha256(tmp_path/'protocol.json')))
    with pytest.raises(RuntimeError,match='no retry'):probe.restore_raw(tmp_path,'j')


def test_returned_raw_only_recovers_new_parser(tmp_path):
    durable(tmp_path/'protocol.json',{});fp=probe.sha256(tmp_path/'protocol.json')
    durable(tmp_path/'calls/facts/j.json',dict(status='failed_or_uncertain',gpu='6',protocol_sha256=fp))
    raw=dict(identity=dict(job_id='j',method='facts',protocol_sha256=fp,gpu='6'),
        generation_state=dict(started=True,returned=True),raw_output='```json\n'+VALID+'\n```',metadata=dict(truncated=False))
    durable(tmp_path/'raw_generations/facts/j.json',raw)
    answer=probe.restore_raw(tmp_path,'j')
    assert answer['parsed']['valid'] and not answer['parsed']['strict_valid']
    assert probe.read_json(tmp_path/'calls/facts/j.json')['recovered_postprocessing']
    assert probe.read_json(tmp_path/'raw_generations/facts/j.json')==raw


def test_raw_return_with_different_gpu_is_not_reused(tmp_path):
    durable(tmp_path/'protocol.json',{});fp=probe.sha256(tmp_path/'protocol.json')
    durable(tmp_path/'calls/facts/j.json',dict(status='failed_or_uncertain',gpu='6',protocol_sha256=fp))
    durable(tmp_path/'raw_generations/facts/j.json',dict(identity=dict(job_id='j',method='facts',protocol_sha256=fp,gpu='7'),
        generation_state=dict(started=True,returned=True),raw_output=VALID,metadata=dict(truncated=False)))
    with pytest.raises(RuntimeError,match='no retry'):probe.restore_raw(tmp_path,'j')
