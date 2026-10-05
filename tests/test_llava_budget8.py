"""八帧同题输入分组及恢复隔离的关键行为测试。"""
from pathlib import Path
import importlib.util
import pytest
from videoqa_generalization import llava_budget8 as experiment


def test_same_input_group_has_fixed_representative():
    items={m:dict(input_identity=dict(rgb=['same'],tokens=[1])) for m in experiment.METHODS}
    groups=experiment.input_groups(items)
    assert len(groups)==1 and groups[0]['representative']=='BASE-Uniform'
    assert groups[0]['methods']==list(experiment.METHODS)


@pytest.mark.parametrize('field',['source','rgb','pixels','tokens','protocol'])
def test_one_identity_difference_prevents_sharing(field):
    identity=dict(source=1,rgb=1,pixels=1,tokens=1,protocol=1)
    items={m:dict(input_identity=dict(identity)) for m in experiment.METHODS}
    items['RD-1.2-exp12']['input_identity'][field]=2
    groups=experiment.input_groups(items)
    assert len(groups)==2 and groups[1]['methods']==['RD-1.2-exp12']


def test_uncertain_raw_never_repeats_generation(tmp_path,monkeypatch):
    from videoqa_full.state import durable
    q='001-1';path=tmp_path/'calls/BASE-Uniform'/f'{q}.json'
    durable(path,dict(status='started_may_generate',protocol_sha256='f',gpu='0'))
    with pytest.raises(FileNotFoundError):experiment.recover_call(tmp_path,dict(question_id=q),'BASE-Uniform','f')
    assert path.exists()


def test_completed_ledger_is_read_only(tmp_path):
    from videoqa_full.state import durable
    q='001-1';path=tmp_path/'calls/BASE-Uniform'/f'{q}.json'
    durable(path,dict(status='completed',protocol_sha256='f',gpu='0',answer='A'))
    before=path.read_bytes();experiment.recover_call(tmp_path,dict(question_id=q),'BASE-Uniform','f')
    assert before==path.read_bytes()


def test_pipeline_stage_order_and_explicit_resume():
    path=Path(__file__).resolve().parents[1]/'scripts/run_llava_budget8_full.py'
    spec=importlib.util.spec_from_file_location('llava8_entry',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    commands=module.pipeline_commands('frozen',['0','1'])
    assert [s for s,_ in commands]==['prepare','synthetic','answer','audit','report']
    assert all(c[-1]=='--resume' for _,c in commands)


def test_duration_repair_does_not_mutate_frozen_spec():
    """历史统计只包含文件大小和路径，排序兼容不能改写冻结协议或预算。"""
    path=Path(__file__).resolve().parents[1]/'scripts/resume_llava8_after_pilot.py'
    spec=importlib.util.spec_from_file_location('llava8_repair',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=dict(video_stats={'v':dict(path='v.mp4',bytes=4)},max_calls=11932,methods=list(experiment.METHODS))
    adapted=module.duration_compatible(original,{'v':42.25})
    assert 'duration_seconds' not in original['video_stats']['v']
    assert adapted['video_stats']['v']['duration_seconds']==42.25
    assert adapted['max_calls']==11932 and adapted['methods']==original['methods']
    with pytest.raises(ValueError):module.duration_compatible(original,{'v':0})
