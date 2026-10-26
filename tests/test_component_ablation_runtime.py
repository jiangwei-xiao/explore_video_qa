"""组件全量执行的输入分组、调用恢复、继承来源和不降门；无真实模型前向。"""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from videoqa_full.state import durable
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_generalization import component_ablation_runtime as runtime
from videoqa_generalization import component_ablation_report as report


def test_input_grouping_does_not_read_correctness_and_is_within_question():
    items={'A':dict(identity={'pixels':'same','tokens':[1]},correct=True),
           'B':dict(identity={'pixels':'same','tokens':[1]},correct=False)}
    groups=runtime.input_groups(items)
    assert len(groups)==1 and groups[0]['representative']=='A' and groups[0]['controls']==['A','B']
    items['B']['identity']['tokens']=[2]
    assert len(runtime.input_groups(items))==2


def test_processed_identity_checks_actual_qwen_pixels_prompt_tokens_and_grid():
    expected=dict(pixel_sha256='p',prompt='text',input_token_ids=[1],video_grid_thw=[[4,24,32]])
    observed=dict(processor_pixel_sha256='p',prompt='text',input_token_ids=[1],video_grid_thw=[[4,24,32]])
    assert runtime.processed_matches('qwen32',expected,observed)
    for key in expected:
        wrong=deepcopy(observed)
        wrong['processor_pixel_sha256' if key=='pixel_sha256' else key]=None
        assert not runtime.processed_matches('qwen32',expected,wrong)


def test_synthetic_boundary_and_global_cap_are_fixed():
    assert [n for n,_ in runtime.synthetic_cases('llava8',True)]==[1,7,8]
    assert [n for n,_ in runtime.synthetic_cases('qwen32',True)]==[1,11,31,32]
    assert len(runtime.synthetic_cases('llava8',True))+7+len(runtime.synthetic_cases('qwen32',True))+7==21
    assert runtime.synthetic_cases('llava8',False)[0][0]==8


def test_historical_shared_call_in_imported_directory_uses_its_actual_parent(tmp_path):
    base=tmp_path/'old'/'imported';q='001-1';method='RD-1.2-exp12';physical='RD-1.2-exp07'
    path=base/'results'/method/(q+'.json')
    answer={'raw_output':'A'}
    old=dict(question_id=q,method=method,representative=physical,gpu='3',protocol_sha256='old',answer=answer)
    durable(path,old)
    cp=base/'calls'/physical/(q+'.json');rp=base/'raw_generations'/physical/(q+'.json')
    durable(cp,dict(status='completed',answer=answer,protocol_sha256='old'))
    identity=dict(question_id=q,method=physical,protocol_sha256='old',gpu='3')
    durable(rp,dict(identity=identity,generation_state=dict(started=True,returned=True)))
    sources=runtime.historical_bundle(path,old)
    assert set(sources)=={str(path),str(cp),str(rp)}
    identity['gpu']='4';durable(rp,dict(identity=identity,generation_state=dict(started=True,returned=True)))
    with pytest.raises(ValueError,match='identity'):
        runtime.historical_bundle(path,old)


def test_confirmed_raw_return_recovers_both_aliases_without_regeneration(tmp_path,monkeypatch):
    run=tmp_path/'qwen32';q='001-1';durable(run/'protocol.json',{'profile':'qwen32'})
    fp=sha256(run/'protocol.json');method='RD-1.2-exp20'
    cp=run/'calls'/method/(q+'.json');rp=run/'raw_generations'/method/(q+'.json')
    durable(cp,dict(status='failed_or_uncertain',protocol_sha256=fp,gpu='1'))
    raw=dict(identity=dict(question_id=q,method=method,protocol_sha256=fp,gpu='1'),
             generation_state=dict(started=True,returned=True),raw_generated_text='A')
    durable(rp,raw)
    import videoqa_budget32.observe as observed
    monkeypatch.setattr(observed,'result_from_raw',lambda raw,row:{'raw_output':raw['raw_generated_text']})
    monkeypatch.setattr(runtime,'result_from_answer',lambda run,row,c,answer,gpu,kind,**kw:dict(control=c,answer=answer,kind=kind))
    group=dict(representative='A',controls=['A','B'])
    assert runtime.recover_group(run,{'question_id':q},group,{'profile':'qwen32'})
    paths=[run/'results'/m/(q+'.json') for m in ['RD-1.2-exp20','RD-1.2-exp04']]
    assert all(p.exists() for p in paths) and read_json(cp)['status']=='completed'
    before=[p.read_bytes() for p in paths]
    assert runtime.recover_group(run,{'question_id':q},group,{'profile':'qwen32'})
    assert before==[p.read_bytes() for p in paths]


def test_unconfirmed_raw_must_pause_without_reissuing(tmp_path):
    run=tmp_path/'qwen32';q='q';durable(run/'protocol.json',{'profile':'qwen32'})
    fp=sha256(run/'protocol.json');method='RD-1.2-exp20'
    durable(run/'calls'/method/(q+'.json'),dict(status='failed_or_uncertain',protocol_sha256=fp,gpu='1'))
    durable(run/'raw_generations'/method/(q+'.json'),dict(identity=dict(question_id=q,method=method,protocol_sha256=fp,gpu='1'),
                                                       generation_state=dict(started=True,returned=False)))
    with pytest.raises(ValueError,match='may not be repeated'):
        runtime.recover_group(run,{'question_id':q},dict(representative='A',controls=['A','B']),{'profile':'qwen32'})
    assert not (run/'results').exists()


def test_both_observed_scores_required_and_interval_is_not_equivalence():
    differences={p:dict(cells=dict(left_only=['q'],right_only=['r']),stratified_video_cluster_bootstrap_95=[-1.,1.])
                 for p in ['llava8','qwen32']}
    result=report.recommend(differences)
    assert result['replacement_candidate'] and not result['lower_confidence_bound_nonnegative_both']
    assert not result['default_replacement_authorized']
    differences['llava8']['cells']['right_only'].append('s')
    assert not report.recommend(differences)['replacement_candidate']
    with pytest.raises(ValueError):report.recommend({'llava8':differences['llava8']})


def test_pipeline_freezes_both_profiles_before_any_answer_and_passes_resume():
    path=ROOT/'scripts/run_rd_component_ablations.py'
    spec=importlib.util.spec_from_file_location('component_ablation_entry_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    commands=module.commands('test_run',['1','2'])
    assert [name for name,_ in commands]==['offline','prepare_llava8','prepare_qwen32','answer_llava8','answer_qwen32','report','audit']
    assert all('--resume' in command for _,command in commands)
    assert all('--gpus' in command for _,command in commands)
