"""全量适配回归：缓存身份、冻结额度、原轨迹及零生成恢复边界。"""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from videoqa_generalization import region_scope_full as study
from videoqa_generalization import region_scope_full_runtime as runtime
from videoqa_generalization.region_scope_full_report import decision,panel_entries
from videoqa_full.state import durable
from videoqa_runtime.common import sha256


def test_exact_partition_and_no_automatic_expansion():
    for p in study.PROFILES:study.validate_partition(p,study.EXPECTED[p])
    with pytest.raises(ValueError):study.validate_partition('qwen32',dict(development=200,parent=1560,new=940))
    assert sum(v['new'] for v in study.EXPECTED.values())==1096


@pytest.mark.parametrize('change',[dict(question_id='wrong'),dict(profile='llava8'),dict(frame_budget=8),dict(method='RD-1.2-exp07')])
def test_valid_cache_in_wrong_slot_rejected(change):
    v=dict(question_id='001-1',profile='qwen32',frame_budget=32,method=study.METHOD)
    study.check_slot(v,'qwen32','001-1');v.update(change)
    with pytest.raises(ValueError):study.check_slot(v,'qwen32','001-1')


def example_trace():
    return dict(region_budgets={'0':2},protected_indices=[0],eligible_indices=[0,1],objective=.8,
        regions=[dict(region_id=0,members=[0,1],reference_coarse_indices=[0,1],prefix=[0,1],density_weight=1,
            coverage_curve=[0.,.4,.8],local_steps=[dict(candidate_index=1,support=.8,redundancy=.2,local_score=.7)])])


def test_trace_exact_discrete_and_numeric_tolerance():
    a=example_trace();b=copy.deepcopy(a);assert study.compare_trace(a,b)==0
    b['regions'][0]['local_steps'][0]['support']+=5e-7;assert 0<study.compare_trace(a,b)<1e-6
    b['regions'][0]['prefix']=[1,0]
    with pytest.raises(ValueError):study.compare_trace(a,b)
    b=copy.deepcopy(a);b['objective']+=2e-6
    with pytest.raises(ValueError):study.compare_trace(a,b)


def test_source_identity_uses_order_and_clock_not_candidate_ids():
    s=dict(video=dict(path='v',start_pts=2**35,time_base='1/90000',duration_fraction='20'),
        selected_frames=[dict(source_frame_index=4,source_pts=2**35+3,candidate_index=1),dict(source_frame_index=5,source_pts=2**35+9,candidate_index=2)])
    t=copy.deepcopy(s);t['selected_frames'][0]['candidate_index']=99
    assert study.source_identity(s)==study.source_identity(t)
    t['selected_frames'].reverse();assert study.source_identity(s)!=study.source_identity(t)


def test_import_ancestry_resolves_physical_shared_call(tmp_path):
    run=tmp_path/'old';q='001-1';method='RD-1.2-exp07';primary='BASE-Uniform';fp='frozen';answer={'raw_output':'A'}
    p=run/'results'/method/(q+'.json');old=dict(question_id=q,method=method,representative=primary,protocol_sha256=fp,answer=answer,gpu='6')
    durable(p,old);cp=run/'calls'/primary/(q+'.json');rp=run/'raw_generations'/primary/(q+'.json')
    durable(cp,dict(status='completed',protocol_sha256=fp,answer=answer,gpu='6'))
    durable(rp,dict(identity=dict(question_id=q,method=primary,protocol_sha256=fp,gpu='6'),generation_state={'started':True,'returned':True}))
    refs,gpu=study.provenance(p,old);assert gpu=='6';assert str(cp) in refs and str(rp) in refs
    v=__import__('json').loads(rp.read_text());v['generation_state']['returned']=False;durable(rp,v)
    with pytest.raises(ValueError):study.provenance(p,old)


def test_returned_raw_restores_without_model_or_reinvocation(tmp_path,monkeypatch):
    q='001-1';run=tmp_path;fp='protocol';gpu='4';answer={'raw_output':'A'};row={'question_id':q}
    durable(run/'protocol.json',{'x':1});fp=sha256(run/'protocol.json')
    durable(run/'calls'/study.METHOD/(q+'.json'),dict(status='completed',protocol_sha256=fp,gpu=gpu,answer=answer))
    durable(run/'raw_generations'/study.METHOD/(q+'.json'),dict(identity=dict(question_id=q,method=study.METHOD,protocol_sha256=fp,gpu=gpu),generation_state={'started':True,'returned':True}))
    monkeypatch.setattr(runtime,'result_from_answer',lambda *a,**kw:dict(restored=True))
    assert runtime.recover_group(run,row,dict(representative='S',controls=['S']),dict(profile='qwen32'))
    assert (run/'results'/study.METHOD/(q+'.json')).exists()
    v=__import__('json').loads((run/'raw_generations'/study.METHOD/(q+'.json')).read_text());v['identity']['question_id']='wrong';durable(run/'raw_generations'/study.METHOD/(q+'.json'),v)
    with pytest.raises(ValueError):runtime.recover_group(run,row,dict(representative='S',controls=['S']),dict(profile='qwen32'))


def test_decision_requires_both_nonnegative():
    assert decision(dict(qwen32=0,llava8=1))['observed_non_degrade_both']
    assert not decision(dict(qwen32=4,llava8=-1))['observed_non_degrade_both']
    assert not decision(dict(qwen32=0,llava8=0))['default_replacement']


def test_panel_medium_long_preferred_and_empty_slots():
    data=[dict(question_id='a',video_id='dev',stratum='medium'),dict(question_id='b',video_id='held',stratum='medium'),dict(question_id='c',video_id='third',stratum='long')]
    profiles={p:dict(comparisons={'RD-1.2-exp07':{'all2700':dict(cells={'left_only':['a','b'],'right_only':['c']})}}) for p in study.PROFILES}
    entries=panel_entries(data,profiles,{'dev'});assert entries[0]['question_id']=='b';assert len({v['question_id'] for v in entries})==len(entries)
    assert all(v['stratum'] in ('medium','long') for v in entries)
    assert len(entries)<8


def test_stage_order_freezes_both_inputs_before_qwen_answers():
    p=Path(__file__).parents[1]/'scripts/run_region_scope_full.py';spec=importlib.util.spec_from_file_location('exp27_cli',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    commands=m.commands('test',['4','5']);names=[v[0] for v in commands]
    assert names.index('prepare_llava8')<names.index('answer_qwen32')<names.index('answer_llava8')
    assert all('--resume' in c for _,c in commands)
