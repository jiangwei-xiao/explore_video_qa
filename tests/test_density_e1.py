"""正式E1拼接与恢复边界；纯CPU合成测试，不运行模型。"""
import json
import pytest
from test_local_distance_scale import fixture
from videoqa_audit import e1_pipeline
from videoqa_audit.e1_run import check_recovery
from videoqa_audit.density_reward_decay import select_density_decay
from videoqa_audit.local_distance_scale import select_local_distance
from videoqa_methods.followups import CallLedger

def test_pipeline_computes_quotas_then_reselects(monkeypatch,tmp_path):
    """正式流程必须从本次特征算D1配额，而不是读取离线配额。"""
    rows,f,plan,video,cfg,_=fixture()
    monkeypatch.setattr(e1_pipeline,'scan_original',lambda *a:({'candidates':rows,'scores':[.9,.8,.1],'video':video,'blip_question_tokens':8},f,{}))
    monkeypatch.setattr(e1_pipeline,'build_regions',lambda *a:plan)
    monkeypatch.setattr(e1_pipeline,'decode_regions',lambda *a:([],[],[],{}))
    result,features,timing=e1_pipeline.prepare_density('unused','question',None,cfg,tmp_path)
    _,budget=select_density_decay(rows,f,plan,video,cfg)
    ids,trace=select_local_distance(rows,f,plan,video,cfg,budget['region_budgets'])
    assert result['selected_indices']==ids and result['selection_trace']==trace
    assert result['budget_trace']==budget and result['application_cache_hits']==0
    assert timing['selection_compete_seconds']>=result['budget_compute_seconds']>=0

def test_uncertain_e1_call_blocks_recovery(tmp_path):
    d=tmp_path/'calls/answer';d.mkdir(parents=True)
    (d/'x__e1.json').write_text(json.dumps(dict(status='failed_or_uncertain',protocol_sha256='f')))
    with pytest.raises(AssertionError):check_recovery(tmp_path,[{'question_id':'x'}],'f')

def test_empty_recovery_does_not_generate(tmp_path):
    assert check_recovery(tmp_path,[{'question_id':'x'}],'f')==[]

def test_completed_e1_call_is_reused(tmp_path):
    (tmp_path/'calls/answer').mkdir(parents=True)
    ledger=CallLedger(tmp_path,'x__e1','f','0')
    assert ledger.invoke('answer',lambda state:{'raw_output':'A'})=={'raw_output':'A'}
    def forbidden(state):raise AssertionError('must not generate again')
    assert ledger.invoke('answer',forbidden)=={'raw_output':'A'}
