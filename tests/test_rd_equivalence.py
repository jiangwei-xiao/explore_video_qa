"""等价审计的来源绑定、求解计数、反例与完成恢复；不需要真实资产或模型。"""
import copy
from pathlib import Path
import numpy as np
import pytest

from videoqa_audit import rd_equivalence as audit
from videoqa_methods import rd_selection as production
from videoqa_full.state import durable,digest
from test_rd_selection import fixture


def test_actual_dp_counts_and_frozen_trace_comparison():
    pool,f=fixture()
    for method,expected in zip(production.METHODS,(1,2)):
        with audit.count_dp() as count:old_ids,old,prefix=audit._legacy(pool,f,4,method)
        assert count[0]==expected
        with audit.count_dp() as count:new_ids,new=production.select_rd(pool,f,4,method,'full')
        assert count[0]==1
        frozen=dict(trace=old,selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in old_ids]))
        assert audit.compare_outputs(pool,(old_ids,old),(new_ids,new),frozen,prefix)<=1e-6
        wrong=copy.deepcopy(new);wrong['regions'][0]['coverage_curve'][1]+=.01
        with pytest.raises(ValueError,match='Numerical'):audit.compare_outputs(pool,(old_ids,old),(new_ids,wrong),frozen,prefix)


def test_feature_mapping_is_checked_by_source_not_only_shape():
    pool,f=fixture();mapped=copy.deepcopy(pool);mapped['original_candidate_indices']=list(range(len(f)))
    assert np.array_equal(audit.mapped_features(mapped,pool,f),f)
    mapped['original_candidate_indices'][0:2]=[1,0]
    with pytest.raises(ValueError,match='source identity'):audit.mapped_features(mapped,pool,f)


def test_old_checkpoint_binding_rejects_current_but_untrusted_hash(tmp_path):
    path=tmp_path/'pool.json';path.write_text('{}');cp=tmp_path/'cp.json'
    durable(cp,dict(files={'pool.json':audit.file_hash(path)}));sources={}
    audit._bound(path,cp,tmp_path,sources)
    path.write_text('{"different":true}')
    with pytest.raises(ValueError,match='binding'):audit._bound(path,cp,tmp_path,{})


def test_record_recovery_never_invokes_selector_and_rejects_source_drift(tmp_path,monkeypatch):
    source=tmp_path/'source.json';source.write_text('{}');path=tmp_path/'record.json'
    data=dict(sources={str(source):audit.file_hash(source)},methods={m:{} for m in production.METHODS})
    data['record_sha256']=digest(data);durable(path,data)
    def forbidden(*a,**kw):raise AssertionError('Restore computed a selection')
    monkeypatch.setattr(production,'select_rd',forbidden);monkeypatch.setattr(audit,'_legacy',forbidden)
    assert audit.verify_record(path)==data
    source.write_text('{"changed":1}')
    with pytest.raises(ValueError,match='source changed'):audit.verify_record(path)


def test_completed_execute_uses_only_audit_without_pool_or_prepare(tmp_path,monkeypatch):
    durable(tmp_path/'completion.json',dict(status='passed'))
    def forbidden(*a,**kw):raise AssertionError('Complete restore started work')
    monkeypatch.setattr(audit,'prepare',forbidden);monkeypatch.setattr(audit,'ProcessPoolExecutor',forbidden)
    monkeypatch.setattr(audit,'audit_completed',lambda root:dict(status='already_completed',new_selection_calls=0))
    assert audit.execute(tmp_path)['new_selection_calls']==0


def test_dp_counter_restores_functions_after_exception():
    functions=[m.allocate_prefix_budget for m in (audit.legacy_prefix,audit.legacy_weighted,production)]
    with pytest.raises(RuntimeError):
        with audit.count_dp():raise RuntimeError('stop')
    assert functions==[m.allocate_prefix_budget for m in (audit.legacy_prefix,audit.legacy_weighted,production)]


@pytest.mark.parametrize('qid,k',[('other-question',8),('q1',16)])
def test_valid_record_in_wrong_question_or_budget_slot_is_rejected(tmp_path,qid,k):
    path=tmp_path/'records/videomme/K8/q1.json';path.parent.mkdir(parents=True)
    value=dict(dataset='videomme',question_id=qid,frame_budget=k,sources={},methods={m:{} for m in production.METHODS})
    value['record_sha256']=digest(value);durable(path,value)
    with pytest.raises(ValueError,match='identity differs'):audit.verify_record(path)


@pytest.mark.parametrize('field,value',[('methods',['wrong-method']),('expected_comparisons',1)])
def test_derived_protocol_scope_changes_are_rejected(tmp_path,field,value):
    spec=dict(methods=list(production.METHODS),scopes=copy.deepcopy(audit.SCOPES),
        expected_comparisons=24896,implementation_id=production.IMPLEMENTATION_ID)
    spec[field]=value;durable(tmp_path/'protocol.json',spec)
    with pytest.raises(ValueError,match='scope changed'):audit.verify_protocol(tmp_path)


def test_completion_is_bound_to_original_protocol_bytes(tmp_path,monkeypatch):
    durable(tmp_path/'protocol.json',{});durable(tmp_path/'summary.json',{})
    durable(tmp_path/'completion.json',dict(comparisons=24896,summary_sha256=audit.file_hash(tmp_path/'summary.json'),protocol_sha256='wrong'))
    monkeypatch.setattr(audit,'verify_protocol',lambda root:{})
    with pytest.raises(ValueError,match='Completion binding'):audit.audit_completed(tmp_path)
