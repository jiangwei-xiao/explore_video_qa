"""已授权换卡接续：无标签分配、旧产物保护以及实际/历史物理GPU明确区分。"""
import copy
import pytest

from videoqa_full.state import durable
from videoqa_runtime.common import read_json,sha256
from videoqa_audit import supervision_quality as d
from videoqa_audit import supervision_quality_relocation as r


def test_remaining_assignment_is_stable_and_never_uses_other_gpus():
    cases=[dict(row=dict(question_id=f'q{i}'),duration_seconds=float(i+8),ordinal=i) for i in range(65)]
    a=r.assign_remaining(cases)
    poisoned=copy.deepcopy(cases)
    for c in poisoned:c['row']['correct']=bool(c['ordinal']%2)
    assert a==r.assign_remaining(list(reversed(poisoned)))
    assert len(a)==65 and set(a.values())=={'0','1','2','3'}
    assert max(list(a.values()).count(g) for g in r.GPUS)-min(list(a.values()).count(g) for g in r.GPUS)<=2


def test_handoff_protects_old_results_bytes(tmp_path):
    durable(tmp_path/'results/O/q.json',dict(raw='A',completed=True))
    relative='results/O/q.json'
    receipt=dict(stopped_coordinator_pid=9876543210123,artifacts={relative:sha256(tmp_path/relative)})
    durable(tmp_path/'relocation_handoff.json',receipt)
    assert r.verify_parent_handoff(tmp_path)==receipt
    durable(tmp_path/relative,dict(raw='B',completed=True))
    with pytest.raises(ValueError,match='artifact changed'):r.verify_parent_handoff(tmp_path)


def test_reused_process_question_records_actual_gpu_and_retains_history(tmp_path,monkeypatch):
    q='q';evaluation={q:dict(answer_index=0,gpu='0',historical_gpu='4',historical={d.METHODS[0]:dict(parsed_answer='A')})}
    durable(tmp_path/'evaluation.json',evaluation)
    entries={c:dict(input_key=c) for c in d.CONDITIONS};groups=d.group_conditions(entries)
    durable(tmp_path/'inputs/q.json',dict(entries=entries,groups=groups))
    durable(tmp_path/'inputs_complete.json',dict(exports={q:sha256(tmp_path/'inputs/q.json')}))
    calls=[]
    def generate(root,row,condition,entry,backend,adapter,gpu):
        calls.append((condition,gpu));return dict(raw_output='A',parsed_answer='A')
    monkeypatch.setattr(d,'_generate',generate)
    d.process_question(tmp_path,dict(row=dict(question_id=q),ordinal=0),None,None,'0')
    assert len(calls)==5 and all(g=='0' for _,g in calls)
    assert all(read_json(tmp_path/'results'/c/'q.json')['gpu']=='0' for c in d.CONDITIONS)
    assert read_json(tmp_path/'evaluation.json')[q]['historical_gpu']=='4'
