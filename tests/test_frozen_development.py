"""独立开发评测的预算、名单与恢复边界；测试不得导入真实后端或生成答案。"""
import json
import pytest
from videoqa_budget32 import development as d


def test_selected_rows_preserves_frozen_order(monkeypatch):
    """接口：输入顺序必须来自冻结名单，而非按结果正确性重新挑选。"""
    data=[dict(question_id=str(i)) for i in range(200)]
    monkeypatch.setattr(d,'rows',lambda:data)
    ids=list(reversed([r['question_id'] for r in data]))
    assert [r['question_id'] for r in d.selected_rows(dict(question_ids=ids))]==ids
    with pytest.raises(AssertionError):d.selected_rows(dict(question_ids=ids[:199]))
    with pytest.raises(AssertionError):d.selected_rows(dict(question_ids=['0']*200))


def test_completed_resume_does_not_start_workers(tmp_path,monkeypatch):
    """接口：200结果完整时，恢复只验证汇总，绝不再创建问答工作进程。"""
    data=[dict(question_id=str(i)) for i in range(200)]
    spec=dict(methods=['RD-1.2'],question_ids=[r['question_id'] for r in data],allowed_gpus=['0'])
    path=tmp_path/'results/RD-1.2';path.mkdir(parents=True)
    for r in data:(path/(r['question_id']+'.json')).write_text('{}')
    seen=[]
    monkeypatch.setattr(d,'verify',lambda run:spec)
    monkeypatch.setattr(d,'selected_rows',lambda spec:data)
    monkeypatch.setattr(d,'validate',lambda run,row,m,s:seen.append(row['question_id']))
    monkeypatch.setattr(d,'summarize',lambda run:{'completed':200})
    monkeypatch.setattr(d.mp,'get_context',lambda *a:pytest.fail('Unexpected worker startup'))
    assert d.execute(tmp_path)=={'completed':200}
    assert len(seen)==200


def test_uncertain_call_stops_before_worker_start(tmp_path,monkeypatch):
    """接口：不确定生成占用原ID，必须暂停而不是自动补一次生成。"""
    path=tmp_path/'calls/RD-1.2';path.mkdir(parents=True)
    (path/'q.json').write_text('{}')
    monkeypatch.setattr(d,'verify',lambda run:dict(methods=['RD-1.2'],allowed_gpus=['0']))
    monkeypatch.setattr(d,'selected_rows',lambda spec:[dict(question_id='q')])
    def fail(*args):raise RuntimeError('Uncertain generation')
    monkeypatch.setattr(d,'recover_call',fail)
    monkeypatch.setattr(d.mp,'get_context',lambda *a:pytest.fail('Unexpected worker startup'))
    with pytest.raises(RuntimeError,match='Uncertain generation'):d.execute(tmp_path)


def test_completion_audit_detects_mutation(tmp_path,monkeypatch):
    """接口：完成恢复要求600份结果/调用/原文内容不变，不能只比较数量。"""
    for folder in ('calls','results','raw_generations'):
        p=tmp_path/folder/'RD-1.2';p.mkdir(parents=True)
        for i in range(200):(p/f'{i}.json').write_text('{}')
    monkeypatch.setattr(d,'execute',lambda run:None)
    d.audit(tmp_path)
    assert json.loads((tmp_path/'completion_audit.json').read_text())['extra_generations']==0
    def change(run):(run/'results/RD-1.2/0.json').write_text('{"changed": true}')
    monkeypatch.setattr(d,'execute',change)
    with pytest.raises(AssertionError):d.audit(tmp_path)
