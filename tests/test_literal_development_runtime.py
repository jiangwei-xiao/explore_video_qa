"""exp32输入分区、来源及方法隔离的CPU回归，绝不调用模型。"""
import pytest
from videoqa_full.state import durable
from videoqa_audit import literal_development_runtime as runtime


def spec():
    ids = [str(i) for i in range(200)]
    return dict(method=runtime.METHOD, question_ids=ids, new_ids=ids[:3], imported_ids=ids[3:],
        maximum_new_real_generations=3, framework={}, environment={}, code_sha256={},
        model_files={}, sources={}, selection_exports={})


def test_complete_input_partition_is_required_before_call(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'framework_identity', lambda: {})
    monkeypatch.setattr(runtime, 'environment', lambda _: {})
    value = spec(); durable(tmp_path / 'protocol.json', value)
    assert runtime.verify(tmp_path)['maximum_new_real_generations'] == 3
    value['imported_ids'].append(value['new_ids'][0]); durable(tmp_path / 'protocol.json', value)
    with pytest.raises(ValueError, match='partition'):
        runtime.verify(tmp_path)


def test_wrong_method_or_population_cannot_share_a_call_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'framework_identity', lambda: {})
    monkeypatch.setattr(runtime, 'environment', lambda _: {})
    value = spec(); value['method'] = 'RD-1.2-exp07'; durable(tmp_path / 'protocol.json', value)
    with pytest.raises(ValueError, match='method or population'):
        runtime.verify(tmp_path)


def test_frozen_source_mutation_pauses_without_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'framework_identity', lambda: {})
    monkeypatch.setattr(runtime, 'environment', lambda _: {})
    value = spec(); file = tmp_path / 'input.json'; durable(file, {'value': 1})
    value['sources'] = {str(file): runtime.sha256(file)}; durable(tmp_path / 'protocol.json', value)
    durable(file, {'value': 2})
    with pytest.raises(ValueError, match='generation source'):
        runtime.verify(tmp_path)


@pytest.mark.parametrize('field,value', [('method', 'RD-1.2-exp07'), ('model', 'other-model'), ('question_id', 'other-question')])
def test_imported_result_metadata_cannot_cross_identity(tmp_path, monkeypatch, field, value):
    """接口：同帧引用也须属于本题/方法/模型，身份异常在读取生成观测前暂停。"""
    row = dict(question_id='q', video_id='v', question='question', options=['A', 'B', 'C', 'D'], stratum='short')
    result = dict(status='completed', model='frozen-model', method=runtime.METHOD, **row)
    result[field] = value
    monkeypatch.setattr(runtime, 'read_json', lambda path: result if 'results' in path.parts else {})
    with pytest.raises(ValueError, match='question identity'):
        runtime.validate_result(tmp_path, row, dict(model='frozen-model'))


def test_completed_resume_never_loads_or_generates(tmp_path, monkeypatch):
    """接口：全200完成态恢复只报告与留痕，不进入GPU等待或任何生成路径。"""
    root = tmp_path / 'root'; run = root / 'qa/qwen32'; dest = run / 'results' / runtime.METHOD
    for i in range(200):
        durable(dest / (str(i) + '.json'), dict(completed=True))
    frozen = dict(new_ids=[], imported_ids=[])
    monkeypatch.setattr(runtime, 'prepare', lambda _: frozen)
    monkeypatch.setattr(runtime, 'verify', lambda _: frozen)
    monkeypatch.setattr(runtime, 'import_answers', lambda *_: None)
    monkeypatch.setattr(runtime, 'development_rows', lambda: [dict(question_id=str(i)) for i in range(200)])
    reports = []
    monkeypatch.setattr(runtime, 'report', lambda *_: reports.append(True))
    monkeypatch.setattr(runtime, 'process', lambda *_: pytest.fail('Completed input was generated again'))
    runtime.execute(root)
    assert reports == [True]
    assert runtime.read_json(run / 'restore_audit.json')['new_generations'] == 0
