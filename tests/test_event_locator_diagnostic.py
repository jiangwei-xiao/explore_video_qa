"""自动定位诊断的输入/额度/恢复测试，全部无模型调用。"""
import importlib.util
from pathlib import Path
import json
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/run_event_locator_diagnostic.py'
spec = importlib.util.spec_from_file_location('event_locator_diag', path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_fixed_population_and_alternating_order():
    order = m.schedule({q: i for i, q in enumerate(m.QIDS)})
    assert len(order) == 10 and len(set(order)) == 10
    assert order[:2] == [(m.QIDS[0], c) for c in m.CONDITIONS]
    assert order[2:4] == [(m.QIDS[1], c) for c in m.CONDITIONS[::-1]]
    assert set(m.QIDS) == {'555-2', '560-2', '599-1', '252-3', '553-3'}


def test_synthetic_ledger_does_not_consume_real_budget(tmp_path):
    p = tmp_path / 'synthetic/calls/synthetic/warmup.json'
    p.parent.mkdir(parents=True); p.write_text('{}')
    assert m.checked_calls(tmp_path) == []


@pytest.mark.parametrize('q,c', [('555-2', 'manual_repair'), ('022-1', m.CONDITIONS[0])])
def test_unplanned_input_cannot_enter_call_budget(tmp_path, q, c):
    p = tmp_path / 'calls' / c / (q + '.json')
    p.parent.mkdir(parents=True); p.write_text('{}')
    with pytest.raises(ValueError): m.checked_calls(tmp_path)


@pytest.mark.parametrize('field', ['processor_pixel_sha256', 'prompt', 'input_token_ids', 'video_grid_thw'])
def test_actual_observation_must_match_all_frozen_inputs(field):
    expected = dict(pixel_sha256='pixels', prompt='original question/options',
                    input_token_ids=[1, 2], video_grid_thw=[[16, 48, 64]])
    actual = dict(processor_pixel_sha256='pixels', prompt=expected['prompt'],
                  input_token_ids=[1, 2], video_grid_thw=[[16, 48, 64]])
    m.check_identity(actual, expected)
    actual[field] = 'changed'
    with pytest.raises(ValueError): m.check_identity(actual, expected)


def test_completed_recovery_never_loads_backend_or_generates(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'verify', lambda root: {})
    monkeypatch.setattr(m, 'rows', lambda: [dict(question_id=q) for q in m.QIDS])
    for q in m.QIDS:
        for c in m.CONDITIONS:
            p = tmp_path / 'results' / c / (q + '.json')
            p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')
    monkeypatch.setattr(m, 'audit', lambda root: 'verified')
    monkeypatch.setattr(m, 'process', lambda *a: pytest.fail('Completed input generated again'))
    assert m.execute(tmp_path) == 'verified'


def test_partial_return_only_recovers_existing_call_and_does_not_start_remaining(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'verify', lambda root: {})
    monkeypatch.setattr(m, 'rows', lambda: [dict(question_id=q) for q in m.QIDS])
    q, c = m.QIDS[0], m.CONDITIONS[0]
    p = tmp_path / 'calls' / c / (q + '.json')
    p.parent.mkdir(parents=True); p.write_text(json.dumps(dict(gpu='0')))
    recovered = []; processed = []
    monkeypatch.setattr(m, 'recover_call', lambda root, row, condition, spec: recovered.append((row['question_id'], condition)))
    monkeypatch.setattr(m, 'process', lambda root, row, condition, backend, adapter, gpu, spec:
                        processed.append((row['question_id'], condition, backend, adapter)))
    with pytest.raises(RuntimeError, match='Partial diagnostic'): m.execute(tmp_path)
    assert recovered == [(q, c)] and processed == [(q, c, None, None)]

