"""扩展评分预算测试，防止合成前向进入真实账本；不调用模型。"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('event_extension', Path(__file__).resolve().parents[1] / 'scripts/probe_event_clause_extension.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_synthetic_and_real_native_batches_have_separate_budgets(tmp_path):
    real = tmp_path / 'calls/259-1/event_clause/0.json'
    synthetic = tmp_path / 'calls/synthetic/warmup/0.json'
    for p in (real, synthetic):
        p.parent.mkdir(parents=True); p.write_text('{}')
    assert m.real_score_paths(tmp_path) == [real]


def test_synthetic_only_is_zero_real_calls(tmp_path):
    synthetic = tmp_path / 'calls/synthetic/warmup/0.json'
    synthetic.parent.mkdir(parents=True); synthetic.write_text('{}')
    assert m.real_score_paths(tmp_path) == []
