"""exp23协调器恢复边界，不依赖模型或真实数据。"""
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).parents[1] / 'scripts/run_optional_seed64_development.py'
spec = importlib.util.spec_from_file_location('optional_runner', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_completed_only_reports_after_unlock(tmp_path, monkeypatch):
    """完成项不调用准备或生成；报告子进程启动前父锁已释放。"""
    monkeypatch.setattr(runner, 'RUN', tmp_path / 'run')
    monkeypatch.setattr(runner, 'QA', tmp_path / 'qa')
    runner.durable(runner.QA / 'completion_audit.json', {'status': 'passed'})
    monkeypatch.setattr(runner, 'prepare', lambda: pytest.fail('unexpected prepare'))
    calls = []

    def subprocess(args, **kwargs):
        with runner.exclusive(runner.RUN / 'run.lock'):
            calls.append(args)

    monkeypatch.setattr(runner.subprocess, 'run', subprocess)
    runner.main('run')
    assert calls == [[runner.PYTHON, str(path), '--stage', 'report']]


def test_prepare_stage_never_starts_subprocess(tmp_path, monkeypatch):
    """只准备不会暗中问答或启动统计进程。"""
    monkeypatch.setattr(runner, 'RUN', tmp_path / 'run')
    monkeypatch.setattr(runner, 'QA', tmp_path / 'qa')
    prepared = []
    monkeypatch.setattr(runner, 'prepare', lambda: prepared.append(True))
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected subprocess'))
    runner.main('prepare')
    assert prepared == [True]


def test_immutable_rejects_changed_input(tmp_path):
    """恢复接受相同内容，拒绝用新输入覆盖已冻结检查点。"""
    dest = tmp_path / 'item.json'
    runner.immutable(dest, {'frames': [1, 2]})
    runner.immutable(dest, {'frames': [1, 2]})
    with pytest.raises(ValueError, match='Frozen content changed'):
        runner.immutable(dest, {'frames': [1, 3]})
