"""外部参考协调器的调用隔离与完成恢复。"""
import importlib.util
from pathlib import Path
import sys
import pytest

script=Path(__file__).parents[1]/'scripts/run_wfs_reference_development.py'
sys.path.insert(0,str(script.parent))
spec=importlib.util.spec_from_file_location('wfs_reference_runner',script)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_completed_does_not_prepare_or_generate(tmp_path,monkeypatch):
    """完成审计存在时只恢复报告，并在子进程启动前释放父锁。"""
    monkeypatch.setattr(runner,'RUN',tmp_path/'coord');monkeypatch.setattr(runner,'QA',tmp_path/'qa')
    runner.durable(runner.QA/'completion_audit.json',dict(status='passed'))
    monkeypatch.setattr(runner,'prepare',lambda:pytest.fail('unexpected prepare'))
    calls=[]
    def invoke(args,**kwargs):
        with runner.exclusive(runner.RUN/'run.lock'):calls.append(args)
    monkeypatch.setattr(runner.subprocess,'run',invoke)
    runner.main('run')
    assert calls==[[runner.PYTHON,str(script),'--stage','report']]


def test_prepare_never_generates(tmp_path,monkeypatch):
    """纯输入准备不加载问答或报告入口。"""
    monkeypatch.setattr(runner,'RUN',tmp_path/'coord');monkeypatch.setattr(runner,'QA',tmp_path/'qa')
    calls=[];monkeypatch.setattr(runner,'prepare',lambda:calls.append('prepare'))
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:pytest.fail('unexpected model subprocess'))
    runner.main('prepare');assert calls==['prepare']
