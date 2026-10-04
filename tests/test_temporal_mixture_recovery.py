"""已完成exp22恢复不得重新进入输入导出或生成阶段。"""
import importlib.util
from pathlib import Path
import sys

scripts=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('mixture_runner',scripts/'run_temporal_mixture_development.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_completed_run_only_invokes_report(tmp_path,monkeypatch):
    """只提供完成审计，不提供模型或选择资产，验证不会进入准备/生成。"""
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'RUN',tmp_path/'coordinator')
    qa=tmp_path/'outputs/diagnostics'/module.NAME
    module.durable(qa/'completion_audit.json',dict(status='passed'))
    calls=[]
    monkeypatch.setattr(module.subprocess,'run',lambda args,**kwargs:calls.append(args))
    module.main()
    assert calls==[[module.PYTHON,str(tmp_path/'scripts/recover_temporal_mixture_report.py')]]
