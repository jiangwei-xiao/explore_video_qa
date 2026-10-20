"""非交互启动验收：工具PATH固定、原入口/编号保留，缺工具即停止。"""
import importlib.util
import os
from pathlib import Path

import pytest


def load():
    """测试接口：导入隔离启动器，不运行真实协调器或模型。"""
    path=Path(__file__).resolve().parents[1]/'scripts/start_longvideo_set_value_trial.py'
    spec=importlib.util.spec_from_file_location('test_longvideo_launcher',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_launcher_preserves_entry_and_prepends_tool_path(monkeypatch):
    m=load();observed={}
    monkeypatch.setenv('PATH','/usr/bin')
    monkeypatch.setattr(m.shutil,'which',lambda name,path:str(m.PREP_BIN/name))
    monkeypatch.setattr(m.os,'chdir',lambda p:observed.update(cwd=p))
    monkeypatch.setattr(m.os,'execve',lambda executable,args,env:observed.update(executable=executable,args=args,env=env))
    m.launch('fixed_trial_r1')
    assert observed['args'][2:]==[str(m.ROOT/'scripts/run_longvideo_set_value_trial.py'),'--stage','run','--run-id','fixed_trial_r1']
    assert observed['env']['PATH']==str(m.PREP_BIN)+os.pathsep+'/usr/bin'
    assert observed['executable']==str(m.PREP_BIN/'python') and observed['cwd']==m.ROOT


def test_missing_tool_rejected_before_exec(monkeypatch):
    m=load()
    monkeypatch.setattr(m.shutil,'which',lambda name,path:None if name=='ffprobe' else str(m.PREP_BIN/name))
    monkeypatch.setattr(m.os,'execve',lambda *a:pytest.fail('Unexpected execution'))
    with pytest.raises(RuntimeError,match='ffprobe'):m.launch('fixed_trial_r1')


def test_invalid_run_id_rejected():
    with pytest.raises(ValueError):load().launch('../another-run')
