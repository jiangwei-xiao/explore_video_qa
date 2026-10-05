"""阶段接续不能重新生成已完成组，也不能漏传显式恢复标志。"""
import importlib.util
from pathlib import Path


def test_resume_commands_exclude_completed_qwen_and_preparation():
    """只恢复剩余两组；同运行编号、同GPU范围及每阶段--resume均须保留。"""
    path=Path(__file__).resolve().parents[1]/'scripts/resume_exp12_generalization_full.py'
    spec=importlib.util.spec_from_file_location('generalization_resume',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    commands=module.remaining_commands('frozen_run',['0','1'])
    assert [stage for stage,_ in commands]==['llava16','wfs32','report']
    for stage,command in commands:
        assert command[command.index('--stage')+1]==stage
        assert command[command.index('--run-id')+1]=='frozen_run'
        assert command[-3:]==['0','1','--resume']
        assert 'prepare' not in command and 'qwen16' not in command
