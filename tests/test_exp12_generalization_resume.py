"""阶段接续不能重新生成已完成组，也不能漏传显式恢复标志。"""
import importlib.util
from pathlib import Path
from videoqa_full.state import durable


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


def test_completed_resume_preserves_generations_and_manual_report(tmp_path,monkeypatch):
    """完成恢复仅调用三组审计，不重写人工报告或运行准备入口。"""
    path=Path(__file__).resolve().parents[1]/'scripts/resume_exp12_generalization_full.py'
    spec=importlib.util.spec_from_file_location('completed_generalization_resume',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for stage in ['qwen16','llava16','wfs32']:
        for folder in ['results','calls','raw_generations']:
            durable(tmp_path/stage/folder/'method/q.json',{'complete':True})
    report=tmp_path/'report.md';report.write_text('Reviewed result',encoding='utf-8')
    stages=[];monkeypatch.setattr(module,'audit',lambda run:stages.append(run.name))
    module.completed_resume(tmp_path)
    assert stages==['qwen16','llava16','wfs32'] and report.read_text()=='Reviewed result'
    from videoqa_runtime.common import read_json
    result=read_json(tmp_path/'recovery/completed_resume.json')
    assert result['extra_generations']==0 and result['unchanged_files']==9
    assert result['preparation_replayed'] is False and result['report_rewritten'] is False
