"""等权恢复、同帧导入隔离及进程内适配绑定的边界检查。"""
import importlib.util
from pathlib import Path
import pytest
from videoqa_generalization import exp07_budget16 as experiment


def test_prefix_selection_recovers_source_order():
    """候选存储编号不等同真实时间顺序，配额恢复不得丢帧或重复。"""
    pool={'candidates':[dict(candidate_index=i,source_pts=16-i) for i in range(16)]}
    trace=dict(regions=[dict(region_id=0,prefix=list(range(8))),dict(region_id=1,prefix=list(range(8,16)))],region_budgets={'0':8,'1':8})
    assert experiment.selection_from_prefix(pool,trace)==list(reversed(range(16)))
    trace['region_budgets']['1']=7
    with pytest.raises(ValueError,match='quota'):experiment.selection_from_prefix(pool,trace)


def test_missing_import_never_generates(tmp_path,monkeypatch):
    """同帧导入文件缺失必须暂停，不能补发模型调用来填满结果数。"""
    def forbidden(*args):raise AssertionError('Imported question was sent to generation')
    monkeypatch.setattr(experiment,'_PROCESS',forbidden)
    with pytest.raises(ValueError,match='generation forbidden'):
        experiment.process(tmp_path,{'question_id':'same'},None,None,'0',{'new_ids':['changed']})


def test_changed_question_uses_original_once_path(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(experiment,'_PROCESS',lambda *args:calls.append(args))
    row={'question_id':'changed'};spec={'new_ids':['changed']}
    experiment.process(tmp_path,row,'backend','adapter','1',spec)
    assert len(calls)==1 and calls[0][1] is row and calls[0][-1] is spec


def test_scoped_bindings_restore_after_failure():
    runtime=experiment.runtime;before=(runtime.validate,runtime.worker,runtime.process)
    with pytest.raises(RuntimeError):
        with experiment.evaluation_scope():
            assert runtime.validate is experiment.validate and runtime.worker is experiment.worker
            assert runtime.process is experiment.process
            raise RuntimeError('Scope exit')
    assert (runtime.validate,runtime.worker,runtime.process)==before


def test_spawn_entry_installs_child_bindings(monkeypatch):
    seen=[]
    monkeypatch.setattr(experiment,'_WORKER',lambda *args:seen.append((experiment.runtime.validate,experiment.runtime.process,args)))
    experiment.worker('run','0',None,None,None)
    assert seen[0][:2]==(experiment.validate,experiment.process)


def test_each_stage_explicitly_resumes_without_wfs():
    path=Path(__file__).resolve().parents[1]/'scripts/run_exp07_budget16_full.py'
    spec=importlib.util.spec_from_file_location('exp07_full_entry',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    commands=module.pipeline_commands('frozen_run',['0','1'])
    assert [s for s,_ in commands]==['prepare','import_qwen','import_llava','synthetic_qwen','synthetic_llava','qwen16','llava16','report','audit']
    for stage,command in commands:
        assert command[-1]=='--resume' and command[command.index('--run-id')+1]=='frozen_run'
        assert 'wfs32' not in command and command[command.index('--stage')+1]==stage
