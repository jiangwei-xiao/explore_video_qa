"""exp12全量适配的分区、恢复无生成和缓存身份边界。"""
from pathlib import Path
import pytest
from videoqa_budget32 import relevance_full as full
from videoqa_full.state import durable


def test_wait_accepts_completed_selection_after_process_exit(tmp_path):
    """中文说明：原准备完成且释放锁后，接续器不要求旧PID继续存活。"""
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_relevance_coverage_full import wait_for_selection
    durable(tmp_path/'selection_summary.json',dict(status='passed'))
    wait_for_selection(tmp_path,999999999)


def test_wait_rejects_missing_producer_without_completion(tmp_path):
    """中文说明：失去真实生产进程且无完成清单时，不能偷偷重启准备任务。"""
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_relevance_coverage_full import wait_for_selection
    with pytest.raises(RuntimeError,match='exited without completion'):
        wait_for_selection(tmp_path,999999999)


def test_frozen_partition_keeps_development_and_new_disjoint():
    data=[dict(question_id=str(i)) for i in range(2700)]
    old,new=full.partition(data,[str(i) for i in range(200)])
    assert len(old)==200 and len(new)==2500 and not set(old)&set(new)
    with pytest.raises(ValueError):full.partition(data,['missing'])


def test_completed_recovery_does_not_start_worker_or_gpu_checks(tmp_path,monkeypatch):
    data=[dict(question_id=str(i)) for i in range(2700)]
    ids=[str(i) for i in range(200,2700)]
    spec=dict(new_ids=ids,allowed_gpus=['0'],imported_ids=[str(i) for i in range(200)])
    monkeypatch.setattr(full,'verify',lambda run:spec)
    monkeypatch.setattr(full,'rows',lambda:data)
    monkeypatch.setattr(full,'validate',lambda *args:None)
    original=Path.glob
    def fake_glob(path,pattern):
        """中文说明：模拟已完整校验的2500份结果，不创建虚假模型回答文件。"""
        if path==tmp_path/'results'/full.METHOD:
            return iter([path/(q+'.json') for q in ids])
        return original(path,pattern)
    monkeypatch.setattr(Path,'glob',fake_glob)
    def forbidden(*args,**kwargs):
        """中文说明：完成恢复不能触发GPU资源等待或工作进程。"""
        raise AssertionError('GPU path called during completed recovery')
    monkeypatch.setattr(full,'worker',forbidden)
    from videoqa_runtime import baseline_run
    monkeypatch.setattr(baseline_run,'available_gpus',forbidden)
    full.execute(tmp_path)
    assert full.read_json(tmp_path/'recovery_status.json')['new_generations']==0
    assert full.read_json(tmp_path/'status.json')['completed_total']==2700


def test_foreign_call_id_is_rejected_before_recovery(tmp_path,monkeypatch):
    monkeypatch.setattr(full,'verify',lambda run:dict(new_ids=['new'],allowed_gpus=['0']))
    monkeypatch.setattr(full,'rows',lambda:[dict(question_id='new')])
    durable(tmp_path/'calls'/full.METHOD/'imported.json',dict(status='completed'))
    with pytest.raises(AssertionError):full.execute(tmp_path)
