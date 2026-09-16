"""诊断预算与输入保护测试；全部为合成数据，不运行任何模型。"""
import importlib.util
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_evidence_diagnostics import validate_case
from run_evidence_diagnostics import check_recovery,check_manifest
from videoqa_runtime.common import write_json


def test_same_pool_and_budget():
    """只允许原池内1至4帧替换。"""
    pool=[{'source_pts':i,'candidate_index':i} for i in range(24)]
    case={'original_frames':pool[:16],'repaired_frames':pool[1:17]}
    validate_case(case,pool)
    case['repaired_frames']=pool[5:21]
    with pytest.raises(ValueError):validate_case(case,pool)
    case['repaired_frames']=pool[:15]+[{'source_pts':99}]
    with pytest.raises(ValueError):validate_case(case,pool)


def test_duplicate_frame_rejected():
    """不能用重复帧填16帧预算。"""
    pool=[{'source_pts':i} for i in range(17)]
    with pytest.raises(ValueError):validate_case({'original_frames':pool[:16],'repaired_frames':pool[:15]+[pool[0]]},pool)


@pytest.mark.parametrize('status',['reserved_may_generate','failed_or_uncertain'])
def test_uncertain_call_never_repeated(tmp_path,status):
    """不确定或失败调用必须暂停，不能当作缺失结果重跑。"""
    write_json(tmp_path/'calls/q.original.json',{'status':status,'manifest_sha256':'x'})
    with pytest.raises(RuntimeError):check_recovery(tmp_path,'x')


def test_twenty_call_limit(tmp_path):
    """总调用记录超过20时直接拒绝恢复。"""
    for i in range(21):write_json(tmp_path/f'calls/{i}.json',{'status':'completed','manifest_sha256':'x'})
    with pytest.raises(ValueError):check_recovery(tmp_path,'x')


def test_unfrozen_input_rejected():
    """没有事前视觉确认不得启动诊断。"""
    with pytest.raises(ValueError):check_manifest({'cases':[],'status':'proposed'})


def test_completed_run_keeps_original_cost(tmp_path,monkeypatch):
    """完成态恢复不加载模型、不检查GPU且不覆盖历史墙钟。"""
    import run_evidence_diagnostics as module
    from videoqa_runtime.common import sha256
    manifest={'cases':[{'question_id':'q'}]}
    write_json(tmp_path/'manifest.json',manifest)
    write_json(tmp_path/'freeze.json',{'manifest_sha256':sha256(tmp_path/'manifest.json')})
    write_json(tmp_path/'pairs/q.json',{'status':'completed'})
    write_json(tmp_path/'summary.json',{'wall_seconds':12.34})
    before=(tmp_path/'summary.json').read_bytes()
    monkeypatch.setattr(module,'check_manifest',lambda _:None)
    def forbidden(*a,**kw):
        """任何GPU检查都说明完成态保护失效。"""
        raise AssertionError('GPU query should not run')
    monkeypatch.setattr(module.subprocess,'check_output',forbidden)
    module.run_diagnostics(tmp_path,['0'])
    assert (tmp_path/'summary.json').read_bytes()==before


@pytest.mark.parametrize('pending',[0,40])
def test_pilot_table_reports_actual_full_progress(pending):
    """首批子表不得误报已经完成的全量初审状态。"""
    from videoqa_audit.report import review_progress_line
    rows=[{'review_status':'pending' if i<pending else 'initial_reviewed'} for i in range(50)]
    line=review_progress_line(rows)
    assert f'{50-pending}/50' in line and f'待审{pending}题' in line
