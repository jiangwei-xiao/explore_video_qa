"""时间字段诊断输入测试，不调用模型、不修改算法。"""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('time_field_repairs',Path(__file__).resolve().parents[1]/'scripts/run_time_field_repairs.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def bank():
    video=dict(path='synthetic.mp4',start_pts=0,time_base='1/30',duration_fraction='40',duration_seconds=40)
    frames=[dict(candidate_index=i,source_frame_index=i*30,source_pts=i*30,timestamp_seconds=float(i)) for i in range(36)]
    return dict(video=video,selected_frames=frames[:32]),dict(video=video,candidates=frames,initial_count=36)


def test_manual_repair_preserves_other_thirty_frames_and_count():
    old,pool=bank();result=m.repaired_selection(old,pool,dict(add=[32,33],remove=[30,31]))
    assert len(result['selected_frames'])==32
    assert result['selected_frames'][:30]==old['selected_frames'][:30]
    assert result['selected_frames'][-2:]==pool['candidates'][32:34]


@pytest.mark.parametrize('change',[
    dict(add=[32],remove=[35]),
    dict(add=[30],remove=[31]),
    dict(add=[36],remove=[31]),
    dict(add=[32,33],remove=[31]),
])
def test_invalid_source_or_budget_cannot_enter_generation(change):
    old,pool=bank()
    with pytest.raises(ValueError):m.repaired_selection(old,pool,change)


def test_population_has_exactly_three_predeclared_questions():
    assert list(m.CHANGES)==['555-2','560-2','599-1']
    assert [len(v['add']) for v in m.CHANGES.values()]==[2,1,1]
