"""测试事实集合比较与人工索引展开，不运行视频或问答模型。"""
import runpy
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
compare=runpy.run_path(str(ROOT/'scripts/analyze_semantic_changes.py'))['fact_transitions']
indices=runpy.run_path(str(ROOT/'scripts/build_full_witness_spec.py'))['indices']


def test_remaining_witness_preserves_fact():
    """同一事实的一个见证被删，另一个还在时，不计整个事实损失。"""
    f={'id':'x','description':'fact','role':'required','limitation':'limit','witnesses':[{'source_pts':1},{'source_pts':2}]}
    r=compare([f],{1,2},{2,3})[0]
    assert r['state']=='retained' and r['before_pts']==[1,2] and r['after_pts']==[2]


def test_empty_witness_not_absence_proof():
    f={'id':'x','description':'unknown','role':'required','limitation':'not found','witnesses':[]}
    assert compare([f],{1},{2})[0]['state']=='not_witnessed_in_either'


def test_non_target_role_preserved():
    f={'id':'x','description':'later event','role':'non_target','limitation':'wrong time','witnesses':[{'source_pts':1}]}
    assert compare([f],set(),{1})[0]['role']=='non_target'


@pytest.mark.parametrize('bad',['0','3-1','1,1','1-3,3'])
def test_invalid_manual_range_rejected(bad):
    with pytest.raises(ValueError):indices(bad)


def test_explicit_range_expansion():
    assert indices('1-3,5')==[1,2,3,5] and indices('')==[]
