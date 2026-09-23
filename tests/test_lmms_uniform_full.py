"""正式均匀入口范围与只读帧提供验收，不加载模型。"""
import pytest
from videoqa_lmms_baseline.run import METHOD
from videoqa_full.state import pilot_ids
from videoqa_lmms_bridge.frames import native_indices


def test_only_uniform_method():
    assert METHOD=='BASE-Uniform'


def test_pilot_includes_short_edges():
    rows=[dict(question_id=f'00{i}-1') for i in range(1,6)]
    assert pilot_ids(rows)==[r['question_id'] for r in rows]+['133-1','134-1']


@pytest.mark.parametrize('n',[1,11,14,16,17,90000])
def test_native_cap(n):
    ids=native_indices(n)
    assert len(ids)==len(set(ids))==min(16,n) and ids==sorted(ids)
