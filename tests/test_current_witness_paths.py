"""精确见证可达性测试，不把结构标签解释为语义判断。"""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('current_paths',Path(__file__).resolve().parents[1]/'scripts/analyze_current_witness_paths.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def trace():
    """接口：构造四帧总预算、两区各两帧的固定前缀。"""
    return dict(regions=[dict(region_id=0,reference_coarse_indices=[0,1,2,3,4],prefix=[0,1,2,3]),
                         dict(region_id=1,reference_coarse_indices=[5,6],prefix=[5,6])],
                region_budgets={'0':2,'1':2})


@pytest.mark.parametrize('index,state',[(0,'exact_witness_selected'),(2,'quota_reachable_but_unselected'),
    (3,'beyond_feasible_quota'),(4,'outside_saved_prefix')])
def test_path_states(index,state):
    assert module.witness_path(index,trace(),{0,1,5,6})['structural_state']==state


def test_wrong_selected_set_rejected():
    with pytest.raises(ValueError,match='Selection'):
        module.witness_path(2,trace(),{0,2,5,6})


def test_missing_owner_rejected():
    with pytest.raises(ValueError,match='exactly one'):
        module.witness_path(9,trace(),{0,1,5,6})
