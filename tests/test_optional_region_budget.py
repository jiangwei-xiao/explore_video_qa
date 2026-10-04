"""零区域配额DP与小集合穷举核对。"""
from itertools import product
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('optional',Path(__file__).parents[1]/'scripts/analyze_optional_region_budget.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_matches_exhaustive_with_zero_budget_region():
    """允许区域不入选，不再要求区域数小于帧数。"""
    curves=[[0,.2,.3],[0,.9,1.],[0,.4,.5]];weights=[1,2,1]
    expected=max((sum(w*c[n] for w,c,n in zip(weights,curves,ns)),ns)
                 for ns in product(range(3),repeat=3) if sum(ns)==2)
    counts,value=module.optional_allocation(curves,weights,2)
    assert (value,tuple(counts))==expected and 0 in counts


def test_all_zero_ties_prefer_earlier_region():
    """代理目标相同时，保持原DP的较早区域优先约定。"""
    assert module.optional_allocation([[0,0,0],[0,0,0]],[1,1],2)==([2,0],0.)


def test_insufficient_capacity_rejected():
    """不能为了凑总预算重复候选。"""
    with pytest.raises(ValueError):module.optional_allocation([[0,.3]],[1],2)
