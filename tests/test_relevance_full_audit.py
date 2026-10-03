"""独立预算最优值检查器的可行性与非凹前缀测试。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from audit_relevance_full_selection import independent_optimum


def test_independent_value_for_weighted_nonconcave_curves():
    assert independent_optimum([[0,.6,.9],[0,.1,.7]],[1,2],3)==pytest.approx(2.)


def test_minimum_one_per_region_and_infeasible_budget():
    assert independent_optimum([[0,.6,.9],[0,.1,.7]],[1,2],2)==pytest.approx(.8)
    with pytest.raises(ValueError):independent_optimum([[0,.6],[0,.1]],[1,2],1)
    with pytest.raises(ValueError):independent_optimum([[0,.6],[0,.1]],[1,2],3)
