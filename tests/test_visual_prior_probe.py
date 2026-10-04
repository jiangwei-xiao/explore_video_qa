"""四选项分布变化的数值与对称重编号约束。"""
import numpy as np
import pytest
from videoqa_audit.visual_prior_probe import distribution_shift


def test_shift_invariance_and_zero():
    """整体logit平移不改变结果；相同分布为零。"""
    a = np.array([2., 4., -5., 8.]);b = np.array([0., 3., 1., 4.])
    assert distribution_shift(a, a) == pytest.approx(0)
    assert distribution_shift(a+1000, b-1000) == pytest.approx(distribution_shift(a,b))
    assert distribution_shift(a[::-1],b[::-1]) == pytest.approx(distribution_shift(a,b))


def test_extreme_and_invalid():
    """极小概率保持有限，非法分布必须拒绝。"""
    assert np.isfinite(distribution_shift([1000,0,0,0],[0,1000,0,0]))
    with pytest.raises(ValueError): distribution_shift([1,2,3],[0,0,0,0])
    with pytest.raises(ValueError): distribution_shift([np.nan,0,0,0],[0,0,0,0])
