"""时间传播固定对照的数值和隔离验收。"""
from fractions import Fraction
import math
import numpy as np
import pytest
from videoqa_audit.temporal_support_probe import temporal_support


def test_two_scans_match_direct_max():
    times=[Fraction(0),Fraction(1,3),Fraction(2),Fraction(10)]
    raw=[.2,None,.8,.1]
    expected=[max((r or 0)/.8*math.exp(-float(abs(t-s))/4) for s,r in zip(times,raw)) for t in times]
    assert np.allclose(temporal_support(times,raw),expected,atol=1e-15,rtol=0)


def test_zero_fallback_and_time_validation():
    assert np.array_equal(temporal_support([0,1],[0,None]),[1,1])
    with pytest.raises(ValueError):temporal_support([1,1],[.1,.2])
    with pytest.raises(ValueError):temporal_support([0,1],[.1,1.1])
