"""条件检索信号的数值边界与四路对称性；不将分数增量解释为概率。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
import pytest
from analyze_conditional_retrieval_signal import log_odds


def test_log_odds_boundary_and_monotonicity():
    values, clipped = log_odds([0, .1, .5, .9, 1])
    assert clipped == 2 and np.isfinite(values).all()
    assert np.all(np.diff(values) > 0) and values[2] == 0


def test_equal_query_has_zero_increment_and_option_permutation_is_invariant():
    question, _ = log_odds([.1, .5, .9])
    identical, _ = log_odds([[.1, .5, .9]] * 4)
    assert np.array_equal(identical - question[None, :], np.zeros((4, 3)))
    options, _ = log_odds([[.2, .4, .8], [.3, .7, .5], [.6, .5, .9], [.1, .9, .3]])
    assert np.array_equal((options - question).max(axis=0), (options[[3, 2, 0, 1]] - question).max(axis=0))


def test_invalid_probability_fails():
    for values in [[float('nan')], [-.01], [1.01]]:
        with pytest.raises(AssertionError):
            log_odds(values)
