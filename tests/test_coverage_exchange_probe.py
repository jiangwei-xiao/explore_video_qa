"""覆盖交换与直接穷举一致，锚点和名额不改变。"""
import numpy as np
import pytest
from videoqa_audit.coverage_exchange_probe import best_single_exchange


def test_matches_exhaustive():
    """随机小矩阵逐一比对所有交换，不假定覆盖提升就是语义提升。"""
    rng = np.random.default_rng(2027)
    for _ in range(20):
        s = rng.random((7, 9));w = rng.random(7);w /= w.sum()
        selected = [0, 3, 5];anchor = 3
        old = float(w @ s[:, selected].max(axis=1))
        options = [(float(w @ s[:, [x for x in selected if x != r] + [a]].max(axis=1)), -a, -r)
                   for r in selected if r != anchor for a in range(9) if a not in selected]
        best = max(options)
        result, trace = best_single_exchange(s, w, selected, anchor)
        assert len(result) == len(set(result)) == 3 and anchor in result
        assert trace['after'] == pytest.approx(max(old, best[0]))
        if best[0] > old:
            assert trace['added_local'] == -best[1]
            assert trace['removed_local'] == -best[2]


def test_anchor_only_and_equal_noop():
    """仅锚点或目标完全持平时保持原输入。"""
    for selected in [[0], [0, 2]]:
        out, trace = best_single_exchange(np.ones((2, 4)), [.5, .5], selected, 0)
        assert out == selected and not trace['changed']


def test_invalid_input():
    """不接受负权重或重复源候选。"""
    with pytest.raises(ValueError):
        best_single_exchange(np.ones((2, 4)), [-1, 2], [0, 1], 0)
    with pytest.raises(ValueError):
        best_single_exchange(np.ones((2, 4)), [.5, .5], [0, 0], 0)
