"""邻帧配对的顺序、冲突、区域与稳定排序测试。"""
import itertools
import numpy as np
import pytest
import videoqa_methods.temporal_pair_selection as selection_module
from videoqa_methods.temporal_pair_selection import pair_representatives


def frames(n):
    return [dict(source_pts=i, candidate_index=i) for i in range(n)]


def test_sorted_pairs_and_anchor_preservation():
    """配对不能交错，且每组恰有一个代表。"""
    f=np.eye(8,dtype=np.float32)
    chosen,t=pair_representatives(frames(8),f,[1,5],[0]*8)
    assert chosen==[0,1,4,5]
    assert t['pairs']==[[0,1],[4,5]]


def test_representative_crowding_and_region_boundary():
    """代表挤占全部邻位或邻位在别区时，无解而不是重复源帧。"""
    assert pair_representatives(frames(5),np.eye(5),[1,2,3],[0]*5)[0] is None
    assert pair_representatives(frames(2),np.eye(2),[0],[0,1])[0] is None


def test_dp_matches_exhaustive():
    """小型输入穷举所有相邻合法配对，与DP总余弦最优值核对。"""
    rng=np.random.default_rng(2027);x=rng.normal(size=(9,4));x=(x/np.linalg.norm(x,axis=1,keepdims=True)).astype(np.float32)
    anchors=[1,4,7];chosen,trace=pair_representatives(frames(9),x,anchors,[0]*9)
    valid=[]
    for companions in itertools.product(*[[a-1,a+1] for a in anchors]):
        pairs=[tuple(sorted([a,b])) for a,b in zip(anchors,companions)]
        if any(pairs[i][1]>=pairs[i+1][0] for i in range(2)):continue
        valid.append(sum(float(np.clip(x[a]@x[b],-1,1)) for a,b in zip(anchors,companions)))
    assert abs(trace['coherence_sum']-max(valid))<1e-7 and len(chosen)==6


def test_storage_order_does_not_define_temporal_order():
    """粗细帧混合存储仍按PTS配对。"""
    c=[dict(source_pts=p,candidate_index=i) for i,p in enumerate([0,2,4,1,3,5])]
    result,t=pair_representatives(c,np.eye(6),[0,2],[0]*6)
    assert [c[i]['source_pts'] for i in result]==[0,1,3,4]


@pytest.mark.parametrize('n', [1, 11, 31, 32])
def test_short_pool_preserves_input_without_selecting(monkeypatch, n):
    """短池原样返回，不能为了组成帧对重新选帧或复制末帧。"""
    def forbidden(*args, **kwargs):
        raise AssertionError('短池不应运行代表选择')
    monkeypatch.setattr(selection_module, 'select_optional_regions', forbidden)
    original = list(range(n))
    chosen, trace = selection_module.select_temporal_pairs(
        {'candidates': frames(n)}, np.eye(n), original)
    assert chosen == original and chosen is not original
    assert trace['fallback'] and trace['reason'] == 'short_or_full_pool_unchanged'


@pytest.mark.parametrize('indices', [list(range(31)), [0] * 32,
                                   list(reversed(range(32))), list(range(1, 33))])
def test_invalid_fallback_is_rejected(indices):
    """回退也必须通过预算、唯一性、源索引及时间顺序校验。"""
    with pytest.raises(ValueError):
        selection_module.select_temporal_pairs({'candidates': frames(32)}, np.eye(32), indices)
