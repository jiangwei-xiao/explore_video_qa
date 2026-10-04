"""等权时间融合的精确目标、身份与退化行为测试。"""
from copy import deepcopy
from fractions import Fraction
from itertools import combinations
import pytest
from videoqa_methods.temporal_mixture import select_equal_temporal_mixture


def selection(times):
    """建立非均匀源PTS样例，不假定平均FPS。"""
    return dict(video=dict(path='clip',time_base='1',start_pts=0,duration_fraction='20'),
        selected_frames=[dict(source_pts=t,source_frame_index=t,timestamp_seconds=t,candidate_index=i)
                         for i,t in enumerate(times)])


def test_exact_dp_matches_exhaustive_search():
    """小集合枚举全部合法方案，验证整数DP及同分早帧顺序。"""
    _,trace=select_equal_temporal_mixture(selection([0,4,8]),selection([2,5,7]),3)
    den=trace['mass_denominator'];prefix=0;centers=[]
    for item in trace['union']:
        centers.append(Fraction(2*prefix+item['mass'],2*den));prefix+=item['mass']
    expected=min((sum(abs(centers[i]-Fraction(2*j+1,6)) for j,i in enumerate(ids)),ids)
                 for ids in combinations(range(6),3))
    assert Fraction(trace['exact_cost'])==expected[0]
    assert tuple(trace['selected_indices'])==expected[1]


def test_identical_inputs_preserved_without_mutation():
    """同源输入且预算足够时不制造差异，不修改旧元数据。"""
    u=selection([0,1,9]);before=deepcopy(u)
    new,trace=select_equal_temporal_mixture(u,u,32)
    assert [r['source_pts'] for r in new['selected_frames']]==[0,1,9]
    assert trace['shared_selected']==3 and u==before


def test_swapping_sources_preserves_physical_selection():
    """两路等权，交换标签不能改变实际选择。"""
    u,r=selection([0,3,8,12]),selection([2,3,11])
    a,_=select_equal_temporal_mixture(u,r,4);b,_=select_equal_temporal_mixture(r,u,4)
    assert [x['source_pts'] for x in a['selected_frames']]==[x['source_pts'] for x in b['selected_frames']]


def test_shared_source_not_duplicated():
    """共享帧有两路权重，但只占一个名额。"""
    out,trace=select_equal_temporal_mixture(selection([0,4]),selection([0,2,4]),4)
    assert len(out['selected_frames'])==3
    assert sum(x['mass'] for x in trace['union'])==trace['mass_denominator']


def test_conflicting_source_identity_rejected():
    """同源编号不得被不同PTS冒充成两帧。"""
    a,b=selection([0,4]),selection([1,4]);b['selected_frames'][0]['source_frame_index']=0
    with pytest.raises(ValueError):select_equal_temporal_mixture(a,b)
