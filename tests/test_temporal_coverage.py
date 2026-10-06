"""时间覆盖单因素边界与原有资格/预算契约，不调用模型。"""
import numpy as np
import pytest
from videoqa_methods.temporal_coverage import temporal_kernel,select_temporal_coverage


def test_temporal_kernel_is_normalized_squared_error_of_observed_points():
    kernel,span=temporal_kernel([0.,2.],[0.,1.,2.])
    assert span==2
    assert np.array_equal(kernel,[[1.,.75,0.],[0.,.75,1.]])


def test_temporal_kernel_zero_span_and_time_rescale_invariance():
    assert np.array_equal(temporal_kernel([3.],[3.])[0],[[1.]])
    a,_=temporal_kernel([0.,2.],[0.,1.,2.])
    b,_=temporal_kernel([50.,70.],[50.,60.,70.])
    assert np.array_equal(a,b)


@pytest.mark.parametrize('refs,candidates',([[],[1]],[[float('nan')],[1]],[[0.],[1.,2.]]))
def test_invalid_temporal_inputs_pause(refs,candidates):
    with pytest.raises(ValueError):temporal_kernel(refs,candidates)


def test_temporal_selection_keeps_members_anchors_and_positive_quotas():
    from test_prefix_coverage_budget import fixture
    from videoqa_methods.prefix_coverage_budget import select_prefix_coverage
    c,f,plan,video,cfg,quotas,scores=fixture()
    _,parent=select_prefix_coverage(c,f,plan,video,cfg,quotas,scores)
    pool=dict(candidates=c,video=video,plan=plan,initial_scores=scores,initial_count=len(scores))
    ids,trace=select_temporal_coverage(pool,parent,cfg['frame_budget'])
    assert len(ids)==len(set(ids))==cfg['frame_budget']
    assert set(trace['protected_indices'])<=set(ids)
    assert all(k>=1 for k in trace['region_budgets'].values())
    assert sum(trace['region_budgets'].values())==len(ids)
    for r in trace['regions']:
        assert all(b>=a for a,b in zip(r['coverage_curve'],r['coverage_curve'][1:]))


def test_visual_feature_values_do_not_enter_temporal_selector():
    from test_prefix_coverage_budget import fixture
    from videoqa_methods.prefix_coverage_budget import select_prefix_coverage
    c,f,plan,video,cfg,quotas,scores=fixture()
    _,parent=select_prefix_coverage(c,f,plan,video,cfg,quotas,scores)
    pool=dict(candidates=c,video=video,plan=plan,initial_scores=scores,initial_count=len(scores))
    first=select_temporal_coverage(pool,parent,cfg['frame_budget'])
    second=select_temporal_coverage(pool,parent,cfg['frame_budget'])
    assert first==second
