"""联合重复度同参照、时间边界、旧纯视觉控制及唯一预算测试。"""
import numpy as np
import pytest
from videoqa_methods.visual_temporal_redundancy import joint_redundancy,select_visual_temporal_redundancy


def test_identical_visual_state_has_time_aware_nonzero_opportunity():
    f=np.array([[1.,0.]],dtype=np.float32)
    assert joint_redundancy(f,[0.],[0.],f)[0]==1
    assert joint_redundancy(f,[4.],[0.],f)[0]==pytest.approx(np.exp(-1))
    assert joint_redundancy(f,[4.],[0.],f,None)[0]==1


def test_max_must_use_same_visual_and_time_reference():
    f=np.array([[1.,0.]],dtype=np.float32);s=np.array([[1.,0.],[0.,1.]],dtype=np.float32)
    actual=joint_redundancy(f,[0.],[100.,0.],s)[0]
    assert actual==pytest.approx(np.exp(-25))
    assert actual<1e-8


@pytest.mark.parametrize('scale',(0.,-1.,float('nan')))
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError):joint_redundancy(np.ones((1,2)),[0.],[0.],np.ones((1,2)),scale)


def test_disabled_time_reproduces_parent_selection_and_coverage():
    from test_prefix_coverage_budget import fixture
    from videoqa_methods.prefix_coverage_budget import select_prefix_coverage
    from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
    c,f,plan,video,cfg,quotas,scores=fixture()
    pool=dict(candidates=c,video=video,plan=plan,initial_scores=scores,initial_count=len(scores))
    _,prefix=select_prefix_coverage(c,f,plan,video,cfg,quotas,scores)
    expected,parent=select_relevance_coverage(pool,f,prefix)
    parent=dict(parent,regions=[dict(b,members=a['members']) for a,b in zip(prefix['regions'],parent['regions'])])
    actual,trace=select_visual_temporal_redundancy(pool,f,parent,cfg['frame_budget'],None)
    assert actual==expected
    assert trace['region_budgets']==parent['region_budgets']
    assert trace['objective']==parent['objective']


def test_joint_rule_preserves_actual_budget_and_original_anchors():
    from test_prefix_coverage_budget import fixture
    from videoqa_methods.prefix_coverage_budget import select_prefix_coverage
    from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
    c,f,plan,video,cfg,quotas,scores=fixture()
    pool=dict(candidates=c,video=video,plan=plan,initial_scores=scores,initial_count=len(scores))
    _,prefix=select_prefix_coverage(c,f,plan,video,cfg,quotas,scores)
    _,parent=select_relevance_coverage(pool,f,prefix)
    parent=dict(parent,regions=[dict(b,members=a['members']) for a,b in zip(prefix['regions'],parent['regions'])])
    ids,trace=select_visual_temporal_redundancy(pool,f,parent,cfg['frame_budget'],4)
    assert len(ids)==len(set(ids))==cfg['frame_budget']
    assert set(trace['protected_indices'])<=set(ids)
    assert trace['coverage_kind']=='original ITM-weighted pure visual coverage'
