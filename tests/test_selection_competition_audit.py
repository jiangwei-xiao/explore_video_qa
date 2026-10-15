"""冻结竞争补审：真实胜出分量、配额边界与错误特征检测。"""
import copy
import numpy as np
import pytest
from videoqa_audit.selection_competition_audit import audit_competition


def fixture():
    """测试接口：三张已归一化视觉状态，原前缀和两配额都有明确手算预期。"""
    f=np.asarray([[1.,0.],[.8,.6],[0.,1.]],dtype=np.float32)
    candidates=[dict(source_frame_index=i,source_pts=i,timestamp_seconds=float(i)) for i in range(3)]
    pool=dict(candidates=candidates,initial_scores=[1.,.9,.8],
              plan=dict(regions=[dict(region_id=0,seed_indices=[0])]))
    score=.8*np.sqrt(.2)
    region=dict(region_id=0,reference_coarse_indices=[0,1,2],prefix=[0,1,2],
                local_steps=[dict(candidate_index=1,support=.8,redundancy=.8,local_score=score),
                             dict(candidate_index=2,support=0.,redundancy=.6,local_score=0.)])
    a=dict(regions=[region],region_budgets={'0':2})
    b=dict(regions=[region],region_budgets={'0':3})
    return pool,f,a,b


def test_actual_competition_and_two_quota_endpoints():
    """检验同一目标在两个实际配额终点的胜者与保留状态，而非只检查输出形状。"""
    pool,f,a,b=fixture();out=audit_competition(pool,f,a,b,[0,2])
    anchor,target=out['records']
    assert anchor['parents']['exp07']['terminal']['mode']=='protected_anchor'
    assert target['parents']['exp07']['terminal']['mode']=='lost_at_last_available_slot'
    assert target['parents']['exp07']['terminal']['competition']['actual_winner']['candidate_index']==1
    assert target['parents']['exp12']['retained']
    assert target['parents']['exp12']['representative_index']==2
    assert target['parents']['exp07']['semantic_equivalence']=='not_inferred'


def test_anchor_only_quota_has_no_fabricated_competition():
    """检验只有保护锚点时不存在补充槽位，不能伪造一次选帧竞争。"""
    pool,f,a,b=fixture();a['region_budgets']['0']=1
    out=audit_competition(pool,f,a,b,[2])['records'][0]
    assert out['parents']['exp07']['terminal']['competition'] is None


def test_feature_mapping_corruption_stops_audit():
    """检验错位特征必须被冻结分量比对拒绝，不能继续解释为机制变化。"""
    pool,f,a,b=fixture()
    with pytest.raises(ValueError,match='component mismatch'):
        audit_competition(pool,f[[0,2,1]],a,b,[2])


def test_zero_gain_fills_without_claiming_information_gain():
    """检验零增益填额仍能恢复，但不被记录为语义信息增益。"""
    pool,f,a,b=fixture();out=audit_competition(pool,f,a,b,[2])['records'][0]
    assert out['trajectory'][1]['target_local_score']==0
    assert out['parents']['exp12']['terminal']['competition']['actual_winner']['local_score']==0
    assert out['parents']['exp12']['semantic_equivalence']=='not_inferred'
