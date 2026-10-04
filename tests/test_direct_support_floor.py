"""直接支持下界的数值边界测试；不将其视为有效选帧方法。"""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('floor',Path(__file__).parents[1]/'scripts/probe_direct_support_floor.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def setup_case(scores):
    """构造三个单位向量的稳定局部次序，不需要视频或模型。"""
    f=np.asarray([[1.,0.],[.8,.6],[0.,1.]],dtype=np.float32)
    pool=dict(candidates=[dict(source_pts=i,candidate_index=i) for i in range(3)],initial_scores=scores,
              plan=dict(regions=[dict(region_id=0,seed_indices=[0],anchor_index=0)]))
    baseline=dict(coarse_prefix_trace=dict(regions=[dict(region_id=0,members=[0,1,2],prefix=[0,1,2],density_weight=1)]))
    return pool,f,baseline


def test_zero_scores_preserve_propagation():
    """原分数全零时不造出直接支持，退回原传播规则。"""
    args=setup_case([0.,0.,0.])
    old=module.prefix_records(*args,False)[0]
    new=module.prefix_records(*args,True)[0]
    assert old['prefix']==new['prefix']
    assert all(step['direct']==0 for step in new['steps'])


def test_floor_does_not_reduce_existing_support():
    """低原问题分数不压低已有支持，只验证公式定义。"""
    result=module.prefix_records(*setup_case([1.,.1,0.]),True)[0]
    assert all(s['support']>=s['propagated'] for s in result['steps'])
    assert result['steps'][0]['support']==result['steps'][0]['propagated']


def test_high_direct_score_can_raise_support():
    """自身分数高于传播时才抬升，仍保留原锚点且源帧唯一。"""
    result=module.prefix_records(*setup_case([1.,.95,.1]),True)[0]
    assert result['steps'][0]['support']>result['steps'][0]['propagated']
    assert result['prefix'][0]==0 and len(set(result['prefix']))==3
