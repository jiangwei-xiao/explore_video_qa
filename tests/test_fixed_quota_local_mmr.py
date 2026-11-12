"""实际配额、原生MMR和调用隔离的CPU回归，不加载模型或历史标签。"""
import copy
import importlib.util
import os
from pathlib import Path
import numpy as np
import pytest
from videoqa_audit import fixed_quota_local_mmr as m
from test_selection_structure import inputs


@pytest.fixture
def reference():
    """接口：只使用固定SHA官方文件，缺失时明确跳过。"""
    p=Path(os.environ.get('WFS_REFERENCE_TEST_PATH',m.ROOT/'third_party/WFS-SB-reference/core.py'))
    if not p.exists():pytest.skip('Pinned official source unavailable')
    m.reference_core(p);return p


def trace(pool,groups,quotas):
    """接口：构造分区及父前缀，故意加入不同的旧可行参照以检查读取口径。"""
    scores=pool['initial_scores'];regions=[];anchors=[]
    for i,ids in enumerate(groups):
        a=max(ids,key=lambda x:scores[x]);anchors.append(a)
        regions.append(dict(region_id=i,members=ids,reference_coarse_indices=ids,prefix=[a]+[x for x in ids if x!=a]))
    return dict(coarse_prefix_trace=dict(regions=regions),region_budgets={str(i):q for i,q in enumerate(quotas)},
        protected_indices=anchors,reference_quotas={'0':999},feasibility_reference_only=True)


@pytest.mark.parametrize('n,k,groups,qs',[(1,8,[[0]],[1]),(7,8,[list(range(7))],[7]),
    (8,8,[[0,2,4,6],[1,3,5,7]],[4,4]),(20,8,[list(range(10)),list(range(10,20))],[1,7])])
def test_exact_native_per_region(n,k,groups,qs,reference):
    """接口：与独立官方直接调用比对，覆盖配额1、等容量、短池及不连续成员映射。"""
    pool,f=inputs(n);t=trace(pool,groups,qs);old=copy.deepcopy((pool,t));fb=f.copy()
    got,record=m.select_region_mmr_fixed_quota(pool,f,t,k,_reference=reference)
    core=m.reference_core(reference);expected=[]
    for ids,q in zip(groups,qs):
        ids=sorted(ids);raw=core.WFSFrameSelector(.5).select_from_segment((0,len(ids)),q,2*np.array(pool['initial_scores'])[ids]-1,f[ids])
        expected.extend(ids[i] for i in raw)
    assert got==sorted(expected);assert record['dp_calls']==0
    assert (pool,t)==old;np.testing.assert_array_equal(f,fb)


def test_negative_cosine_and_native_tie(reference):
    """接口：负余弦奖励及首个同分候选必须沿用参考实现。"""
    pool,_=inputs(4);pool['initial_scores']=[.9,.9,.5,.5]
    f=np.array([[1,0],[1,0],[-1,0],[0,1]],np.float32);t=trace(pool,[[0,1,2,3]],[2])
    got,record=m.select_region_mmr_fixed_quota(pool,f,t,2,_reference=reference)
    assert got==[0,2];assert record['regions'][0]['steps'][1]['maximum_selected_cosine']==-1


@pytest.mark.parametrize('damage',['missing_actual','anchor','overlap','capacity','zero','budget','bad_prefix','feature'])
def test_reject_noncontract_inputs(damage,reference):
    """接口：参照配额、错误锚点、重叠成员及错位特征不得被回退掩盖。"""
    pool,f=inputs(10);t=trace(pool,[list(range(5)),list(range(5,10))],[2,2])
    if damage=='missing_actual':del t['region_budgets']
    if damage=='anchor':
        r=t['coarse_prefix_trace']['regions'][0];r['prefix']=list(reversed(r['prefix']));t['protected_indices'][0]=r['prefix'][0]
    if damage=='overlap':t['coarse_prefix_trace']['regions'][1]['members'][0]=0
    if damage=='capacity':t['region_budgets']['0']=6
    if damage=='zero':t['region_budgets']['0']=0
    if damage=='budget':t['region_budgets']['0']=1
    if damage=='bad_prefix':t['coarse_prefix_trace']['regions'][0]['prefix']=[0,0]
    if damage=='feature':f=f[:-1]
    with pytest.raises(ValueError):m.select_region_mmr_fixed_quota(pool,f,t,4,_reference=reference)


def test_fixed_population_and_reference_order():
    """接口：来源优先级P→G→W，阶段顺序两输入先冻结且LL先于Q，恢复显式传递。"""
    from videoqa_audit.fixed_quota_local_mmr_runtime import histories,validate_call_population
    import tempfile
    assert [p.parent.name for p in histories('llava8','128-1')]==[m.PARENT,m.GLOBAL,m.WFS]
    p=m.ROOT/'scripts/run_fixed_quota_local_mmr.py';spec=importlib.util.spec_from_file_location('fixed_mmr_entry',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cs=module.commands('test',['0','1','2','3'])
    assert [s for s,c in cs]==['offline','prepare','prepare','freeze-calls','answer','answer','report','panel','audit']
    assert all('--resume' in c for s,c in cs)
    with tempfile.TemporaryDirectory() as d:
        spec=dict(max_new_calls=1,new_groups={'q':[m.CONTROL]})
        assert validate_call_population(d,spec)=={(m.CONTROL,'q')}
        with pytest.raises(ValueError):validate_call_population(d,dict(max_new_calls=1,new_groups={'q':[m.GLOBAL]}))
