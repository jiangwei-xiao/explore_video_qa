"""共同选择链的行为等价与一次求解契约；纯CPU，不运行模型。"""
import copy
import inspect
import numpy as np
import pytest

from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_methods import rd_selection as new
from videoqa_methods.budget_contract import budget_config, select_exp07
from videoqa_methods.density_selection import build_regions
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage


def fixture(k=4, scores=None, times=None, requested=None, fine=(), identical=False, start=0, end=None):
    """测试接口：构造真实PTS及独立目标时间，可覆盖短池、VFR与长时间轴。"""
    times = [i+.25 for i in range(31)] if times is None else times
    requested = times if requested is None else requested
    video = dict(path='synthetic.mp4',start_pts=start, time_base='1/4', duration_fraction=str(end if end is not None else max(times)+5))
    candidates = [dict(candidate_index=i, source_frame_index=i, source_pts=start+round(t*4),
        timestamp_seconds=t, requested_seconds=r) for i,(t,r) in enumerate(zip(times,requested))]
    if scores is None:
        scores=[.1]*len(times)
        for i in [1,10,20,29]:
            if i<len(times):scores[i]=.9-i*.001
    pool=dict(video=video,candidates=candidates,initial_count=len(times),initial_scores=list(scores),
              plan=build_regions(candidates,scores,video,budget_config(k)))
    for t in fine:
        i=len(candidates);candidates.append(dict(candidate_index=i,source_frame_index=i,
            source_pts=start+round(t*4),timestamp_seconds=t,requested_seconds=t))
    f=np.ones((len(candidates),8),dtype=np.float32) if identical else np.random.default_rng(2027).random((len(candidates),8)).astype(np.float32)
    f/=np.linalg.norm(f,axis=1,keepdims=True)
    return pool,f


def legacy(pool,f,k,method):
    """测试接口：仅对照层调用冻结旧链；新生产接口无此依赖。"""
    chosen,prefix=select_exp07(pool,f,k)
    if method==new.METHODS[1]:chosen,trace=select_relevance_coverage(pool,f,prefix)
    else:trace=prefix
    return chosen,trace,prefix


def assert_equivalent(pool,f,k,method):
    """测试接口：离散量严格比较，数值量采用计划规定的绝对容差。"""
    old,trace,prefix=legacy(pool,f,k,method);picked,detail=new.select_rd(pool,f,k,method,'full')
    assert picked==old and detail['region_budgets']==trace['region_budgets']
    assert abs(detail['objective']-trace['objective'])<=1e-6
    for a,b,local in zip(detail['regions'],trace['regions'],prefix['regions']):
        for key in ['region_id','prefix','reference_coarse_indices','density_weight']:assert a[key]==b[key]
        assert a['members']==local['members']
        assert np.allclose(a['coverage_curve'],b['coverage_curve'],atol=1e-6,rtol=0)
        for x,y in zip(a['local_steps'],local['local_steps']):
            assert x['candidate_index']==y['candidate_index']
            for key in ['support','redundancy','local_score']:assert abs(x[key]-y[key])<=1e-6
    return picked,detail


@pytest.mark.parametrize('method',new.METHODS)
@pytest.mark.parametrize('level',[None,0.,.2,1.])
def test_random_and_uniform_score_equivalence(method,level):
    p,f=fixture(scores=None if level is None else [level]*31)
    assert_equivalent(p,f,4,method)


@pytest.mark.parametrize('method',new.METHODS)
def test_short_coarse_pool_keeps_larger_eligible_budget(method):
    p,f=fixture(k=8,times=[.25,1.25,2.25],fine=(.75,1.75))
    selected,trace=assert_equivalent(p,f,8,method)
    assert trace['seed_count']==3 and trace['actual_frame_count']==5 and len(selected)==5


@pytest.mark.parametrize('method',new.METHODS)
def test_zero_gain_fill_and_near_ties(method):
    for identical in [True,False]:
        p,f=fixture(identical=identical)
        if not identical:
            f[2]=f[1]+np.float32(1e-8);f[2]/=np.linalg.norm(f[2])
        assert_equivalent(p,f,4,method)


@pytest.mark.parametrize('method',new.METHODS)
@pytest.mark.parametrize('mode',['summary','full'])
def test_one_dp_no_legacy_selector_and_no_history_dependency(monkeypatch,method,mode):
    import videoqa_methods.budget_contract as contract
    import videoqa_methods.prefix_coverage_budget as prefix
    import videoqa_methods.relevance_coverage_budget as relevance
    p,f=fixture();gold,_,_=legacy(p,f,4,method)
    p['reference_quotas']={'do_not_read':'not a real quota'}
    calls=[];allocate=new.allocate_prefix_budget
    def spy(*a,**kw):calls.append(1);return allocate(*a,**kw)
    def forbidden(*a,**kw):raise AssertionError('Production called a full historical selector')
    monkeypatch.setattr(new,'allocate_prefix_budget',spy)
    monkeypatch.setattr(contract,'select_exp07',forbidden)
    monkeypatch.setattr(prefix,'select_prefix_coverage',forbidden)
    monkeypatch.setattr(relevance,'select_relevance_coverage',forbidden)
    selected,trace=new.select_rd(p,f,4,method,mode)
    assert selected==gold and len(calls)==1
    assert 'reference_quotas' not in trace and 'old_allocation_weighted_objective' not in trace
    assert ('regions' in trace)==(mode=='full')


def test_modes_and_inputs_are_unchanged_and_region_projection_is_minimal(monkeypatch):
    p,f=fixture();saved=copy.deepcopy(p);before=f.copy();fn=new.coarse_selection_cells
    def projected(plan,video):
        assert set(plan)=={'regions'}
        assert all(set(r)=={'region_id','left_fraction','right_fraction','seed_indices','anchor_index','weight'} for r in plan['regions'])
        return fn(plan,video)
    monkeypatch.setattr(new,'coarse_selection_cells',projected)
    a,x=new.select_rd(p,f,4,new.METHODS[1]);b,y=new.select_rd(p,f,4,new.METHODS[1],'full')
    assert a==b and x['region_budgets']==y['region_budgets'] and x['objective']==y['objective']
    assert p==saved and np.array_equal(f,before)
    assert list(inspect.signature(new.select_rd).parameters)==['pool','features','frame_budget','method_id','trace_mode']


def test_vfr_target_connection_is_not_replaced_with_seed_window_union():
    p,f=fixture(k=2,times=[.25,9.25],requested=[.25,1.25])
    _,trace=assert_equivalent(p,f,2,new.METHODS[1])
    assert len(trace['regions'])==1 and trace['regions'][0]['original_window']==['0','53/4']


@pytest.mark.parametrize('times,end', [([.25,8.25],None),([.25,1.25],1.25),([.25,32400.25],None)])
def test_shared_endpoint_clipping_and_long_int64_pts(times,end):
    p,f=fixture(k=2,times=times,end=end,start=2**33)
    for method in new.METHODS:assert_equivalent(p,f,2,method)


def test_connected_trigger_and_density_remain_distinct():
    for times,trigger in [([.25,1.25,2.25,7.25],True),([.25,1.25,6.25,7.25],False)]:
        p,f=fixture(k=4,times=times)
        assert p['plan']['regions'][0]['weight']==3 and p['plan']['regions'][0]['refine']==trigger
        assert_equivalent(p,f,4,new.METHODS[1])


def test_bad_budget_pool_duplicate_source_and_invalid_features_are_rejected():
    p,f=fixture()
    with pytest.raises(ProtocolError,match='seed plan'):new.select_rd(p,f,2,new.METHODS[0])
    bad=copy.deepcopy(p);bad['candidates'][1]['source_pts']=bad['candidates'][0]['source_pts']
    with pytest.raises(ProtocolError,match='Duplicate'):new.select_rd(bad,f,4,new.METHODS[0])
    for changed in [f*2,np.full_like(f,np.nan),f[:-1]]:
        with pytest.raises(ProtocolError,match='features'):new.select_rd(p,changed,4,new.METHODS[0])
    with pytest.raises(ProtocolError):new.select_rd(p,f,4,'RD-1.2')
