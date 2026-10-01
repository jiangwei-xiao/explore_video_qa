"""帧组单位成本、时间边界、重叠计费与纯覆盖支配关系测试。"""
import numpy as np
import pytest
from videoqa_methods.seed_bundle_selection import propose_seed_bundles,select_seed_bundles


def fixture():
    c=[dict(candidate_index=i,source_frame_index=i,source_pts=4*i+1,timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(12)]
    v=dict(start_pts=0,time_base='1/4',duration_fraction='13')
    seeds=[2,4,6,9];scores=[.1]*12
    for i in seeds:scores[i]=.8
    p=dict(candidates=c,video=v,initial_count=12,initial_scores=scores,plan=dict(seed_indices=seeds,regions=[dict(region_id=0,anchor_index=4)]))
    f=np.random.default_rng(2027).uniform(.1,1,(12,6)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True)
    t=dict(regions=[dict(region_id=0,members=list(range(12)),reference_coarse_indices=list(range(12)),density_weight=3)])
    return p,f,t


def test_coverage_groups_dominated_by_best_singleton_per_cost():
    """非负最大相似覆盖为子模：组合增益不超过各单项增益之和。"""
    rng=np.random.default_rng(2027)
    for _ in range(50):
        sim=rng.random((13,9));covered=sim[:,:2].max(1);columns=[2,4,7]
        single=[np.maximum(covered,sim[:,j]).mean()-covered.mean() for j in columns]
        group=np.maximum(covered,sim[:,columns].max(1)).mean()-covered.mean()
        assert group<=sum(single)+1e-12 and group/len(columns)<=max(single)+1e-12


def test_bundle_endpoints_unique_temporal_and_bounded():
    p,f,t=fixture();groups,edges,details=propose_seed_bundles(p,f)
    assert groups and edges and abs(sum(e['weight'] for e in edges)-1)<1e-12
    for record in details:
        for partner in record['partners']:
            d=partner['index']-record['seed_index'];assert 0<abs(d)<=4
            assert (d<0)==(partner['side']=='before')
    assert all(len(g)==len(set(g)) and 2<=len(g)<=3 for g in groups)


@pytest.mark.parametrize('pairs',[False,True])
def test_budget_protection_and_unique_cost(pairs):
    p,f,t=fixture();ids,trace=select_seed_bundles(p,f,t,budget=7,with_pairs=pairs)
    assert len(ids)==len(set(ids))==7 and 4 in ids and ids==sorted(ids)
    assert sum(s['cost'] for s in trace['steps'])+1==7
    seen={4}
    for s in trace['steps']:
        assert len(s['added'])==s['cost'] and not set(s['added'])&seen
        seen.update(s['added'])
    if not pairs:assert trace['multi_frame_steps']==0
    assert trace['final_coverage']>=trace['initial_coverage']-1e-7


def test_boundary_and_flat_features_are_finite():
    p,f,t=fixture();f[:]=1/np.sqrt(f.shape[1])
    ids,trace=select_seed_bundles(p,f,t,budget=4)
    assert len(ids)==4 and all(np.isfinite(s['gain']) for s in trace['steps'])


def test_repeat_and_short_pool():
    p,f,t=fixture();a=select_seed_bundles(p,f,t,budget=32);b=select_seed_bundles(p,f,t,budget=32)
    assert a==b and len(a[0])==12


def test_pair_completion_has_complementarity():
    weights={(1,2):.6,(2,3):.4}
    value=lambda s:sum(w for pair,w in weights.items() if set(pair)<=s)
    assert value({1})==value({2})==0 and value({1,2})==.6
