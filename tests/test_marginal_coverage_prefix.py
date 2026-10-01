"""局部覆盖边际序列的独立增量与稳定边界验证。"""
import numpy as np
from videoqa_methods.density_selection import build_regions,frame_key
from videoqa_methods.option_seed_entry import select_option_pool
from videoqa_methods.marginal_coverage_prefix import select_marginal_coverage_prefix


def make_fixture(equal=False):
    c=[dict(candidate_index=i,source_frame_index=i,source_pts=4*i+1,timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(15)]
    v=dict(start_pts=0,time_base='1/4',duration_fraction='16');cfg=dict(seed_budget=3,frame_budget=7,core_gap_seconds='2',window_radius_seconds='1',refinement_min_core_seeds=3)
    f=np.random.default_rng(2).random((15,8)).astype(np.float32)
    if equal:f[:]=1
    f/=np.linalg.norm(f,axis=1,keepdims=True);scores=[.1]*15;scores[1]=.9;scores[7]=.5;scores[13]=.2
    p=dict(candidates=c,video=v,initial_count=15,initial_scores=scores,plan=build_regions(c,scores,v,cfg))
    _,trace=select_option_pool(p,f,cfg);return p,f,trace


def test_every_local_step_maximizes_independent_marginal():
    p,f,t=make_fixture();selected,new=select_marginal_coverage_prefix(p,f,t)
    assert len(selected)==len(set(selected))==7
    assert {r['anchor_index'] for r in p['plan']['regions']}<=set(selected)
    for r in new['regions']:
        refs=r['reference_coarse_indices'];weights=np.array(r['reference_weights']);chosen=[];coverage=np.zeros(len(refs))
        for pos,i in enumerate(r['prefix']):
            if pos:
                def gain(j):return float(weights@np.maximum(np.clip(f[refs]@f[j],0,1)-coverage,0))
                best=max(gain(j) for j in r['members'] if j not in chosen)
                assert abs(gain(i)-best)<1e-6
            chosen.append(i);coverage=np.maximum(coverage,np.clip(f[refs]@f[i],0,1))
            assert abs(float(weights@coverage)-r['coverage_curve'][pos+1])<1e-6


def test_identical_features_fill_unique_with_temporal_tie():
    p,f,t=make_fixture(True);selected,new=select_marginal_coverage_prefix(p,f,t)
    assert len(selected)==len(set(selected))==7
    for r in new['regions']:
        rest=sorted(set(r['members'])-{r['prefix'][0]},key=lambda i:frame_key(p['candidates'][i]))
        assert r['prefix'][1:]==rest[:len(r['prefix'])-1]
