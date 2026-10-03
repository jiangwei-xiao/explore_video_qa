"""种子子集重建必须保持源帧和特征对应，不能扩大模型调用。"""
import copy
from fractions import Fraction
import numpy as np
import pytest
from videoqa_methods.density_selection import build_regions,refinement_targets
from videoqa_methods.retarget_seed_pool import retarget_pool


def fixture():
    """接口：构造含完整2FPS映射的小池；原视频时钟用百分之一秒刻度。"""
    video=dict(start_pts=0,time_base='1/100',duration_fraction='8')
    coarse=[dict(candidate_index=i,source_pts=100*i+25,source_frame_index=100*i+25,
        source_seconds=i+.25,timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(8)]
    scores=[.9-.05*i for i in range(8)]
    cfg=dict(seed_budget=8,frame_budget=32,core_gap_seconds='2',window_radius_seconds='1',
        refinement_min_core_seeds=3,fine_fps=2,coarse_fps=1,phase_fraction='1/4')
    plan=build_regions(coarse,scores,video,cfg);candidates=copy.deepcopy(coarse);records=[]
    for region in plan['regions']:
        mappings=[]
        for t in refinement_targets(region,video,cfg):
            i=len(candidates);pts=int(t*100)
            candidates.append(dict(candidate_index=i,source_pts=pts,source_frame_index=pts,
                source_seconds=float(t),timestamp_seconds=float(t),requested_seconds=float(t),requested_fraction=str(t),region_id=region['region_id']))
            mappings.append(dict(target_fraction=str(t),status='added',candidate_index=i,source_pts=pts,source_frame_index=pts))
        records.append(dict(region_id=region['region_id'],mappings=mappings))
    features=np.random.default_rng(2027).random((len(candidates),6)).astype(np.float32)
    features/=np.linalg.norm(features,axis=1,keepdims=True)
    return dict(video=video,candidates=candidates,initial_count=8,initial_scores=scores,plan=plan,refinement=records),features,cfg


def test_control_identity_and_subset_features():
    pool,f,cfg=fixture();before=copy.deepcopy(pool)
    control,cf=retarget_pool(pool,f,cfg)
    assert control['candidates']==pool['candidates'] and np.array_equal(cf,f)
    new,nf=retarget_pool(pool,f,dict(cfg,seed_budget=4))
    assert new['new_candidate_count']==4 and new['plan']['seed_indices']==[0,1,2,3]
    assert np.array_equal(nf,f[new['original_candidate_indices']]) and pool==before


def test_condition_disables_refinement():
    pool,f,cfg=fixture();new,nf=retarget_pool(pool,f,dict(cfg,seed_budget=2))
    assert new['new_candidate_count']==0 and np.array_equal(nf,f[:8])


def test_missing_mapping_is_not_silently_recomputed():
    pool,f,cfg=fixture();pool['refinement'][0]['mappings'].pop(0)
    with pytest.raises(ValueError,match='Missing frozen'):retarget_pool(pool,f,dict(cfg,seed_budget=4))


def test_actual_pts_outside_new_endpoint_is_dropped():
    pool,f,cfg=fixture();index=11
    pool['candidates'][index].update(source_pts=376,source_frame_index=376,source_seconds=3.76,timestamp_seconds=3.76)
    pool['refinement'][0]['mappings'][3].update(source_pts=376,source_frame_index=376)
    new,_=retarget_pool(pool,f,dict(cfg,seed_budget=4,window_radius_seconds='1/2'))
    assert index not in new['original_candidate_indices']
    assert any(m['status']=='outside_new_window' for r in new['refinement'] for m in r['mappings'])
