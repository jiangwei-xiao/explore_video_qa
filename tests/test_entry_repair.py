"""人工入口诊断的身份、时间与无排除一致性测试。"""
import numpy as np
import pytest
from videoqa_audit.entry_repair import replay_coarse
from videoqa_methods.density_selection import build_regions
from videoqa_audit.density_reward_decay import select_density_decay
from videoqa_audit.local_distance_scale import select_local_distance

def fixture():
    rows=[dict(candidate_index=i,source_pts=i*10+3,source_frame_index=i*30,timestamp_seconds=i+.3,requested_seconds=i+.25) for i in range(20)]
    f=np.eye(20,dtype=np.float32);scores=[.99-i*.02 for i in range(20)]
    video=dict(time_base='1/10',start_pts=0,duration_fraction='21')
    cfg=dict(seed_budget=4,frame_budget=6,core_gap_seconds=2,window_radius_seconds=4,refinement_min_core_seeds=3)
    return rows,scores,f,video,cfg

def test_empty_exclusion_replays_two_pass():
    rows,scores,f,v,cfg=fixture();p=build_regions(rows,scores,v,cfg);_,b=select_density_decay(rows,f,p,v,cfg)
    ids,_=select_local_distance(rows,f,p,v,cfg,b['region_budgets']);r=replay_coarse(rows,scores,f,v,cfg,[])
    assert ids==r['selected_original_indices'] and p==r['plan']

def test_filtered_identity_time_and_refill():
    args=fixture();r=replay_coarse(*args,[0,1,2]);assert r['seed_original_indices']==[3,4,5,6]
    assert not {0,1,2}&set(r['selected_original_indices'])
    assert r['selected_frames']==[args[0][i] for i in r['selected_original_indices']]
    assert r['new_candidates']==0 and not r['actual_refinement']

def test_gaps_not_compressed():
    args=fixture();r=replay_coarse(*args,[1,2,3,4]);p=r['plan']
    assert len(p['cores'])==2
    assert r['regions_original'][0]['seed_indices']==[0,5,6,7]

@pytest.mark.parametrize('ids',[[0,0],[-1],[20],list(range(20))])
def test_invalid_exclusion_refused(ids):
    with pytest.raises(ValueError):replay_coarse(*fixture(),ids)
