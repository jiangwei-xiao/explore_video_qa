"""密度奖励递减离线诊断的边界与作用范围测试，无模型调用。"""
import copy
import numpy as np
import pytest
from videoqa_audit.density_reward_decay import select_density_decay
from videoqa_audit.density_local_validation import supported_selection


def fixture(seed=2027):
    """生成两个区域，锚点各一；重复奖励与视觉内容分别可控。"""
    rng=np.random.default_rng(seed);f=rng.normal(size=(12,8)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True)
    rows=[dict(candidate_index=i,source_pts=i,source_frame_index=i,timestamp_seconds=float(i)) for i in range(12)]
    plan={'regions':[dict(region_id=0,left_fraction='0',right_fraction='5',anchor_index=0,seed_indices=[0,1,2,3,4],weight=5),
                     dict(region_id=1,left_fraction='6',right_fraction='11',anchor_index=6,seed_indices=[6],weight=1)]}
    return rows,f,plan,dict(start_pts=0,time_base='1',duration_fraction='12'),dict(frame_budget=10)


def test_first_proposals_equal_original_soft():
    a=fixture();_,old=supported_selection(*a);_,new=select_density_decay(*a)
    assert new['rounds'][0]['winner']['candidate_index']==old['rounds'][0]['winner']['candidate_index']
    for x,y in zip(old['rounds'][0]['proposals'],new['rounds'][0]['proposals']):assert x['priority']==y['priority']


def test_weight_decays_only_after_region_receives_frames():
    _,t=select_density_decay(*fixture());counts={0:1,1:1}
    for step in t['rounds']:
        for p in step['proposals']:
            n=counts[p['region_id']];assert p['selected_before']==n
            assert p['effective_weight']==1+(p['weight']-1)/n
            assert 1<=p['effective_weight']<=p['weight']
            assert p['priority']==p['effective_weight']*p['utility']
        counts[step['winner']['region_id']]+=1


def test_unit_weights_are_unchanged():
    a=fixture();
    for r in a[2]['regions']:r['weight']=1
    assert select_density_decay(*a)[0]==supported_selection(*a)[0]


def test_one_region_selection_order_unchanged():
    rows,f,_,video,cfg=fixture();p={'regions':[dict(region_id=0,left_fraction='0',right_fraction='11',anchor_index=0,seed_indices=[0,1,2],weight=3)]}
    assert select_density_decay(rows,f,p,video,cfg)[0]==supported_selection(rows,f,p,video,cfg)[0]


def test_unique_cap_and_anchor_protection():
    a=fixture();a[-1]['frame_budget']=20;ids,t=select_density_decay(*a)
    assert ids==list(range(12)) and set(t['protected_indices'])<=set(ids)


def test_zero_gain_uses_stable_time_order():
    a=fixture();a[1][:]=0;a[1][:,0]=1;ids,t=select_density_decay(*a)
    assert [s['winner']['candidate_index'] for s in t['rounds']]==[1,2,3,4,5,7,8,9]


def test_invalid_features_rejected():
    a=fixture();a[1][1]=0
    with pytest.raises(ValueError):select_density_decay(*a)
