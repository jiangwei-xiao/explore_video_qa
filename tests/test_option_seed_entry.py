"""完整入口单因素边界与纯视觉恢复测试；不加载任何真实模型。"""
from pathlib import Path
import runpy
import numpy as np
import pytest
from videoqa_methods.option_seed_entry import option_seed_plan,select_option_pool
from videoqa_audit.option_coarse_replay import replay_coarse
from videoqa_methods.density_selection import build_regions
from videoqa_full.state import durable


def fixture():
    video=dict(start_pts=0,time_base='1/4',duration_fraction='12')
    c=[dict(candidate_index=i,source_frame_index=i,source_pts=4*i+1,timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(10)]
    cfg=dict(seed_budget=3,frame_budget=4,core_gap_seconds='2',window_radius_seconds='1',refinement_min_core_seeds=3)
    return c,video,cfg


def test_equal_routes_exact_original_plan():
    c,v,cfg=fixture();s=np.linspace(0,1,10).tolist()
    plan,_=option_seed_plan(c,s,[s]*4,v,cfg)
    assert plan==build_regions(c,s,v,cfg)


def test_full_selector_same_as_coarse_replay_when_no_added_frames():
    c,v,cfg=fixture();s=np.linspace(0,1,10).tolist();f=np.eye(10,dtype=np.float32)
    routes=np.random.default_rng(2027).random((4,10)).tolist()
    plan,_=option_seed_plan(c,s,routes,v,cfg)
    pool=dict(candidates=c,initial_scores=s,initial_count=10,plan=plan,video=v)
    ids,trace=select_option_pool(pool,f,cfg)
    expected=replay_coarse(c,f,s,plan['seed_indices'],v,cfg)
    assert ids==expected['selected_indices'] and trace['objective']==expected['trace']['objective']


def visual_api():
    return runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/run_option_seed_experiment.py'))['visual_once']


class FakeScorer:
    def __init__(self):self.calls=0
    def visual_batch(self,images):
        self.calls+=1
        return np.ones((len(images),2),dtype=np.float32)/np.sqrt(2),.1,.2


def test_completed_visual_recovery_does_not_call_model(tmp_path):
    durable(tmp_path/'protocol.json',{})
    fn=visual_api();model=FakeScorer();args=(tmp_path,'q',0)
    values=fn(*args,model,[None],[{'sha256':'abc'}],'0')
    restored=fn(*args,None,[None],[{'sha256':'abc'}],'0')
    assert np.array_equal(values,restored) and model.calls==1


def test_changed_visual_identity_stops(tmp_path):
    durable(tmp_path/'protocol.json',{})
    fn=visual_api();model=FakeScorer();fn(tmp_path,'q',0,model,[None],[{'sha256':'abc'}],'0')
    with pytest.raises(AssertionError):fn(tmp_path,'q',0,model,[None],[{'sha256':'abc'}],'1')
    assert model.calls==1


def test_uncertain_visual_forward_is_not_retried(tmp_path):
    durable(tmp_path/'protocol.json',{})
    class Failed:
        calls=0
        def visual_batch(self,images):
            self.calls+=1;raise RuntimeError('Interrupted return')
    fn=visual_api();model=Failed()
    with pytest.raises(RuntimeError):fn(tmp_path,'q',0,model,[None],[{'sha256':'abc'}],'0')
    with pytest.raises(AssertionError):fn(tmp_path,'q',0,model,[None],[{'sha256':'abc'}],'0')
    assert model.calls==1
