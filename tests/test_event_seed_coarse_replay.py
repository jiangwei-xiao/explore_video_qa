"""入口只读重放回归：相同信号等价、旧分数不被事件分数覆盖。"""
import importlib.util
from pathlib import Path
import numpy as np
from videoqa_methods.budget_contract import budget_config,select_exp07
from videoqa_methods.density_selection import build_regions

spec=importlib.util.spec_from_file_location('event_seed_replay',Path(__file__).resolve().parents[1]/'scripts/replay_event_seed_coarse_selection.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    v=dict(path='synthetic',start_pts=0,time_base='1/1000',duration_fraction='41',duration_seconds=41)
    c=[dict(candidate_index=i,source_frame_index=i*30,source_pts=i*1000+250,requested_seconds=i+.25,timestamp_seconds=i+.25) for i in range(40)]
    s=np.linspace(1,0.01,40).tolist();plan=build_regions(c,s,v,budget_config(32))
    f=np.random.default_rng(2027).normal(size=(40,8)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True)
    return dict(video=v,candidates=c,initial_count=40,initial_scores=s,plan=plan),f


def test_identical_locator_signal_replays_original_selection():
    pool,f=fixture();expected,_=select_exp07(pool,f,32)
    actual,trace,plan=m.locator_only(pool,f,pool['initial_scores'])
    assert actual==expected and plan==pool['plan']
    assert len(actual)==32 and sum(trace['region_budgets'].values())==32


def test_alternative_locator_does_not_overwrite_original_support_scores():
    pool,f=fixture();old=list(pool['initial_scores'])
    selected,trace,plan=m.locator_only(pool,f,list(reversed(old)))
    assert pool['initial_scores']==old
    assert plan['seed_indices']!=pool['plan']['seed_indices']
    assert len(set(selected))==32
    assert set(trace['protected_indices'])<=set(selected)
