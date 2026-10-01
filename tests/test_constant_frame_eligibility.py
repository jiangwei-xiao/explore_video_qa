"""恒定RGB资格单因素边界、无禁用等价与惰性检查测试，不加载模型。"""
import numpy as np
import pytest
from videoqa_methods.constant_frame_eligibility import pixel_statistics,valid_seed_plan,select_eligible,select_checked
from videoqa_methods.density_selection import build_regions
from videoqa_methods.option_seed_entry import select_option_pool


def fixture():
    """固定有理数PTS与可重复归一化特征，覆盖跨区及同分。"""
    video=dict(start_pts=0,time_base='1/4',duration_fraction='20')
    c=[dict(candidate_index=i,source_frame_index=i,source_pts=4*i+1,timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(18)]
    cfg=dict(seed_budget=4,frame_budget=8,core_gap_seconds='2',window_radius_seconds='1',refinement_min_core_seeds=3)
    scores=np.random.default_rng(2027).random(18).tolist()
    f=np.random.default_rng(1).normal(size=(18,8)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True)
    return c,video,cfg,scores,f


@pytest.mark.parametrize('value',[0,6,255])
def test_exact_scalar_constant(value):
    assert pixel_statistics(np.full((7,9,3),value,np.uint8))['constant_rgb']


def test_dark_content_single_pixel_and_color_not_rejected():
    a=np.zeros((7,9,3),np.uint8);a[1,1,0]=1
    assert not pixel_statistics(a)['constant_rgb']
    assert not pixel_statistics(np.tile(np.array([0,0,1],np.uint8),(7,9,1)))['constant_rgb']
    with pytest.raises(ValueError):pixel_statistics(a.astype(np.float32))


def test_no_invalid_seeds_reproduce_original_geometry():
    c,v,cfg,s,f=fixture()
    inspect=lambda ids:{i:dict(constant_rgb=False) for i in ids}
    plan,trace=valid_seed_plan(c,s,v,cfg,inspect)
    assert plan==build_regions(c,s,v,cfg) and not trace['rejected']


def test_seed_replacement_and_all_invalid_stop():
    c,v,cfg,s,f=fixture();order=sorted(range(len(c)),key=lambda i:-s[i]);bad=set(order[:3])
    inspect=lambda ids:{i:dict(constant_rgb=i in bad) for i in ids}
    plan,_=valid_seed_plan(c,s,v,cfg,inspect)
    assert set(plan['seed_indices'])==set(order[3:7])
    with pytest.raises(ValueError,match='No nonconstant'):valid_seed_plan(c,s,v,cfg,lambda ids:{i:dict(constant_rgb=True) for i in ids})


def test_fewer_valid_seeds_never_repeat():
    c,v,cfg,s,f=fixture();good={2,7}
    plan,trace=valid_seed_plan(c,s,v,cfg,lambda ids:{i:dict(constant_rgb=i not in good) for i in ids})
    assert set(plan['seed_indices'])==good and trace['effective_seed_budget']==2


@pytest.mark.parametrize('tie',[False,True])
def test_unmasked_matches_frozen_selector(tie):
    c,v,cfg,s,f=fixture()
    if tie:s=[.5]*len(c);f=np.eye(len(c),dtype=np.float32)
    pool=dict(candidates=c,video=v,initial_count=len(c),initial_scores=s,plan=build_regions(c,s,v,cfg))
    old,ot=select_option_pool(pool,f,cfg);new,nt=select_eligible(pool,f,cfg,set())
    assert old==new and ot['region_budgets']==nt['region_budgets'] and ot['objective']==nt['objective']


def test_mask_preserves_coverage_references_and_budget():
    c,v,cfg,s,f=fixture();pool=dict(candidates=c,video=v,initial_count=len(c),initial_scores=s,plan=build_regions(c,s,v,cfg))
    seeds=set(pool['plan']['seed_indices']);bad=set(range(len(c)))-seeds
    ids,trace=select_eligible(pool,f,cfg,bad)
    assert set(ids)==seeds and len(ids)==len(seeds)
    assert {i for r in trace['regions'] for i in r['reference_coarse_indices']}==set(range(len(c)))
    with pytest.raises(ValueError,match='protected'):select_eligible(pool,f,cfg,{next(iter(seeds))})


def test_lazy_rejection_rechecks_new_final_frames():
    c,v,cfg,s,f=fixture();pool=dict(candidates=c,video=v,initial_count=len(c),initial_scores=s,plan=build_regions(c,s,v,cfg))
    initial,_=select_eligible(pool,f,cfg,set());seeds=set(pool['plan']['seed_indices'])
    bad=next(i for i in initial if i not in seeds);calls=[]
    def inspect(ids):
        calls.append(list(ids));return {i:dict(constant_rgb=i==bad) for i in ids}
    ids,trace=select_checked(pool,f,cfg,inspect)
    assert bad not in ids and len(ids)==cfg['frame_budget'] and len(calls)==2
    assert set(calls[-1])==set(ids) and trace['banned_indices']==[bad]


def test_pixel_checkpoint_reuses_completed_decode(tmp_path,monkeypatch):
    """CPU恢复复用逐帧RGB记录；源身份或协议改变必须拒绝。"""
    from pathlib import Path
    import runpy
    from PIL import Image
    from videoqa_full.state import durable
    api=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/run_constant_frame_experiment.py'))
    c,v,cfg,s,f=fixture();pool=dict(candidates=c,video=v);durable(tmp_path/'protocol.json',{'test':1})
    calls=[]
    def decode(self):
        calls.append(1);return [Image.new('RGB',(2,2),(0,0,0)) for _ in self.selection['selected_frames']],['fake']*len(self.selection['selected_frames'])
    monkeypatch.setattr(api['FrameProvider'],'decode',decode)
    cls=api['PixelInspector'];first=cls(tmp_path,'q',pool);assert first([0])[0]['constant_rgb']
    second=cls(tmp_path,'q',pool);second([0]);assert len(calls)==1
    altered=dict(pool,candidates=[dict(c[0],source_frame_index=999),*c[1:]])
    with pytest.raises(AssertionError):cls(tmp_path,'q',altered)([0])
    durable(tmp_path/'protocol.json',{'test':2})
    with pytest.raises(AssertionError):cls(tmp_path,'q',pool)
