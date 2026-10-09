"""文字状态重复度：默认阈值、方向性、纯视觉等价及预算保护。"""
import numpy as np
import pytest
from test_region_refinement_scope import make_pool
from videoqa_methods.budget_contract import select_exp07
from videoqa_methods.text_state_redundancy import normalize_text_state,text_similarity,text_attenuation,select_text_state


def test_default_ocr_threshold_and_chord_numbers_preserved():
    assert normalize_text_state([dict(text='CM7 B7 Banks',score=.5),dict(text='fake',score=.49)])==['b7','banks','cm7']
    with pytest.raises(ValueError):normalize_text_state([dict(text='x',score=float('nan'))])


def test_text_similarity_and_missing_detection_not_rewarded():
    assert text_similarity([],[])==1
    assert text_similarity(['a'],[])==0
    assert text_similarity(['a','b'],['b','c'])==1/3
    assert text_attenuation([],['a'])==1
    assert text_attenuation(['a'],[])==0


def test_empty_and_identical_text_reproduce_parent_input():
    p,f=make_pool(seed_times=(2.25,3.25,10.25,11.25),fine=())
    before,t=select_exp07(p,f,4)
    for tokens in ([],['same','logo']):
        after,trace=select_text_state(p,f,t,[tokens for _ in f],4)
        assert after==before and trace['region_budgets']==t['region_budgets']
        assert trace['protected_indices']==t['protected_indices']


def test_new_text_state_can_beat_visual_duplicate_with_fixed_quota():
    p,_=make_pool(seed_times=(2.25,3.25,4.25,10.25),fine=())
    # 真实问题/答案均不进入此合成竞争：同外观，但一个候选有独立文本状态。
    f=np.ones((len(p['candidates']),1),dtype=np.float32);before,t=select_exp07(p,f,4)
    states=[['same'] for _ in f];index=15;states[index]=['new','state']
    after,trace=select_text_state(p,f,t,states,4)
    assert index in after and index not in before
    assert len(after)==len(set(after))==4 and trace['region_budgets']==t['region_budgets']
    assert set(t['protected_indices'])<=set(after)


def test_joint_similarity_is_computed_before_max_over_references():
    # 这不是max(视觉)×max(文字)，两个最大值可能来自不同参照。
    visual=np.array([.9,.4]);text=np.array([.1,1.])
    assert float(np.max(visual*text))==.4
    assert float(np.max(visual))*float(np.max(text))==.9
