"""弱信号干预上限和精确父选择回退。"""
import numpy as np
from test_region_refinement_scope import make_pool
from videoqa_methods.budget_contract import select_exp07
from videoqa_methods.bounded_text_state import select_bounded_text_state


def test_no_text_reproduces_parent_source_selection():
    p,f=make_pool(seed_times=(2.25,3.25,10.25,11.25),fine=())
    before,t=select_exp07(p,f,4);after,trace=select_bounded_text_state(p,f,t,[[] for _ in f],4)
    assert before==after and trace['winner'] is None


def test_only_one_last_slot_changes_and_all_other_frames_stay():
    p,_=make_pool(seed_times=(2.25,3.25,4.25,10.25),fine=())
    f=np.ones((len(p['candidates']),1),dtype=np.float32);before,t=select_exp07(p,f,4)
    states=[['same'] for _ in f];states[15]=['different']
    after,trace=select_bounded_text_state(p,f,t,states,4)
    assert len(set(after)-set(before))==1 and len(set(before)-set(after))==1
    assert len(after)==4 and set(t['protected_indices'])<=set(after)
    r=next(r for r in t['regions'] if r['region_id']==trace['winner']['region_id'])
    assert trace['winner']['removed_index']==r['prefix'][t['region_budgets'][str(r['region_id'])]-1]
