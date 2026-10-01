"""严格语法范围与固定配额支持消融边界；不使用真实答案。"""
from copy import deepcopy
import numpy as np
from videoqa_audit.option_structure_support import option_structure,fixed_quota_support
from videoqa_methods.density_selection import build_regions


def test_integer_counts_only():
    assert option_structure('How many athletes jump?',['A. 3.','B. 2.','C. 4.','D. 5.'])=='integer_count_options'
    assert option_structure('What year was it?',['A. 1990.','B. 1991.','C. 1992.','D. 1993.'])=='other'
    assert option_structure('How many objects?',['A. 3 red ones.','B. 2 blue ones.','C. 4 black ones.','D. 5.'])=='other'
    assert option_structure('How many objects?',['A. Three.','B. Two.','C. Four.','D. Five.'])=='other'


def test_references_need_complete_question_definitions():
    opts=['A. (a)(b)(c).','B. (b)(a)(c).','C. (c)(b)(a).','D. (c)(a)(b).']
    assert option_structure('Order?\n(a) First event.\n(b) Second event.\n(c) Third event.',opts)=='event_reference_permutations'
    assert option_structure('What order?',opts)=='other'
    assert option_structure('Order?\n(a) Event.\n(b) Event.',opts)=='other'
    assert option_structure('Order?\n(a) Event.\n(b) Event.\n(c) Event.',[opts[0]]*3+['D. (a)(a)(c).'])=='other'


def test_semantic_option_words_are_not_reference_permutations():
    q='Order?\n(a) Event.\n(b) Event.'
    assert option_structure(q,['A. First (a), then (b).']*4)=='other'


def test_equal_weights_keep_quota_and_do_not_mutate_scores():
    v=dict(start_pts=0,time_base='1/4',duration_fraction='8')
    c=[dict(candidate_index=i,source_frame_index=i,source_pts=4*i+1,timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(6)]
    cfg=dict(seed_budget=3,frame_budget=4,core_gap_seconds='2',window_radius_seconds='2',refinement_min_core_seeds=3)
    scores=[.8,.7,.6,.5,.4,.3];plan=build_regions(c,scores,v,cfg)
    pool=dict(candidates=c,video=v,initial_scores=scores,plan=plan);before=deepcopy(pool)
    selected,trace=fixed_quota_support(pool,np.eye(6,dtype=np.float32),{'0':4},cfg,equal=True)
    assert len(selected)==4 and trace['region_budgets']=={'0':4} and pool==before
    assert set(trace['supports']['0']['seed_weights'].values())=={1.0}
