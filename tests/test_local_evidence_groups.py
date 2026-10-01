"""局部缺失侧修补的预算、保护及来源不变测试，无模型调用。"""
import numpy as np
from videoqa_audit.local_evidence_groups import complete_anchor_sides


def fixture(selected,refine=True):
    c=[dict(candidate_index=i,source_frame_index=i,source_pts=4*i+1,timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(9)]
    v=dict(start_pts=0,time_base='1/4',duration_fraction='10')
    r=dict(region_id=0,anchor_index=4,seed_indices=[3,4,5],left_fraction='0',right_fraction='9',refine=refine)
    p=dict(candidates=c,video=v,plan=dict(regions=[r]),initial_scores=[.5]*9,initial_count=9)
    t=dict(regions=[dict(region_id=0,members=list(range(9)),prefix=selected)],region_budgets={'0':len(selected)})
    return p,np.eye(9,dtype=np.float32),t


def test_already_both_sides_exactly_unchanged():
    p,f,t=fixture([4,2,6]);ids,trace=complete_anchor_sides(p,f,t)
    assert ids==[2,4,6] and trace['replacement_count']==0


def test_missing_side_preserves_sole_existing_side_and_anchor():
    p,f,t=fixture([4,1,2]);ids,trace=complete_anchor_sides(p,f,t)
    assert 4 in ids and 1 in ids and any(i>4 for i in ids)
    assert trace['replacement_count']==1 and trace['regions'][0]['changes'][0]['removed_index']==2


def test_quota_below_three_is_unchanged():
    p,f,t=fixture([4,1]);ids,trace=complete_anchor_sides(p,f,t)
    assert ids==[1,4] and not trace['regions'][0]['eligible']


def test_nonrefinement_region_unchanged():
    p,f,t=fixture([4,1,2],False);ids,trace=complete_anchor_sides(p,f,t)
    assert ids==[1,2,4] and trace['replacement_count']==0


def test_no_candidates_on_side_falls_back():
    p,f,t=fixture([4,6,7]);p['plan']['regions'][0]['left_fraction']='17/4'
    ids,trace=complete_anchor_sides(p,f,t)
    assert ids==[4,6,7] and trace['replacement_count']==0


def test_two_missing_sides_replace_at_most_two_and_keep_budget():
    p,f,t=fixture([4,0,8]);p['plan']['regions'][0].update(left_fraction='2',right_fraction='7')
    ids,trace=complete_anchor_sides(p,f,t)
    assert len(ids)==3 and 4 in ids and trace['replacement_count']==2
    assert trace['regions'][0]['new_side_coverage']=={'before':True,'after':True}


def test_repeat_is_deterministic_and_new_sources_not_privileged():
    p,f,t=fixture([4,1,2]);a=complete_anchor_sides(p,f,t)
    p['initial_count']=6
    assert complete_anchor_sides(p,f,t)==a
