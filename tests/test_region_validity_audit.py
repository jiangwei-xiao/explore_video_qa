"""位置筛查不是过滤器，测试精确边界及复核选择不依赖答案。"""
from videoqa_audit.region_validity_audit import describe_regions,review_candidates

def record(pts):
    return dict(question_id='x',correct=False,selection=dict(video=dict(duration_fraction='100',time_base='1/10',start_pts=20),
        candidates=[dict(source_pts=pts)],initial_scores=[.2],initial_count=1,
        plan=dict(regions=[dict(region_id=0,anchor_index=0,seed_indices=[0],weight=1,refine=False,left_fraction='0',right_fraction='100')]),selection_trace=dict(region_selected={'0':[0]})))

def test_exact_tail_boundary():
    assert describe_regions(record(970))[0]['last_five_percent']
    assert not describe_regions(record(969))[0]['last_five_percent']

def test_answer_does_not_change_screening():
    r=record(970);before=describe_regions(r);r['correct']=True;r['answer']={'parsed_answer':'D'}
    assert before==describe_regions(r)

def test_distinct_questions_and_quota_order():
    rows=[dict(question_id=q,region_id=i,quota=n,last_five_percent=True) for q,i,n in [('b',0,8),('a',1,8),('a',2,7),('c',0,6)]]
    assert review_candidates(rows,3)==['a','b','c']
