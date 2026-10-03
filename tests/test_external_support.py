"""正式支持扩展接口的同池等价、固定配额及参考提议测试。"""
import numpy as np
from test_retarget_seed_pool import fixture
from videoqa_methods.seed_budget_selection import select_seed_budget
from videoqa_methods.density_selection import build_regions
from videoqa_methods.external_support import select_external_support


def test_no_outside_reference_preserves_parent():
    pool,f,_=fixture();old,parent=select_seed_budget(pool,f,7)
    selected,trace=select_external_support(pool,f,parent['prefix_trace'],parent)
    assert selected==old and trace['proposals']==[]


def test_reference_changes_support_but_not_quota_or_anchor():
    c=[dict(candidate_index=i,source_pts=i,source_frame_index=i,timestamp_seconds=float(i),requested_seconds=float(i)) for i in range(10)]
    video=dict(start_pts=0,time_base='1',duration_fraction='10');scores=[.5,.4,.9,.95,.91,.7,.65,.6,.3,.2]
    cfg=dict(seed_budget=3,frame_budget=4,core_gap_seconds='2',window_radius_seconds='0',refinement_min_core_seeds=3)
    pool=dict(video=video,candidates=c,initial_count=10,initial_scores=scores,plan=build_regions(c,scores,video,cfg))
    f=np.eye(10,dtype=np.float32);old,parent=select_seed_budget(pool,f,4)
    selected,trace=select_external_support(pool,f,parent['prefix_trace'],parent)
    assert selected==[2,3,4,5] and old==[0,2,3,4]
    assert trace['region_budgets']==parent['region_budgets'] and 3 in selected
