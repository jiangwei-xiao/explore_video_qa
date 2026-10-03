"""等权效用不等于均分配额，且不改变局部帧顺序或保护锚点。"""
from videoqa_methods.equal_region_budget import select_equal_region_budget


def test_equal_utility_still_allocates_adaptively():
    candidates=[dict(candidate_index=i,source_pts=i) for i in range(6)]
    trace=dict(regions=[dict(region_id=0,prefix=[0,1,2],coverage_curve=[0,.8,.9,.95],density_weight=10),
                        dict(region_id=1,prefix=[3,4,5],coverage_curve=[0,.1,.2,.6],density_weight=1)],
               region_budgets={'0':3,'1':1},protected_indices=[0,3],eligible_indices=list(range(6)))
    selected,new=select_equal_region_budget(candidates,trace)
    assert new['region_budgets']=={'0':1,'1':3}
    assert selected==[0,3,4,5]
    assert [r['prefix'] for r in new['regions']]==[[0,1,2],[3,4,5]]
    assert trace['regions'][0]['density_weight']==10 and 'budget_weight' not in trace['regions'][0]


def test_common_multiplier_keeps_exact_discrete_solution():
    candidates=[dict(candidate_index=i,source_pts=i) for i in range(4)]
    trace=dict(regions=[dict(region_id=0,prefix=[0,1],coverage_curve=[0,.5,1],density_weight=7),
                        dict(region_id=1,prefix=[2,3],coverage_curve=[0,.5,1],density_weight=7)],
               region_budgets={'0':2,'1':1},protected_indices=[0,2],eligible_indices=list(range(4)))
    selected,new=select_equal_region_budget(candidates,trace)
    assert selected==[0,1,2] and new['exact_equal_weight_equivalence']
