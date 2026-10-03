"""新池上的exp12选择必须保持原公式，并与同池旧路径等价。"""
import copy
from test_relevance_coverage_budget import fixture
from test_retarget_seed_pool import fixture as pool_fixture
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_methods.retarget_seed_pool import retarget_pool
from videoqa_methods.seed_budget_selection import select_seed_budget


def test_generic_builder_matches_existing_parent():
    pool,f,_,parent=fixture([.9 if i in [1,7,13] else .1 for i in range(15)])
    expected,trace=select_relevance_coverage(pool,f,parent)
    actual,new=select_seed_budget(pool,f,7)
    assert actual==expected and new['region_budgets']==trace['region_budgets']


def test_retargeted_pool_budget_and_anchors():
    pool,f,cfg=pool_fixture();new,nf=retarget_pool(pool,f,dict(cfg,seed_budget=4));saved=copy.deepcopy(new)
    indices,trace=select_seed_budget(new,nf,7)
    assert len(indices)==len(set(indices))==7
    assert set(trace['protected_indices'])<=set(indices) and new==saved


def test_budget_upper_bound_not_duplicate_padding():
    pool,f,cfg=pool_fixture();new,nf=retarget_pool(pool,f,dict(cfg,seed_budget=2))
    indices,trace=select_seed_budget(new,nf,32)
    assert len(indices)==8 and indices==list(range(8))
