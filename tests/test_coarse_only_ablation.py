"""补查消融的可行配额、无细帧等价和输入只读测试。"""
import copy
import numpy as np
import pytest
from test_relevance_coverage_budget import fixture
from videoqa_methods.coarse_only_ablation import feasible_reference,select_coarse_only
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage


def test_reference_feasible():
    assert feasible_reference([2,5,1],6)==[2,3,1]
    with pytest.raises(ValueError):feasible_reference([2,1],4)


@pytest.mark.parametrize('scores',[[.2]*15,[.9 if i in (1,7,13) else .1 for i in range(15)]])
def test_no_refinement_matches_exp12_and_input_unchanged(scores):
    pool,features,_,parent=fixture(scores);old=copy.deepcopy(pool);saved=features.copy()
    expected,trace=select_relevance_coverage(pool,features,parent)
    selected,actual=select_coarse_only(pool,features,parent,7)
    assert selected==expected and actual['region_budgets']==trace['region_budgets']
    assert pool==old and np.array_equal(features,saved)


def test_short_pool_selects_every_coarse_candidate():
    pool,features,_,parent=fixture([.2]*15)
    selected,trace=select_coarse_only(pool,features,parent,32)
    assert selected==list(range(15)) and trace['actual_budget']==15


def test_added_fine_feature_cannot_enter_or_change_coarse_result():
    pool,features,_,parent=fixture([.2]*15)
    expected,_=select_coarse_only(pool,features,parent,7)
    for frame in pool['candidates']:frame['source_frame_index']*=4
    pool['candidates'].append(dict(candidate_index=15,source_frame_index=6,source_pts=7,
        timestamp_seconds=1.75,requested_seconds=1.75))
    enlarged=np.concatenate([features,features[2:3]],axis=0)
    actual,_=select_coarse_only(pool,enlarged,parent,7)
    assert actual==expected and 15 not in actual
