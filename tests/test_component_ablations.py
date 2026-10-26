"""组件消融的真实配额、全局资格、同父控制与预算边界测试；不调用模型。"""
from copy import deepcopy
import inspect

import numpy as np
import pytest

from videoqa_generalization import component_ablations as component
from videoqa_methods.budget_contract import budget_config
from videoqa_methods.density_selection import build_regions, frame_key
from videoqa_methods.coarse_eligibility import coarse_selection_cells, select_global_coarse
from videoqa_methods.coarse_only_ablation import feasible_reference, select_coarse_only
from videoqa_methods.prefix_coverage_budget import select_prefix_coverage
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_audit.density_reward_decay import select_density_decay


def fixture(with_fine=True, k=8, n=40):
    """接口：用固定种子构造真实有理数时钟和多区域数值资产，不代表研究成绩。"""
    candidates = [dict(candidate_index=i, source_frame_index=4*i+1, source_pts=4*i+1,
                       timestamp_seconds=i+.25, requested_seconds=i+.25) for i in range(n)]
    video = dict(path='synthetic.mp4', start_pts=0, time_base='1/4', duration_fraction=str(n+1))
    scores = [.02] * n
    for rank, i in enumerate([2, 3, 4, 22, 23, 24, 35, 37] if n == 40 else range(n)):
        scores[i] = .95 - rank * .02
    cfg = budget_config(k); plan = build_regions(candidates, scores, video, cfg)
    if with_fine:
        candidates.append(dict(candidate_index=n, source_frame_index=15 if n == 40 else 3,
            source_pts=15 if n == 40 else 3, timestamp_seconds=3.75 if n == 40 else .75,
            requested_seconds=3.75 if n == 40 else .75))
    features = np.random.default_rng(2027).normal(size=(len(candidates), 16)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    pool = dict(candidates=candidates, video=video, initial_count=n, initial_scores=scores, plan=plan)
    cells = coarse_selection_cells(plan, video)['regions']
    from videoqa_methods.density_selection import source_time
    from fractions import Fraction
    sizes = [sum(Fraction(r['left_fraction']) <= source_time(c,video) <= Fraction(r['right_fraction'])
                 for c in candidates) for r in cells]
    reference = dict(zip((str(r['region_id']) for r in cells), feasible_reference(sizes, min(k,len(candidates)))))
    _, prefix = select_prefix_coverage(candidates, features, plan, video, cfg, reference, scores)
    return pool, features, prefix


def test_A_is_original_coarse_only_and_does_not_mutate_parent():
    pool, features, prefix = fixture(); before = deepcopy(pool); f = features.copy()
    expected, _ = select_coarse_only(pool, features, prefix, 8)
    selected, trace = component.select_component('A', pool, features, prefix, 8)
    assert selected == expected and len(selected) == 8 and max(selected) < pool['initial_count']
    assert pool == before and np.array_equal(features, f)
    assert trace['initial_count'] == 40 and trace['actual_budget'] == 8


def test_B_uses_real_global_first_pass_and_same_local_prefixes():
    pool, features, prefix = fixture(); cfg = budget_config(8)
    cells = coarse_selection_cells(pool['plan'], pool['video'])
    _, allocation = select_density_decay(pool['candidates'], features, cells, pool['video'], cfg)
    expected, _ = select_global_coarse(pool['candidates'], features, pool['plan'], pool['video'],
                                     cfg, allocation['region_budgets'], pool['initial_scores'])
    selected, trace = component.select_component('B', pool, features, prefix, 8)
    assert selected == expected and trace['region_budgets'] == allocation['region_budgets']
    assert set(trace['first_pass_trace']['eligible_indices']) == set(range(len(features)))
    assert trace['quota_origin'] == 'recomputed_select_density_decay_on_global_coarse_cells'
    assert trace['direct_second_pass_exact'] and trace['local_prefixes_unchanged']
    assert sum(trace['region_budgets'].values()) == 8


def test_B_ignores_feasibility_placeholder_and_reference_selection():
    pool, features, prefix = fixture(); changed = deepcopy(prefix)
    changed['reference_quotas'] = {str(r['region_id']): 999999 for r in prefix['regions']}
    changed['reference_prefix_selection'] = []
    a, ta = component.select_component('B', pool, features, prefix, 8)
    b, tb = component.select_component('B', pool, features, changed, 8)
    assert a == b and ta['region_budgets'] == tb['region_budgets']
    assert ta['region_budgets'] != prefix['reference_quotas']


def test_A_without_fine_exactly_reproduces_parent():
    pool, features, prefix = fixture(False)
    parent, _ = select_relevance_coverage(pool, features, prefix)
    selected, _ = component.select_component('A', pool, features, prefix, 8)
    assert parent == selected


def test_short_pool_budget_is_not_padded_or_repeated():
    pool, features, prefix = fixture(True, k=32, n=7)
    a, _ = component.select_component('A', pool, features, prefix, 32)
    b, _ = component.select_component('B', pool, features, prefix, 32)
    assert len(a) == len(set(a)) == 7 and a == list(range(7))
    assert len(b) == len(set(b)) == 8
    assert b == sorted(b, key=lambda i: frame_key(pool['candidates'][i]))


@pytest.mark.parametrize('q', [0, -1, 99, True])
def test_invalid_quota_rejected(q):
    prefix = dict(regions=[dict(region_id=0, prefix=[3, 4])])
    with pytest.raises(ValueError, match='quota'):
        component.cut_prefixes(prefix, {'0': q}, 1)


def test_quota_scope_total_and_source_identity():
    prefix = dict(regions=[dict(region_id=0, prefix=[3, 4])])
    with pytest.raises(ValueError, match='identity'):
        component.cut_prefixes(prefix, {'other': 1}, 1)
    with pytest.raises(ValueError, match='closed'):
        component.cut_prefixes(prefix, {'0': 1}, 2)
    pool, _, _ = fixture()
    a = dict(video=pool['video'], selected_frames=pool['candidates'][:8])
    b = deepcopy(a)
    for f in b['selected_frames']:f['candidate_index'] += 1000
    assert component.source_identity(a) == component.source_identity(b)
    b['selected_frames'].reverse()
    assert component.source_identity(a) != component.source_identity(b)


def test_selector_interface_has_no_labels_or_independent_M():
    assert list(inspect.signature(component.select_component).parameters) == [
        'control', 'pool', 'features', 'prefix', 'frame_budget']
    with pytest.raises(ValueError):component.budget('unknown')
