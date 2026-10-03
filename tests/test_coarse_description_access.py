"""粗候选描述资格对照的退化、等价性与顺序无关性验收。"""
import numpy as np
from test_local_query_coverage import fixture
from videoqa_methods.coarse_description_access import select_coarse_description_access
from videoqa_methods.local_query_coverage import select_local_query_coverage
from videoqa_methods.density_selection import build_regions
from videoqa_methods.option_seed_entry import select_option_pool


def test_constant_and_disabled_replay_baseline():
    pool, f, cfg, old, trace = fixture()
    for enabled in [True, False]:
        selected, new = select_coarse_description_access(pool, f, cfg, trace['region_budgets'], np.full((4, 15), .2), enabled)
        assert selected == old and new['region_budgets'] == trace['region_budgets']


def test_all_window_domain_equals_previous_selector():
    pool, f, cfg, _, _ = fixture()
    cfg = dict(cfg, window_radius_seconds='20')
    pool['plan'] = build_regions(pool['candidates'], pool['initial_scores'], pool['video'], cfg)
    _, baseline = select_option_pool(pool, f, cfg)
    scores = np.random.default_rng(11).random((4, 15))
    old, trace = select_local_query_coverage(pool, f, cfg, baseline['region_budgets'], scores)
    new, other = select_coarse_description_access(pool, f, cfg, baseline['region_budgets'], scores)
    assert new == old and other['region_budgets'] == trace['region_budgets']
    assert other['objective'] == trace['objective']


def test_description_permutation_and_candidate_identity():
    pool, f, cfg, _, baseline = fixture()
    scores = np.random.default_rng(11).random((4, 15))
    selected, trace = select_coarse_description_access(pool, f, cfg, baseline['region_budgets'], scores)
    other, other_trace = select_coarse_description_access(pool, f, cfg, baseline['region_budgets'], scores[[2, 3, 0, 1]])
    assert selected == other and trace['region_budgets'] == other_trace['region_budgets']
    assert len(set(selected)) == len(selected) == 7
    assert set(baseline['protected_indices']) <= set(selected)
    assert sorted(i for r in trace['regions'] for i in r['description_indices']) == list(range(15))
