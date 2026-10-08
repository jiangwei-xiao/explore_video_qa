"""单预算适配与结构推导边界测试；不运行任何模型。"""
import inspect
import numpy as np
import pytest
from videoqa_methods.budget_contract import budget_config, select_exp07
from videoqa_methods.density_selection import build_regions
from videoqa_methods.prefix_coverage_budget import select_prefix_coverage
from videoqa_audit.framework_contraction import inspect_regions, merge_intervals


def pool_at(times, requested=None, budget=8, fine=()):
    """测试接口：用精确四分之一秒源时钟建立小池，可独立设置VFR目标映射。"""
    requested = times if requested is None else requested
    video = dict(start_pts=0, time_base='1/4', duration_fraction=str(max(times) + 5))
    candidates = [dict(candidate_index=i, source_frame_index=i, source_pts=round(t*4),
                       timestamp_seconds=t, requested_seconds=r) for i, (t, r) in enumerate(zip(times, requested))]
    scores = [0.9 - i*0.01 for i in range(len(times))]
    plan = build_regions(candidates, scores, video, budget_config(budget))
    for t in fine:
        i=len(candidates); candidates.append(dict(candidate_index=i, source_frame_index=i,
            source_pts=round(t*4), timestamp_seconds=t, requested_seconds=t))
    return dict(video=video, candidates=candidates, initial_count=len(times), initial_scores=scores, plan=plan)


def minimal_trace(pool):
    """测试接口：结构审计只需要配额和参考身份，不提供人工语义或预测。"""
    return dict(region_budgets={str(r['region_id']):1 for r in pool['plan']['regions']},
                regions=[dict(region_id=r['region_id'], reference_coarse_indices=r['coarse_indices'])
                         for r in pool['plan']['regions']])


def test_single_budget_adapter_matches_legacy_and_keeps_short_seed_count_separate():
    pool = pool_at([0.25, 1.25, 2.25], fine=(0.75, 1.75))
    f=np.eye(5, dtype=np.float32)
    picked, trace = select_exp07(pool, f, frame_budget=8)
    old, _ = select_prefix_coverage(pool['candidates'], f, pool['plan'], pool['video'],
                                    dict(frame_budget=8), {'0':5}, pool['initial_scores'])
    assert picked == old and len(picked) == 5
    assert trace['budget_contract'] == dict(frame_budget=8, seed_count=3, actual_frame_count=5)
    assert list(inspect.signature(budget_config).parameters) == ['frame_budget']


def test_mismatched_seed_pool_is_rejected_instead_of_silently_retargeted():
    pool=pool_at([0.25, 1.25, 2.25], budget=2)
    with pytest.raises(ValueError, match='single-budget seed plan'):
        select_exp07(pool, np.eye(3, dtype=np.float32), frame_budget=8)


def test_density_connections_do_not_determine_refinement_trigger():
    # 两条连接可以分属两个二种子核心，也可以连续组成三种子核心。
    two_two=pool_at([0.25,1.25,6.25,7.25], budget=4)
    three_one=pool_at([0.25,1.25,2.25,7.25], budget=4)
    assert two_two['plan']['regions'][0]['weight'] == three_one['plan']['regions'][0]['weight'] == 3
    assert not two_two['plan']['regions'][0]['refine']
    assert three_one['plan']['regions'][0]['refine']
    r=inspect_regions(two_two, [0], minimal_trace(two_two), 4)
    assert r['regions'][0]['density_threshold_would_differ']


def test_direct_seed_windows_have_a_vfr_counterexample():
    # 两个相邻粗目标命中相隔9秒的源帧；原核心用目标连接后形成连续窗口。
    pool=pool_at([0.25,9.25], requested=[0.25,1.25], budget=2)
    r=inspect_regions(pool, [0], minimal_trace(pool), 2)
    assert not r['source_gap_condition'] and not r['direct_seed_union_equal']


def test_closed_window_union_clipping_and_shared_endpoint():
    from fractions import Fraction as F
    assert merge_intervals([(F(0),F(4)),(F(4),F(8)),(F(10),F(11))]) == [[F(0),F(8)],[F(10),F(11)]]
