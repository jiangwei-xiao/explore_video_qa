"""零模型双数据集审计的关键边界回归。"""
import pytest
from videoqa_audit.cross_dataset_review import (
    TASKS, normalize_option_period, freeze_balanced_subset, cluster_report, audit_selection_paths)


@pytest.mark.parametrize('raw,native,expected', [
    (' C.\n', 'C.', 'C'), ('A.', 'A.', 'A'), ('D. Bicycle', 'D. Bicycle', 'D. Bicycle'),
    ('c.', 'c.', 'c.'), ('C..', 'C..', 'C..'), ('C)', 'C)', 'C)'),
    ('C. explanation', 'C. explanation', 'C. explanation'), ('B.', 'A', 'A')])
def test_period_scope(raw, native, expected):
    assert normalize_option_period(raw, native) == expected


def test_subset_deterministic_identity_only():
    rows = [dict(question_id=f'{t}{i}', task_type=t, video_id=f'{t}-v{i}') for t in TASKS for i in range(3)]
    a = freeze_balanced_subset(rows, 2)
    assert a == freeze_balanced_subset(list(reversed(rows)), 2)
    assert a['question_count'] == a['unique_content_videos'] == 14
    with pytest.raises(ValueError, match='identity fields'):
        freeze_balanced_subset([dict(rows[0], answer_index=0)], 1)
    with pytest.raises(ValueError, match='Insufficient'):
        freeze_balanced_subset(rows, 4)


def test_macro_versus_micro_and_shared_video_clusters():
    rows = [dict(question_id=f'{t}{i}', task_type=t, video_id=f'v{i}') for t in TASKS for i in range(2)]
    a = {r['question_id']: dict(correct=r['video_id']=='v0') for r in rows}
    b = {r['question_id']: dict(correct=False) for r in rows}
    result = cluster_report(rows, {'a': a, 'b': b}, macro=True, repetitions=100)
    assert result['clusters'] == 2
    assert result['scores']['a']['main_metric_percent'] == 50
    assert len(result['pairs']['a minus b']['cells']['left_only']) == 7
    assert result == cluster_report(rows, {'a': a, 'b': b}, macro=True, repetitions=100)
    uneven = rows + [dict(question_id='extra'+str(i), task_type=TASKS[0], video_id='v0') for i in range(2)]
    extended = dict(a, extra0=dict(correct=True), extra1=dict(correct=True))
    score = cluster_report(uneven, {'a': extended}, macro=True, repetitions=100)['scores']['a']
    assert score['main_metric_percent'] == pytest.approx(100*(.75+6*.5)/7)
    assert score['accuracy_percent'] == 100*9/16


def test_quota_capacity_and_stable_score_ties():
    candidates = [dict(candidate_index=i, source_frame_index=i, source_pts=i,
                       timestamp_seconds=float(i), requested_seconds=float(i)) for i in range(5)]
    regions = [dict(region_id=0, prefix=[0, 2, 1], reference_coarse_indices=[0, 1, 2],
                    members=[0, 1, 2], density_weight=1, coverage_curve=[0, .4, .6, 1]),
               dict(region_id=1, prefix=[3, 4], reference_coarse_indices=[3, 4],
                    members=[3, 4], density_weight=1, coverage_curve=[0, .5, 1])]
    pool = dict(candidates=candidates, initial_scores=[1, 1, .5, .4, .2], initial_count=5,
                plan=dict(seed_indices=[0, 1, 2]))
    left = dict(regions=regions, region_budgets={'0': 2, '1': 1})
    right = dict(regions=regions, region_budgets={'0': 1, '1': 2})
    result = audit_selection_paths(pool, left, right, 3)
    path = result['topk_paths'][1]
    assert path['candidate_index'] == 1
    assert path['prefix_position'] == 3 and path['max_feasible_quota'] == 2
    assert path['exp07']['state'] == path['exp12']['state'] == 'beyond_feasible_quota'
