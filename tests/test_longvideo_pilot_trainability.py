"""长视频CPU诊断的质量口径、监督隔离、配对统计与恢复边界。"""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from videoqa_runtime.common import sha256
from videoqa_audit.set_value_trainability import TrainingAssets
from videoqa_audit import longvideo_pilot_trainability as diagnostic


def public_row():
    """接口：只有公开选项的数值测试行，不含标签或教师反馈。"""
    return dict(options=['one', 'two', 'three', 'four'], option_letters=list('DABC'))


@pytest.mark.parametrize('value,expected', [
    ('', True), ('   ', True), ('##### From [0:03:50 to 0:04:00]:', True),
    ('## FROM [12:03:50.5 to 12:04:00.25]: ', True),
    ('From [0:03:50 to 0:04:00]: an actual action', False),
    ('##### From [0:03:50 to 0:04:00]: an answer', False),
    ('A scene from [0:03:50 to 0:04:00]', False),
])
def test_option_defects_require_complete_heading_or_empty(value, expected):
    row = public_row(); row['options'][0] = value
    result = diagnostic.option_defects(row)
    assert bool(result) is expected
    if expected:
        assert result[0]['letter'] == 'D'


def population():
    """接口：构造固定折容量的人工公开总体；不代表实际视频或研究成绩。"""
    groups = sorted(['g' + str(i) for i in range(200)], key=lambda group: hashlib.sha256(
        ('2027/' + diagnostic.SCOPE + '/' + group).encode()).hexdigest())
    labels = {i: 'p' + str(i) for i in range(200)}
    labels.update(dict(zip((2, 4, 9), sorted(diagnostic.EXCLUDED))))
    labels.update(dict(zip((0, 1, 3), sorted(diagnostic.UNCERTAIN))))
    rows = []
    for i, group in enumerate(groups):
        row = dict(public_row(), question_id=labels[i], video_id='v' + str(i),
                   content_group=group, split='train', pilot=True)
        if labels[i] in diagnostic.EXCLUDED:
            row['options'][2] = '##### From [0:00:50 to 0:01:00]:'
        rows.append(row)
    return rows[::-1]


def test_population_partitions_before_mask_without_reading_answers():
    rows = population(); changed = deepcopy(rows)
    for i, row in enumerate(changed):
        row.update(answer_letter='ABCD'[i % 4], correct=bool(i % 2))
    valid, quality, folds = diagnostic.freeze_population(rows)
    other = diagnostic.freeze_population(changed)
    assert quality == other[1] and folds == other[2]
    assert len(valid) == 197 and len(quality['sensitivity_question_ids']) == 194
    assert diagnostic.UNCERTAIN.issubset({r['question_id'] for r in valid})
    assert [len(f['heldout_indices']) for f in folds['folds']] == [40, 40, 39, 40, 38]
    assert sorted(i for f in folds['folds'] for i in f['heldout_indices']) == list(range(197))
    for f in folds['folds']:
        assert not set(f['train_indices']) & set(f['heldout_indices'])
        assert not set(f['train_question_ids']) & set(f['heldout_question_ids'])
    changed[0]['split'] = 'val'
    with pytest.raises(ValueError, match='training rows'):
        diagnostic.freeze_population(changed)


def test_changed_quality_population_stops_instead_of_replacing_question():
    rows = population(); rows[0]['options'][0] = ''
    with pytest.raises(ValueError, match='quality partition'):
        diagnostic.freeze_population(rows)
    rows = population(); rows[0]['content_group'] = rows[1]['content_group']
    with pytest.raises(ValueError, match='not unique'):
        diagnostic.freeze_population(rows)


def test_original_letter_mapping_and_strict_four_option_parser():
    row = dict(public_row(), question_id='q')
    def result(raw, pred, correct):
        return dict(question_id='q', phase='pilot', correct=correct,
                    answer=dict(raw_output=raw, parsed_answer=pred,
                                framework_metric=dict(correct=correct)))
    assert diagnostic.verify_answer(row, result('A.', 'A', True), 'A') == 1
    assert diagnostic.verify_answer(row, result('(D)', 'D', False), 'A') == 0
    assert diagnostic.verify_answer(row, result('E', None, False), 'A') == 0
    with pytest.raises(ValueError, match='binding'):
        diagnostic.verify_answer(row, result('A.', 'A', True), 'D')


def small_data():
    """接口：两个独立视频的人工张量用于比较分母及去重，不执行真实问答。"""
    x = np.zeros((2, 4, 8, 770), dtype=np.float32)
    y = np.array([[1, 1, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    rows = [dict(question_id='q' + str(i), video_id='v' + str(i), content_group='c' + str(i),
                 duration_seconds=300 + 500 * i, split='train') for i in range(2)]
    data = TrainingAssets(rows, x, y, [[0, 2], [0, 1, 2, 3]],
                          [[[[j, 0]] for j in range(4)] for _ in range(2)])
    predictions = [dict(chosen_index=0, baseline_index=2, mean_itm_index=0, old_head_index=2,
                        scores=[0., 0., 0., 0.]),
                   dict(chosen_index=1, baseline_index=0, mean_itm_index=0, old_head_index=0,
                        scores=[0., 1., 0., 0.])]
    return data, predictions


def test_sensitivity_reuses_predictions_and_question_averaged_preferences():
    data, predictions = small_data(); before = deepcopy(predictions)
    primary = diagnostic.evaluate_subset(data, predictions, [0, 1])
    sensitive = diagnostic.evaluate_subset(data, predictions, [1])
    assert predictions == before
    assert primary['preference_questions'] == 2 and primary['preference_accuracy'] == .75
    assert primary['unique_preference_pairs'] == 4
    assert primary['correct']['learned_oof'] == 2
    assert primary['comparisons']['training_fold_best']['net_correct'] == 2
    assert primary['improvement_videos'] == 2
    assert sensitive['questions'] == 1 and sensitive['preference_accuracy'] == 1.
    assert sensitive['comparisons']['training_fold_best']['new_correct'] == ['q1']
    assert primary['by_duration']['600–1800s']['questions'] == 1


def test_decision_requires_all_fit_and_both_cohorts_without_extra_sensitivity_gate():
    keys = ('preference_at_least_sixty', 'net_gain_over_training_fold_best',
            'no_lower_than_best_fixed', 'improvements_in_at_least_three_videos')
    main = dict(criteria=dict.fromkeys(keys, True))
    sensitive = deepcopy(main); sensitive['criteria']['preference_at_least_sixty'] = False
    assert diagnostic.recommend(main, sensitive, True)['passed']
    assert diagnostic.recommend(main, sensitive, False)['decision'] == 'training_not_fully_fitted'
    sensitive['criteria']['net_gain_over_training_fold_best'] = False
    assert diagnostic.recommend(main, sensitive, True)['decision'] == 'close_current_frozen_blip_set_head'
    assert not diagnostic.recommend(main, sensitive, True)['automatic_teacher_extension']


def test_freeze_json_rejects_protocol_change(tmp_path):
    path = tmp_path / 'fixed.json'; diagnostic.freeze_json(path, dict(a=1))
    before = path.read_bytes(); diagnostic.freeze_json(path, dict(a=1))
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='record differs'):
        diagnostic.freeze_json(path, dict(a=2))


def test_completed_restore_is_read_only_and_detects_source_change(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    source = tmp_path / 'original.json'; source.write_text('{}')
    report = tmp_path / 'report.json'; report.write_text(json.dumps(dict(status='completed')))
    completion = dict(sources={str(source): sha256(source)}, artifacts={'report.json': sha256(report)})
    (tmp_path / 'completion.json').write_text(json.dumps(completion))
    def forbidden(*args, **kwargs):
        raise AssertionError('Restore attempted to train or load assets')
    monkeypatch.setattr(diagnostic, 'cross_validate', forbidden)
    monkeypatch.setattr(diagnostic, 'load_pilot_assets', forbidden)
    before = report.read_bytes()
    assert diagnostic.run(tmp_path)['status'] == 'completed'
    assert report.read_bytes() == before
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='fingerprint changed'):
        diagnostic.run(tmp_path)
