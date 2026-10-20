"""只读证据汇总的计分、比较方向、缺测及完成恢复测试；不调用任何模型。"""
from copy import deepcopy
import json

import pytest

from videoqa_runtime.common import sha256
from videoqa_audit import rd_evidence as evidence


def macro_score():
    """接口：人工不平衡任务分母，专用于检出宏/微平均混用。"""
    tasks = {name: dict(n=1, correct=1) for name in evidence.MLVU_TASKS}
    tasks['plotQA'] = dict(n=100, correct=0)
    return dict(n=106, correct=6, accuracy_percent=100 * 6 / 106,
                main_metric_percent=100 * 6 / 7, tasks=tasks)


def test_macro_metric_preserves_task_denominators_and_full_precision():
    raw = macro_score(); value = evidence.metric_record(raw, 'M-Avg', 106)
    assert value['score_percent'] == 100 * 6 / 7
    assert value['accuracy_percent'] == 100 * 6 / 106
    assert value['score_percent'] != value['accuracy_percent']
    raw['main_metric_percent'] = raw['accuracy_percent']
    with pytest.raises(ValueError, match='macro/micro'):
        evidence.metric_record(raw, 'M-Avg')


def test_missing_or_wrong_task_counts_cannot_be_zero_filled():
    raw = macro_score(); raw['tasks'].pop('needle')
    with pytest.raises(ValueError, match='seven task'):
        evidence.metric_record(raw, 'M-Avg')
    raw = macro_score(); raw['tasks']['needle']['correct'] = 0
    with pytest.raises(ValueError, match='population/count'):
        evidence.metric_record(raw, 'M-Avg')
    with pytest.raises((KeyError, TypeError)):
        evidence.metric_record(dict(score=None), 'accuracy')


def test_video_accuracy_count_and_nested_metric_binding():
    raw = dict(all=dict(n=2700, correct=1710, accuracy=1710 / 2700))
    result = evidence.metric_record(raw, 'accuracy', 2700)
    assert result['score_percent'] == 100 * 1710 / 2700
    with pytest.raises(ValueError, match='population'):
        evidence.metric_record(raw, 'accuracy', 2174)
    raw['all']['accuracy'] = .9
    with pytest.raises(ValueError, match='Accuracy/count'):
        evidence.metric_record(raw, 'accuracy')


def test_source_pointer_escapes_profile_slashes_and_preserves_method_keys():
    document = {'videomme/llava8': {'native': {'scores': {'RD-1.2-exp12': {'n': 2700}}}}}
    location = evidence.pointer('videomme/llava8', 'native', 'scores', 'RD-1.2-exp12')
    assert location == '/videomme~1llava8/native/scores/RD-1.2-exp12'
    assert evidence.resolve_pointer(document, location) == {'n': 2700}
    assert evidence.resolve_pointer({'a~b': [123]}, evidence.pointer('a~b', 0)) == 123
    with pytest.raises(ValueError, match='absolute JSON pointer'):
        evidence.resolve_pointer(document, 'videomme/llava8')


def paired():
    """接口：不对称区间和四格用于检查反向比较端点及得失交换。"""
    return dict(cells=dict(both_correct=['q0'], left_only=['q1', 'q2'], right_only=['q3'], both_wrong=['q4']),
                difference_pp=20., video_cluster_95_pp=[-10., 30.])


def test_reverse_pair_swaps_interval_and_gain_loss_together():
    raw = paired(); before = deepcopy(raw)
    result = evidence.paired_record('a', 'b', raw, 5, reverse=True)
    assert (result['left'], result['right']) == ('b', 'a')
    assert result['difference_pp'] == -20.
    assert result['interval_95_pp'] == [-30., 10.]
    assert (result['improved'], result['regressed'], result['net_correct']) == (1, 2, -1)
    assert raw == before


def test_pair_duplicate_population_and_missing_interval_rejected():
    raw = paired(); raw['cells']['both_wrong'] = ['q0']
    with pytest.raises(ValueError, match='population'):
        evidence.paired_record('a', 'b', raw, 5)
    raw = paired(); raw.pop('video_cluster_95_pp')
    with pytest.raises(ValueError, match='interval'):
        evidence.paired_record('a', 'b', raw, 5)


def test_author_tables_are_not_local_paired_or_normalized_scores():
    local = dict(profile='llava8', dataset='mlvu', score_version='native')
    reference = dict(profile='llava8', dataset='mlvu', score=66.9, budget_policy='fixed')
    assert evidence.comparison_qualification(local, reference) == 'same_named_setting_author_reported_not_local_paired'
    local['score_version'] = 'period_normalized'
    assert evidence.comparison_qualification(local, reference) == 'different_scoring_version_context_only'
    local['score_version'] = 'native'; reference['profile'] = 'qwen32'
    assert evidence.comparison_qualification(local, reference) == 'different_model_budget_or_dataset'
    reference['profile'] = 'llava8'; reference['budget_policy'] = 'at_most_32_plus_auxiliary_VLM_analysis'
    assert evidence.comparison_qualification(local, reference) == 'same_budget_ceiling_auxiliary_analysis_context_only'
    reference['score'] = None
    assert evidence.comparison_qualification(local, reference) == 'not_reported_for_target_configuration'


def test_reader_checks_only_requested_files_and_changed_hash(tmp_path):
    path = tmp_path / 'record.json'; path.write_text('{"n":197}')
    reader = evidence.EvidenceReader(tmp_path)
    assert reader.read('record.json') == dict(n=197)
    assert set(reader.sources) == {'record.json'}
    reader.verify(); path.write_text('{"n":194}')
    with pytest.raises(ValueError, match='mutated'):
        reader.verify()
    with pytest.raises(ValueError, match='fingerprint'):
        reader.read('record.json', expected='0' * 64)
    with pytest.raises(ValueError, match='outside repository'):
        reader.path('../outside.json')


def test_completed_restore_never_collects_or_changes_files(tmp_path, monkeypatch):
    root = tmp_path / 'repo'; root.mkdir(); output = tmp_path / 'analysis'; output.mkdir()
    original = root / 'source.json'; original.write_text('{}')
    record = output / 'evidence.json'; record.write_text('{"status":"completed"}')
    completion = dict(sources={'source.json': sha256(original)}, artifacts={'evidence.json': sha256(record)})
    (output / 'completion.json').write_text(json.dumps(completion))
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed restore must not collect or write tables')
    monkeypatch.setattr(evidence, 'collect_tables', forbidden)
    before = record.read_bytes()
    assert evidence.run(output, root)['status'] == 'completed' and record.read_bytes() == before
    original.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='source changed'):
        evidence.run(output, root)
