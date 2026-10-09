"""有界诊断入口回归：预算、原始返回恢复及不确定状态不重发。"""
import copy
from pathlib import Path
import pytest

from videoqa_audit import text_state_diagnostic as d
from videoqa_full.state import durable
from videoqa_runtime.common import read_json, sha256


def test_population_is_single_predeclared_pair():
    assert d.QUESTION == '022-1'
    assert d.PROFILE == 'llava8'
    assert d.CONDITIONS == ('original', 'candidate')
    assert 'exp30' in str(d.SOURCE)


def test_no_raw_return_forbids_retry(tmp_path):
    durable(tmp_path/'protocol.json', {})
    call = tmp_path/'calls/original/022-1.json'
    durable(call, dict(status='failed_or_uncertain', protocol_sha256=sha256(tmp_path/'protocol.json')))
    with pytest.raises(RuntimeError, match='Uncertain generation'):
        d.recover_return(tmp_path, 'original', '022-1')
    assert read_json(call)['status'] == 'failed_or_uncertain'


def test_completed_return_is_not_regenerated(tmp_path):
    durable(tmp_path/'protocol.json', {})
    call = tmp_path/'calls/original/022-1.json'
    durable(call, dict(status='completed', protocol_sha256=sha256(tmp_path/'protocol.json'), answer='saved'))
    old = sha256(call)
    d.recover_return(tmp_path, 'original', '022-1')
    assert sha256(call) == old


def test_confirmed_raw_reconstructs_without_backend(tmp_path, monkeypatch):
    from videoqa_generalization import llava_observe
    durable(tmp_path/'protocol.json', {})
    fp = sha256(tmp_path/'protocol.json')
    call = tmp_path/'calls/candidate/022-1.json'
    durable(call, dict(status='failed_or_uncertain', gpu='2', protocol_sha256=fp))
    raw = dict(identity=dict(question_id='022-1', method='candidate', protocol_sha256=fp, gpu='2'),
               generation_state={'started':True, 'returned':True})
    durable(tmp_path/'raw_generations/candidate/022-1.json', raw)
    monkeypatch.setattr(d, 'row', lambda: {})
    monkeypatch.setattr(llava_observe, 'result_from_raw', lambda r, row: dict(answer={'parsed_answer':'B'}, observed={}))
    d.recover_return(tmp_path, 'candidate', '022-1')
    assert read_json(call)['status'] == 'completed'
    assert read_json(call)['recovered_postprocessing'] is True


def test_actual_pixels_and_tokens_are_checked(monkeypatch):
    metric = {'pred_answer':'B'}
    class Task:
        def videomme_process_results(self, doc, results):
            return {'videomme_perception_score':metric}
    monkeypatch.setattr(d, 'load_task', lambda: Task())
    monkeypatch.setattr(d, 'task_doc', lambda row: row)
    expected = dict(pixel_sha256='pixels', prompt='prompt', input_token_ids=[3, -200, 5])
    item = dict(input_identity=expected)
    answer = dict(raw_output='B', parsed_answer='B', framework_metric=metric,
                  observed=dict(expected, visual_tokens=1680, max_new_tokens=16,
                                generate_calls=1, prefill_calls=1, generated_token_ids=[6]))
    d.validate_answer(item, answer, {})
    for name, changed in [('pixel_sha256','wrong'), ('input_token_ids',[3]), ('visual_tokens',3360),
                          ('generate_calls',2), ('prefill_calls',0), ('max_new_tokens',8)]:
        bad = copy.deepcopy(answer); bad['observed'][name] = changed
        with pytest.raises(ValueError): d.validate_answer(item, bad, {})
