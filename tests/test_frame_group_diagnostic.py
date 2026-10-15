"""单槽位诊断回归：构造、任务隔离、O门槛及同进程恢复边界，不运行模型。"""
import copy
from fractions import Fraction
from unittest.mock import patch

import pytest

from videoqa_audit import frame_group_diagnostic as diag
from videoqa_full.state import durable


def fixture_inputs():
    """接口：合成合法有理PTS池；候选只用于纯函数测试，不进入研究输入。"""
    video = dict(start_pts=0, time_base='1/10', duration_fraction='12', duration_seconds=12)
    def frame(p):
        return dict(source_pts=p, source_frame_index=p, timestamp_seconds=float(Fraction(p,10)), candidate_index=p)
    selection = dict(video=video, selected_frames=[frame(p) for p in (10,20,60,90)])
    pool = dict(candidates=[frame(p) for p in (10,20,25,59,60,90)])
    case = dict(id='test', profile='llava8', removed_pts=60, clue_pts=20, target_pts=25, control_pts=59)
    return case, selection, pool


def test_single_slot_protection_and_no_mutation():
    case, selection, pool = fixture_inputs(); before = copy.deepcopy(selection)
    out, slot = diag.build_conditions(case, selection, pool)
    assert slot == 2 and selection == before and out['O'] == selection
    for c in ('T','C'):
        assert [i for i,(a,b) in enumerate(zip(out['O']['selected_frames'],out[c]['selected_frames'])) if a != b] == [2]
        assert out[c]['selected_frames'][1]['source_pts'] == 20


@pytest.mark.parametrize('replacement', [20, 90, 100, 10])
def test_duplicate_absent_or_out_of_slot_rejected(replacement):
    case, selection, pool = fixture_inputs(); case['target_pts'] = replacement
    with pytest.raises(ValueError): diag.build_conditions(case, selection, pool)


def test_wrong_slot_after_clue_rejected():
    case, selection, pool = fixture_inputs(); case['clue_pts'] = 10
    with pytest.raises(ValueError): diag.build_conditions(case, selection, pool)


def test_qwen_pairs_and_internal_padding():
    _, selection, _ = fixture_inputs()
    assert diag.temporal_pairs(selection) == [[10,20],[60,90]]
    selection['selected_frames'].pop()
    assert diag.temporal_pairs(selection) == [[10,20],[60,60]]


def test_task_specific_period_normalization():
    answer = dict(raw_output=' C. ', parsed_answer=None)
    assert diag.canonical_option(dict(dataset='mlvu'), answer) == 'C'
    assert diag.canonical_option(dict(dataset='videomme'), answer) is None
    assert diag.canonical_option(dict(dataset='mlvu'), dict(raw_output='D. Bicycle',parsed_answer=None)) is None


def test_fixed_order_gpu_and_budget_population():
    c = diag.cases()
    assert len(c) == 5 and [x['gpu'] for x in c] == ['2','0','1','6','7']
    assert [x['order'] for x in c] == [['O','T','C'],['O','C','T'],['O','T','C'],['O','C','T'],['O','T','C']]


def test_cpu_dispatch_task_isolation():
    with patch('videoqa_mlvu.backends.cpu_identity',return_value='mlvu') as m, patch('videoqa_audit.region_scope_runtime.cpu_identity',return_value='videomme') as v:
        assert diag.cpu_input(dict(dataset='mlvu',profile='llava8'),None,None,None) == 'mlvu'
        assert diag.cpu_input(dict(dataset='videomme',profile='llava8'),None,None,None) == 'videomme'
        assert m.call_count == v.call_count == 1


def test_partial_process_cannot_continue(tmp_path):
    case = diag.cases()[0]
    (tmp_path/'calls/O').mkdir(parents=True); (tmp_path/'calls/O/started.json').write_text('{}')
    with patch.object(diag,'verify',return_value=dict(case=case)), patch.object(diag,'recover_returns',return_value=None):
        with pytest.raises(RuntimeError,match='cross-process'):
            diag.worker(tmp_path)


def test_original_mismatch_skips_only_other_conditions(tmp_path):
    case = diag.cases()[0]; spec = dict(case=case,row=dict(answer_index=3))
    durable(tmp_path/'protocol.json',spec)
    for condition in ('O','T','C'):
        durable(tmp_path/'inputs'/(condition+'.json'),dict(no_effective_input_difference=False))
    itempath = tmp_path/'inputs/O.json'
    answer = dict(raw_output='D',parsed_answer='D')
    durable(tmp_path/'results/O.json',dict(answer=answer,option='D',correct=True,native_correct=True,input_sha256=diag.sha256(itempath)))
    for condition,qid in (('O',case['question_id']),('synthetic','warmup')):
        call=dict(question_id=qid,method=condition,gpu=case['gpu'],pid=42,status='completed',answer=answer,protocol_sha256=diag.sha256(tmp_path/'protocol.json'))
        durable(tmp_path/'calls'/condition/(qid+'.json'),call)
        durable(tmp_path/'raw_generations'/condition/(qid+'.json'),dict(identity=dict(question_id=qid,method=condition,gpu=case['gpu'],protocol_sha256=call['protocol_sha256']),generation_state=dict(started=True,returned=True)))
    with patch.object(diag,'validate_answer'):
        value = diag.finish_case(tmp_path)
        assert value['historical_replay_matches'] is False
        assert value['real_calls'] == 1 and value['skipped'] == dict(T='original_did_not_reproduce_history',C='original_did_not_reproduce_history')
        before = {str(p):diag.sha256(p) for p in tmp_path.glob('calls/*/*.json')}
        diag.finish_case(tmp_path)
        assert before == {str(p):diag.sha256(p) for p in tmp_path.glob('calls/*/*.json')}


def test_uncertain_return_refuses_repetition(tmp_path):
    case = diag.cases()[0]
    durable(tmp_path/'protocol.json', dict(case=case))
    durable(tmp_path/'calls/O'/(case['question_id']+'.json'), dict(method='O',gpu=case['gpu'],protocol_sha256=diag.sha256(tmp_path/'protocol.json')))
    with patch.object(diag,'verify',return_value=dict(case=case)):
        with pytest.raises(RuntimeError,match='no automatic retry'): diag.recover_returns(tmp_path)
