"""全量适配的纯CPU验收：调用隔离、短池、精确时间和视频簇统计。"""
from copy import deepcopy
import numpy as np
import pytest
from videoqa_full.state import (METHODS,durable,read_json,invoke_once,pilot_ids,
                               method_order,validate_frames,exclusive,digest)
from videoqa_full.qwen_backend import padded_count,qa_text
from videoqa_full.report import cluster_interval,compare
from videoqa_full.run import replay,recovery


@pytest.mark.parametrize('n,expected',[(1,2),(2,2),(11,12),(14,14),(15,16),(16,16)])
def test_internal_padding_does_not_change_budget(n,expected):
    assert padded_count(n)==expected


@pytest.mark.parametrize('n',[0,17,-1])
def test_bad_budget(n):
    with pytest.raises(ValueError):padded_count(n)


def test_original_qa_text_retained():
    from videoqa_runtime.protocol import question_text
    args=('What follows?', ['A. x','B. y','C. z','D. w'],12,[.25,3.75,9.25])
    assert '<image>\n'+qa_text(*args)==question_text(*args)
    assert '0.25s, 3.75s, 9.25s' in qa_text(*args)


def test_order_and_pilot():
    assert method_order('llava',5)==('RD-1.2',)
    assert set(method_order('qwen25vl',1))==set(METHODS)
    assert method_order('qwen25vl',1)==METHODS[1:]+METHODS[:1]
    assert pilot_ids([dict(question_id=str(i)) for i in range(5)])==['0','1','2','3','4','133-1','134-1']


def finish(state):
    state.update(started=True,returned=True)
    return dict(raw_output='A',parsed_answer='A')


def test_success_recovery_never_calls_twice(tmp_path):
    first=invoke_once(tmp_path/'llava','001-1','RD-1.2','h','0',finish)
    def forbidden(state):raise AssertionError('Must not generate')
    second=invoke_once(tmp_path/'llava','001-1','RD-1.2','h','0',forbidden)
    assert first[0]==second[0] and second[1]


def test_failed_or_uncertain_call_keeps_budget(tmp_path):
    def failing(state):
        state['started']=True
        raise RuntimeError('synthetic failure')
    with pytest.raises(RuntimeError):invoke_once(tmp_path/'qwen25vl','001-1','RD-1.2','h','0',failing)
    with pytest.raises(RuntimeError):invoke_once(tmp_path/'qwen25vl','001-1','RD-1.2','h','0',finish)
    assert len(list(tmp_path.glob('**/calls/**/*.json')))==1


def test_call_scope_and_fingerprint(tmp_path):
    for model in ('llava','qwen25vl'):
        invoke_once(tmp_path/model,'001-1','RD-1.2','h','0',finish)
    invoke_once(tmp_path/'qwen25vl','001-1','BASE-Uniform','h','0',finish)
    assert len(list(tmp_path.glob('**/calls/**/*.json')))==3
    with pytest.raises(RuntimeError):invoke_once(tmp_path/'llava','001-1','RD-1.2','other','0',finish)


def test_exclusive_lock(tmp_path):
    with exclusive(tmp_path/'lock'):
        with pytest.raises(BlockingIOError):
            with exclusive(tmp_path/'lock'):pass


def selection_fixture(n):
    from videoqa_methods.density_selection import build_regions
    from videoqa_audit.density_reward_decay import select_density_decay
    from videoqa_audit.local_distance_scale import select_local_distance
    rows=[dict(candidate_index=i,source_pts=1000+1000*i+250,source_frame_index=i*27+5,
               timestamp_seconds=i+.25,requested_seconds=i+.25) for i in range(n)]
    video=dict(time_base='1/1000',start_pts=1000,duration_fraction=str(n),duration_seconds=float(n))
    cfg=dict(seed_budget=16,frame_budget=16,core_gap_seconds='2',window_radius_seconds='4',refinement_min_core_seeds=3)
    scores=[1-i/(n+1) for i in range(n)];plan=build_regions(rows,scores,video,cfg)
    f=np.random.default_rng(2027).normal(size=(n,8)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True)
    _,budget=select_density_decay(rows,f,plan,video,cfg)
    ids,trace=select_local_distance(rows,f,plan,video,cfg,budget['region_budgets'])
    return dict(candidates=rows,plan=plan,video=video,selected_indices=ids,selected_frames=[rows[i] for i in ids],
                budget_trace=budget,selection_trace=trace),f,cfg


@pytest.mark.parametrize('n',[1,11,14,16,25])
def test_generic_rd_replay_short_and_normal(n):
    s,f,cfg=selection_fixture(n)
    assert replay(s,f,cfg)['effective_budget']==min(n,16)


def test_vfr_pts_not_average_frame_number():
    s,_,_=selection_fixture(11)
    validate_frames(s)
    bad=deepcopy(s);bad['selected_frames'][1]['timestamp_seconds']+=.1
    with pytest.raises(AssertionError):validate_frames(bad)
    bad=deepcopy(s);bad['selected_frames'][1]=bad['selected_frames'][0]
    with pytest.raises(AssertionError):validate_frames(bad)


def test_selection_trace_tampering():
    s,f,cfg=selection_fixture(25);s['selected_indices']=list(reversed(s['selected_indices']))
    with pytest.raises(AssertionError):replay(s,f,cfg)


def test_cluster_bootstrap_keeps_video_questions_together():
    data=[dict(question_id=f'{v}-{i}',video_id=str(v),stratum='short') for v in range(2) for i in range(3)]
    left={r['question_id']:dict(correct=r['video_id']=='0') for r in data}
    right={r['question_id']:dict(correct=False) for r in data}
    assert cluster_interval(data,left,right,1000)==[0.,100.]
    v=compare(data,left,right)
    assert v['difference_pp']==50 and len(v['cells']['left_only'])==3


def test_unequal_cluster_subset_bootstrap_denominator():
    data=[dict(question_id='a',video_id='a',stratum='short'),dict(question_id='b1',video_id='b',stratum='short'),
          dict(question_id='b2',video_id='b',stratum='short')]
    records={r['question_id']:dict(correct=True) for r in data}
    assert cluster_interval(data,records,records,1000)==[0.,0.]


def test_recovery_rejects_unknown_call(tmp_path):
    durable(tmp_path/'qwen25vl/calls/RD-1.2/unknown.json',dict(method='RD-1.2',model='qwen25vl',
            protocol_sha256='h',status='completed'))
    with pytest.raises(AssertionError):recovery(tmp_path,'qwen25vl',[dict(question_id='001-1')],{},'h')


def test_source_identity_hash_binds_question_and_options():
    assert digest(['question',['a','b','c','d']])!=digest(['question',['b','a','c','d']])
