"""监督敏感性诊断：标签隔离、固定候选、一次生成与原始输入失败边界。"""
import copy
import hashlib
from pathlib import Path

import pytest

from videoqa_full.state import durable
from videoqa_runtime.common import read_json, sha256
from videoqa_audit import supervision_quality as d
from videoqa_audit.supervision_quality_report import bank_statistics, paired_comparison, choose_panel


def rows():
    """接口：300视频的公开训练题，答案仅作为投影隔离测试的毒化字段。"""
    return [dict(question_id=f'q{v}_{i}',split='train',video_id=str(v),video_sha256=str(v),video_path=f'{v}.mp4',
                 index_path=f'{v}.json',question='What?',options=list('ABCDE'),task_type='CW',answer_index=i)
            for v in range(300) for i in range(2)]


def frames(n=15):
    return [dict(candidate_index=i,source_pts=i*100+25,source_frame_index=i*30,timestamp_seconds=i+.25) for i in range(n)]


def test_panel_and_donors_ignore_answers_and_input_order():
    data=rows();public=d.public_training_rows(data);duration={str(v):float(v+8) for v in range(300)}
    a=d.select_panel(public,duration)
    poisoned=copy.deepcopy(data)
    for r in poisoned:r['answer_index']=4-r['answer_index'];r['prediction']='A'
    b=d.select_panel(list(reversed(d.public_training_rows(poisoned))),duration)
    assert a==b and len(a)==120 and len({c['row']['video_id'] for c in a})==120
    assert all('answer_index' not in c['row'] and 'prediction' not in c['row'] for c in a)
    assert all(c['row']['video_sha256']!=c['donor']['video_sha256'] for c in a)


def test_donor_nearest_duration_excludes_same_content():
    public=d.public_training_rows(rows());public[2]['video_sha256']=public[3]['video_sha256']='0'
    panel=d.select_panel(public,{str(v):float(v) for v in range(300)})
    for c in panel:
        if c['row']['video_id']=='0':assert c['donor']['video_id']=='2'
    with pytest.raises(ValueError,match='public'):
        d.select_panel(rows(),{str(v):float(v) for v in range(300)})


@pytest.mark.parametrize('peak,expected',[(0,list(range(8))),(7,list(range(4,12))),(14,list(range(7,15)))])
def test_local_block_anchor_and_clipped_boundaries(peak,expected):
    f=frames();score=[0.]*15;score[peak]=1.
    assert [x['candidate_index'] for x in d.candidate_groups(f,score,'q')['L']]==expected


def test_ties_and_hash_random_use_absolute_source_identity():
    f=frames();s=[.5]*15;a=d.candidate_groups(f,s,'q')
    assert a['L']==f[:8]
    b=d.candidate_groups(list(reversed(f)),list(reversed(s)),'q')
    assert a==b and len({x['source_pts'] for x in a['R']})==8
    expected=sorted(f,key=lambda x:hashlib.sha256(('2027/supervision_quality_v1/random/q/'+str(x['source_pts'])+'/'+str(x['source_frame_index'])).encode()).hexdigest())[:8]
    assert a['R']==sorted(expected,key=lambda x:x['source_pts'])


def test_short_or_duplicate_source_pool_rejected():
    with pytest.raises(ValueError):d.candidate_groups(frames(7),[1]*7,'q')
    f=frames();f[1]['source_pts']=f[0]['source_pts']
    with pytest.raises(ValueError,match='Duplicate'):d.candidate_groups(f,[1]*15,'q')


def test_groups_share_only_full_input_identity_and_keep_O_first():
    entries={c:dict(input_key=c) for c in d.CONDITIONS};entries['R']['input_key']='L'
    groups=d.group_conditions(entries)
    assert len(groups)==4 and groups[-1]['conditions']==['L','R']
    assert all(d.ordered_groups(groups,i)[0]['representative']=='O' for i in range(8))
    record=dict(frames=[dict(source_pts=1,source_frame_index=2,rgb_sha256='rgb')])
    a=d.condition_key(record,dict(prompt='q',pixel='one'))
    assert a!=d.condition_key(record,dict(prompt='q',pixel='two'))
    assert a!=d.condition_key(record,dict(prompt='other',pixel='one'))


def test_gray_sources_are_explicitly_artificial_and_can_repeat_pixels():
    record=dict(artificial=True,frames=[dict(rgb_sha256='same') for _ in range(8)])
    assert d.source_order(record)==[['artificial_gray',i] for i in range(8)]


@pytest.mark.parametrize('raw,expected',[(' C. ',True),('(C)',True),('D',False),('C. because',False),('',False)])
def test_original_reproduction_uses_strict_parsed_option(raw,expected):
    assert d.historical_reproduced(raw,'C') is expected


def test_failed_original_skips_all_remaining_conditions(tmp_path,monkeypatch):
    q='q';row=dict(question_id=q);case=dict(row=row,ordinal=0)
    durable(tmp_path/'evaluation.json',{q:dict(answer_index=0,gpu='1',historical={d.METHODS[0]:dict(parsed_answer='A')})})
    durable(tmp_path/'inputs'/f'{q}.json',dict(entries={'O':dict(input_key='O')}))
    durable(tmp_path/'inputs_complete.json',dict(exports={q:sha256(tmp_path/'inputs'/f'{q}.json')}))
    calls=[]
    def generate(*args,**kwargs):calls.append(args[2]);return dict(raw_output='B',parsed_answer='B')
    monkeypatch.setattr(d,'_generate',generate);monkeypatch.setattr(d,'publish_result',lambda *a:None)
    d.process_question(tmp_path,case,None,None,'1')
    assert calls==['O']
    assert all(read_json(tmp_path/'results'/c/'q.json')['status']=='skipped_original_not_reproduced' for c in d.CONDITIONS[1:])


def test_completed_call_restores_without_loading_images_or_backend(tmp_path,monkeypatch):
    durable(tmp_path/'qa/protocol.json',{'test':'fingerprint'})
    expected=dict(prompt='q',input_token_ids=[1],pixel_sha256='x',source_frames=8,visual_tokens=1680)
    answer=dict(raw_output='A',observed=dict(expected,max_new_tokens=16,generate_calls=1,prefill_calls=1))
    durable(tmp_path/'qa/calls/O/q.json',dict(protocol_sha256=sha256(tmp_path/'qa/protocol.json'),gpu='0',status='completed',answer=answer))
    def forbidden(*args):raise AssertionError('Restored completed call touched media/model')
    monkeypatch.setattr(d,'_load_images',forbidden)
    assert d._generate(tmp_path,dict(question_id='q'),'O',dict(identity=expected),None,None,'0')==answer


def test_bank_dedup_and_more_negatives_do_not_raise_oracle():
    four=bank_statistics([[1,1,1,1]],[['a','b','c','d']])
    six=bank_statistics([[1,1,1,1,0,0]],[['a','b','c','d','e','e']])
    assert four['mixed_questions']==0 and six['mixed_questions']==1
    assert four['any_candidate_correct']==six['any_candidate_correct']==1
    assert six['unique_preference_pairs']==4 and six['duplicate_candidate_inputs']==1
    with pytest.raises(ValueError,match='conflicting'):bank_statistics([[1,0]],[['a','a']])


def test_video_bootstrap_and_four_grid_preserve_pairing():
    rs=[dict(question_id=f'q{i}',video_id=str(i)) for i in range(4)]
    p=paired_comparison(rs,[1]*4,[0]*4)
    assert p['interval_95']==[1.,1.] and p['net_correct']==4 and len(p['new_correct'])==4
    with pytest.raises(ValueError):paired_comparison([rs[0]]*4,[1]*4,[0]*4)


def test_review_slots_empty_without_backfill():
    item=dict(question_id='q',video_id='v',reproduced=True,historical_state='all_correct',
              results={c:dict(correct=True) for c in d.CONDITIONS})
    p=choose_panel([item])
    assert len(p['cases'])==1 and p['cases'][0]['category']=='controls_still_correct'
    assert p['slots']['new_preference']['empty_slots']==4
