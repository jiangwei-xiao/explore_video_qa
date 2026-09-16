"""首轮改进的边界、不变量、查询约束及恢复测试。"""
import copy
import json
import numpy as np
import pytest
from videoqa_methods.improvements import reselect_local, validate_query, lexical_tokens, compare_upstream
from videoqa_methods.followups import CallLedger, durable
from videoqa_runtime.baseline_selection import ProtocolError


def pool_fixture():
    """生成两段、4名额的小型精确PTS池；只在单元测试缩小预算。"""
    rows=[dict(candidate_index=i,source_pts=t,timestamp_seconds=float(t)) for i,t in enumerate([0,1,2,3,4,5,1.5,4.5])]
    # 所有真实PTS为整数，时间基为1/2。
    for r in rows: r['source_pts']=int(r['source_pts']*2)
    pool=dict(candidates=rows,scores=[.9,.8,.7,.9,.8,.7,.99,.99],initial_count=6,
              video=dict(time_base='1/2',start_pts=0), segments=[dict(start_pts=0,representative_score=.9),dict(start_pts=6,representative_score=.9)],
              quotas=[2,2],coarse_selected=[0,1,3,4],lambda_value=.5,
              hotspot_trace=[dict(selected=True,left_fraction='1',right_fraction='2'),dict(selected=True,left_fraction='4',right_fraction='5')])
    return pool,np.eye(8,dtype=np.float32),dict(frames=4,redundancy_coefficient=.2)


def test_local_fixed_and_quota():
    pool,f,cfg=pool_fixture(); selected,trace=reselect_local(pool,f,cfg)
    assert selected==[0,6,3,7]
    assert trace[0]['fixed_indices']==[0] and trace[1]['fixed_indices']==[3]
    assert trace[0]['steps'][0]['redundancy_reference_indices']==[0]
    assert trace[0]['steps'][0]['segment_count_before']==1


def test_inclusive_endpoints():
    p,f,c=pool_fixture(); _,t=reselect_local(p,f,c)
    assert set(t[0]['eligible_indices'])=={1,2,6}
    assert set(t[1]['eligible_indices'])=={4,5,7}


def test_no_new_identity():
    p,f,c=pool_fixture(); p['initial_count']=8
    assert reselect_local(p,f,c)[0]==p['coarse_selected']


def test_zero_local_budget():
    p,f,c=pool_fixture(); p['hotspot_trace']=[dict(selected=True,left_fraction='3/2',right_fraction='3/2')]
    assert reselect_local(p,f,c)[0]==p['coarse_selected']


def test_zero_segment_quota():
    p,f,c=pool_fixture(); p['quotas']=[4,0]; p['coarse_selected']=[0,1,2,6]
    out,trace=reselect_local(p,f,c)
    assert all(p['candidates'][i]['source_pts']<6 for i in out)
    assert trace[1]['selected_indices']==[]


def test_fixed_similarity_changes_choice():
    p,f,c=pool_fixture(); f[6]=f[0]
    out,_=reselect_local(p,f,c)
    assert 1 in out and 6 not in out


def test_no_cross_segment_redundancy():
    p,f,c=pool_fixture(); f[6]=f[3]
    assert 6 in reselect_local(p,f,c)[0]


def test_negative_gain_fills_and_ties_earliest():
    p,f,c=pool_fixture(); p['scores']=[0.]*8; f[:]=f[0]; c['redundancy_coefficient']=2.
    out,trace=reselect_local(p,f,c)
    assert len(out)==4 and out==p['coarse_selected']
    assert all(t['steps'][0]['gain']<0 for t in trace)


def test_multiwindow_same_segment_shared_budget():
    p,f,c=pool_fixture(); p['segments']=[p['segments'][0]]; p['quotas']=[4]
    out,trace=reselect_local(p,f,c)
    assert len(trace)==1 and trace[0]['replaceable_slots']==2
    assert out==[0,6,3,7]


def test_cross_segment_window_membership():
    p,f,c=pool_fixture(); p['hotspot_trace']=[dict(selected=True,left_fraction='1',right_fraction='5')]
    out,trace=reselect_local(p,f,c)
    assert len(out)==4 and trace[0]['quota']==2 and trace[1]['quota']==2
    assert all(p['candidates'][i]['source_pts']<6 for i in trace[0]['selected_indices'])


@pytest.mark.parametrize('question,query',[
    ('What did Tom do after brushing his teeth?','Tom after brushing his teeth.'),
    ('How many 6-year-old children are not running?','How many 6-year-old children not running.'),
    ('Which box is larger than the red box?','The box larger than the red box.'),
    ('What is John’s action?','John\'s action.'),
])
def test_query_preserves_content(question,query):
    assert not validate_query(question,query)['fallback']


@pytest.mark.parametrize('query',[
    'Tom brushing teeth.', 'Tom after brushing teeth on vacation.', 'Tom before brushing teeth.',
    'A. Tom after brushing teeth.', '<image> Tom after brushing teeth.',
    'Description: Tom after brushing teeth.', 'Tom after brushing teeth.\nExplanation: evidence', '',
])
def test_query_invalid_fallback(query):
    q='What did Tom do after brushing his teeth?'
    result=validate_query(q,query)
    assert result['fallback'] and result['query']==q


def test_numbers_negation_and_duplicates():
    assert validate_query('How many 5 people are not running?', '6 people running.')['fallback']
    assert validate_query('Which box is larger than the red box?', 'The red box larger.')['fallback']
    assert validate_query('What is visible?', 'visible', reached_limit=True)['fallback']
    assert lexical_tokens('JOHN’S 20 cats')==["john's",'20','cats']


def test_call_ledger_reuses_completed(tmp_path):
    ledger=CallLedger(tmp_path,'001-1','fingerprint','0'); calls=[]
    def function(state): calls.append(1); state['returned']=True; return {'label':'MIXED'}
    assert ledger.invoke('scope',function)=={'label':'MIXED'}
    assert CallLedger(tmp_path,'001-1','fingerprint','0').invoke('scope',function)=={'label':'MIXED'}
    assert len(calls)==1


def test_call_ledger_started_is_durable_before_invocation(tmp_path):
    def function(state):
        assert json.loads((tmp_path/'calls/query/001-1.json').read_text())['status']=='started_may_generate'
        raise RuntimeError('uncertain')
    ledger=CallLedger(tmp_path,'001-1','f','0')
    with pytest.raises(RuntimeError): ledger.invoke('query',function)
    with pytest.raises(ProtocolError): ledger.invoke('query',lambda s:{})


def test_call_ledger_fingerprint_guard(tmp_path):
    ledger=CallLedger(tmp_path,'001-1','f','0'); ledger.invoke('answer',lambda s:{'parsed_answer':'A'})
    with pytest.raises(ProtocolError): CallLedger(tmp_path,'001-1','changed','0').invoke('answer',lambda s:{})


def test_upstream_exact_discrete_tolerant_float():
    p,f,c=pool_fixture(); p.update(scope={'label':'MIXED'},segment_ids=[0,0,0,1,1,1,0,1])
    old=copy.deepcopy(p); old['scores'][0]+=5e-7
    assert compare_upstream(p,f,old,f)['passed']
    old['coarse_selected']=[0,2,3,4]
    assert not compare_upstream(p,f,old,f)['passed']


def test_query_generation_config_isolated_and_no_video(monkeypatch):
    """用假后端验证128Token配置只作用于查询，不执行真实GPU模型调用。"""
    import torch
    from types import SimpleNamespace
    from transformers import GenerationConfig
    from videoqa_methods.improvements import rewrite_query
    class Inputs(dict):
        def to(self,device): return self
    class Tokenizer:
        eos_token_id=2
        pad_token_id=0
        def __call__(self,prompt,return_tensors,truncation):
            assert truncation is False
            assert 'Do not answer' in prompt and '<image>' not in prompt
            return Inputs(input_ids=torch.tensor([[3,4,5]]),attention_mask=torch.ones((1,3)))
        def batch_decode(self,output,skip_special_tokens): return ['Tom after brushing his teeth.']
    class Model:
        generation_config=GenerationConfig(max_new_tokens=8,bos_token_id=1,eos_token_id=2)
        def generate(self,ids,attention_mask,generation_config):
            assert generation_config.max_new_tokens==128
            assert generation_config.do_sample is False
            assert self.generation_config.max_new_tokens==8
            return torch.tensor([[55,2]])
    backend=SimpleNamespace(tokenizer=Tokenizer(),model=Model(),context=2048)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    state={}; result=rewrite_query('What did Tom do after brushing his teeth?',backend,state)
    assert state['started'] and state['returned'] and not result['fallback']
    assert backend.model.generation_config.max_new_tokens==8


def test_proxy_normalized_rank_and_unlabelled_candidates():
    """代理只衡量已列见证，未标注帧不会被计为无关。"""
    from videoqa_methods.followups_report import fact_metrics
    p,_,_=pool_fixture(); p['selected_frames']=[p['candidates'][i] for i in p['coarse_selected']]
    metric=fact_metrics(p,[p['candidates'][6]['source_pts']])
    assert metric['best_rank']==1 and metric['normalized_rank']==0 and not metric['retained']
    assert fact_metrics(p,[999])==dict(best_rank=None,normalized_rank=None,retained=False)
