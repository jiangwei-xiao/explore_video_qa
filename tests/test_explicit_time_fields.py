"""字段格式与只读提议回归，不使用答案或真实模型前向。"""
import pytest
from videoqa_audit.explicit_time_fields import question_mode,extract_time_fields,propose_time_regions


def regions(text,score=0.95):
    return [dict(text=text,score=score,box=[[0,0],[200,0],[200,40],[0,40]])]


@pytest.mark.parametrize('text,key',[
    ('10:00 AM','time:10:00:am'),('7:30pm','time:07:30:pm'),('09:24','time:09:24:?'),
    ('School starts at8.30 and then','time:08:30:?'),('8pm meds','time:08:?:pm'),
    ('09：24','time:09:24:?'),
])
def test_literal_formats_without_unknown_filling(text,key):
    found=extract_time_fields(regions(text),'clock')
    assert [x['key'] for x in found]==[key]
    assert not found[0]['semantic_attribution_verified']


@pytest.mark.parametrize('text',['PLEASE SUBSCRIBE','24:30','12:68 PM','$9.24','0:00 AM','123:45'])
def test_invalid_or_unrelated_format_not_forced_into_time(text):
    assert extract_time_fields(regions(text),'clock')==[]


def test_ocr_confidence_is_fixed_default_and_invalid_values_fail():
    assert extract_time_fields(regions('09:24',0.49),'clock')==[]
    assert extract_time_fields(regions('09:24',0.5),'clock')
    with pytest.raises(ValueError):extract_time_fields(regions('09:24',float('nan')),'clock')


def test_duration_does_not_claim_clock_or_price_semantics():
    assert extract_time_fields(regions('20 minutes'),'duration')[0]['key']=='duration:20:minute'
    assert extract_time_fields(regions('20 minutes'),'clock')==[]
    assert extract_time_fields(regions('25:11'),'duration')[0]['key']=='time:25:11:?'
    assert extract_time_fields(regions('25:11'),'clock')==[]
    assert extract_time_fields(regions('0.25'),'duration')==[]
    assert extract_time_fields(regions('0.25'),'clock')[0]['decimal_ambiguity']


def test_qualitative_literal_question_is_not_numeric_claim():
    assert question_mode('What time of day is depicted?')=='qualitative'
    assert extract_time_fields(regions('8pm'),'qualitative')==[]
    assert question_mode('How long does it take?')=='duration'
    assert question_mode('What time do classes begin?')=='clock'
    assert question_mode('How many people are there?') is None


def inputs(times=range(12)):
    video=dict(path='synthetic',time_base='1',start_pts=0,duration_fraction='20')
    c=[dict(candidate_index=i,source_pts=t,source_frame_index=t*24,timestamp_seconds=float(t),requested_seconds=float(t)) for i,t in enumerate(times)]
    return video,c


def test_repeated_field_region_context_endpoints_and_unique_source():
    v,c=inputs();f=[[] for _ in c]
    for i in (5,6):f[i]=extract_time_fields(regions('09:24'),'clock')
    s=[0.1]*len(c);s[1]=s[10]=0.9
    result=propose_time_regions(c,s,f,v)
    assert len(result['regions'])==len(result['proposals'])==1
    region=result['regions'][0]
    assert region['members']==[5,6]
    assert region['context_indices']==list(range(1,11))
    assert region['context_anchor_index']==1
    assert result['proposals'][0]['candidate_index']==5
    assert result['selection_writes']==0


def test_vfr_context_uses_source_time_not_sampling_indices():
    v,c=inputs([0,1,9,10]);f=[[],extract_time_fields(regions('09:24'),'clock'),[],[]]
    result=propose_time_regions(c,[0.1,0.2,1.0,0.5],f,v)
    assert result['regions'][0]['context_indices']==[0,1]
    assert result['proposals'][0]['context_score']==0.2


def test_missing_detection_splits_not_fills_field_region():
    v,c=inputs();f=[[] for _ in c]
    for i in (5,7):f[i]=extract_time_fields(regions('09:24'),'clock')
    result=propose_time_regions(c,[0.1]*len(c),f,v)
    assert [r['members'] for r in result['regions']]==[[5],[7]]


def test_scores_and_sources_must_be_frozen_and_ordered():
    v,c=inputs()
    with pytest.raises(ValueError):propose_time_regions(c,[float('nan')]*len(c),[[]]*len(c),v)
    with pytest.raises(ValueError):propose_time_regions(c[::-1],[0.1]*len(c),[[]]*len(c),v)
