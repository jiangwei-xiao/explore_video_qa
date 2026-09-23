"""Qwen正式入口的范围、源帧身份与原始生成恢复测试，无GPU调用。"""
import pytest
from videoqa_runtime.common import sha256,read_json
from videoqa_full.state import durable,invoke_once
from videoqa_lmms_qwen.run import METHODS,GPUS,method_order,selection,recover_call
from videoqa_lmms_qwen.observe import clean_raw


def test_method_and_gpu_scope():
    assert METHODS==('BASE-Uniform','BASE-BLIP-TopK','RD-1.2') and GPUS==('4','5','6','7')
    assert [method_order(i) for i in range(3)]==[METHODS,METHODS[1:]+METHODS[:1],METHODS[2:]+METHODS[:2]]


@pytest.mark.parametrize('n',[1,11,14,16])
def test_source_frames_preserved(tmp_path,n):
    video=dict(time_base='1/10',start_pts=0,duration_seconds=20)
    frames=[dict(source_pts=i,source_frame_index=i,timestamp_seconds=i/10) for i in range(n)]
    item=dict(selection=dict(video=video,selected_frames=frames),rgb_sha256=['rgb']*n)
    rel='selections/BASE-Uniform/001-1.json';durable(tmp_path/rel,item)
    spec=dict(selection_exports={rel:sha256(tmp_path/rel)})
    assert selection(tmp_path,spec,'001-1','BASE-Uniform')==item
    durable(tmp_path/rel,dict(item,changed=True))
    with pytest.raises(AssertionError):selection(tmp_path,spec,'001-1','BASE-Uniform')


def test_raw_postprocessing_uses_reference_stops():
    assert clean_raw(dict(raw_generated_text='B<stop>other',stop_terms=['<stop>']))=='B'


def test_recover_finished_generation_without_new_call(tmp_path):
    row=dict(question_id='001-1',video_id='video',question='Which color?',options=['A. Red','B. Blue','C. Green','D. White'],
        answer_index=1,stratum='short',domain='Knowledge',original={'sub_category':'Humanity & History'},task_type='Object Recognition')
    durable(tmp_path/'protocol.json',{'model':'test'});fp=sha256(tmp_path/'protocol.json');m=METHODS[0]
    call=dict(status='failed_or_uncertain',protocol_sha256=fp,gpu='4')
    durable(tmp_path/'calls'/m/'001-1.json',call)
    raw=dict(identity=dict(question_id='001-1',method=m,protocol_sha256=fp,gpu='4'),
        generation_state=dict(started=True,returned=True),raw_generated_text='B',stop_terms=[],observed={})
    durable(tmp_path/'raw_generations'/m/'001-1.json',raw)
    recover_call(tmp_path,row,m,{})
    got,reused,_=invoke_once(tmp_path,'001-1',m,fp,'4',lambda _:pytest.fail('must not generate again'))
    assert reused and got['parsed_answer']=='B'
    assert read_json(tmp_path/'calls'/m/'001-1.json')['recovered_postprocessing']


def test_uncertain_generation_without_raw_stays_blocked(tmp_path):
    durable(tmp_path/'protocol.json',{});fp=sha256(tmp_path/'protocol.json')
    durable(tmp_path/'calls'/METHODS[0]/'001-1.json',dict(status='failed_or_uncertain',protocol_sha256=fp,gpu='4'))
    with pytest.raises(FileNotFoundError):recover_call(tmp_path,{'question_id':'001-1'},METHODS[0],{})
