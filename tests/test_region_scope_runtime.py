"""exp27输入引用与恢复边界测试，禁止不确定生成再次消费调用。"""
import json
from pathlib import Path
import pytest
from videoqa_audit.region_scope_runtime import source_order,recover,METHOD
from videoqa_runtime.common import sha256,read_json


def test_source_order_ignores_internal_candidate_id_but_not_order_or_video_clock():
    a=dict(video=dict(path='v.mp4',start_pts=0,time_base='1/10',duration_fraction='3'),
        selected_frames=[dict(candidate_index=5,source_frame_index=2,source_pts=2),dict(candidate_index=7,source_frame_index=8,source_pts=8)])
    b=dict(a,selected_frames=[dict(f,candidate_index=i) for i,f in enumerate(a['selected_frames'])])
    assert source_order(a)==source_order(b)
    assert source_order(a)!=source_order(dict(b,selected_frames=list(reversed(b['selected_frames']))))
    assert source_order(a)!=source_order(dict(b,video=dict(b['video'],path='other.mp4')))


def setup_return(tmp_path,returned):
    """测试接口：建立一个已开始调用及可控返回，不创建模型或视频。"""
    (tmp_path/'protocol.json').write_text('{}');fp=sha256(tmp_path/'protocol.json')
    call=tmp_path/'calls'/METHOD/'q.json';call.parent.mkdir(parents=True);call.write_text(json.dumps(dict(status='started_may_generate',protocol_sha256=fp,gpu='7')))
    raw=tmp_path/'raw_generations'/METHOD/'q.json';raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps(dict(identity=dict(question_id='q',method=METHOD,protocol_sha256=fp,gpu='7'),generation_state=dict(started=True,returned=returned))))
    return call


def test_uncertain_generation_is_not_recovered_or_restarted(tmp_path):
    call=setup_return(tmp_path,False)
    with pytest.raises(ValueError,match='uncertain'):recover(tmp_path,dict(question_id='q'),dict(profile='llava8'))
    assert read_json(call)['status']=='started_may_generate'


def test_complete_raw_return_only_recovers_postprocessing(tmp_path,monkeypatch):
    call=setup_return(tmp_path,True)
    import videoqa_generalization.llava_observe as observed
    count=[]
    def rebuild(raw,row):
        count.append(row['question_id']);return dict(answer=dict(raw_output='A'),observed={})
    monkeypatch.setattr(observed,'result_from_raw',rebuild)
    recover(tmp_path,dict(question_id='q'),dict(profile='llava8'))
    recover(tmp_path,dict(question_id='q'),dict(profile='llava8'))
    assert count==['q'] and read_json(call)['status']=='completed'
    assert len(list((tmp_path/'calls'/METHOD).glob('*.json')))==1


def test_cpu_qwen_identity_records_processor_expanded_prompt(monkeypatch):
    import torch
    from PIL import Image
    import videoqa_audit.region_scope_runtime as runtime
    class Task:
        def videomme_doc_to_text(self,doc,kwargs):return 'question'
    class Processor:
        def apply_chat_template(self,*args,**kwargs):return 'short video placeholder'
        def __call__(self,**kwargs):
            kwargs['text'][0]='expanded video placeholders'
            return dict(input_ids=torch.tensor([[1,2,3]]),pixel_values_videos=torch.zeros(1,3),video_grid_thw=torch.tensor([[1,2,2]]))
    monkeypatch.setattr(runtime,'load_task',lambda:Task())
    monkeypatch.setattr(runtime,'task_doc',lambda row:{})
    image=Image.new('RGB',(2,2),'white')
    try:value=runtime.cpu_identity('qwen32',(Processor(),None),[image],{})
    finally:image.close()
    assert value['prompt']=='expanded video placeholders' and value['input_token_ids']==[1,2,3]
