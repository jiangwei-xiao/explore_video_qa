"""exp28开发/全量隔离与严格恢复测试，不加载模型。"""
import json
from pathlib import Path
import pytest
from videoqa_runtime.common import sha256,read_json
from videoqa_audit.conditional_merge_runtime import METHOD,source_order,recover


def test_stage_population_is_explicit_and_fixed(tmp_path,monkeypatch):
    import videoqa_audit.conditional_merge_runtime as r
    monkeypatch.setattr(r,'development_rows',lambda:[dict(question_id='dev')])
    monkeypatch.setattr(r,'rows',lambda:[dict(question_id='dev'),dict(question_id='other')])
    p=tmp_path/'study_stage.json';p.write_text(json.dumps(dict(population='development200')))
    assert [x['question_id'] for x in r.study_rows(tmp_path)]==['dev']
    p.write_text(json.dumps(dict(population='full2700')))
    assert [x['question_id'] for x in r.study_rows(tmp_path)]==['dev','other']


def test_full_development_reference_is_new_candidate_not_parent(tmp_path,monkeypatch):
    import videoqa_audit.conditional_merge_runtime as r
    monkeypatch.setattr(r,'development_rows',lambda:[dict(question_id='dev')])
    monkeypatch.setattr(r,'parent_result',lambda profile,q:Path('/frozen')/(q+'.json'))
    (tmp_path/'study_stage.json').write_text(json.dumps(dict(population='full2700',development_root='/development')))
    assert str(r.comparison_result(tmp_path,'qwen32','dev'))=='/development/qa/qwen32/results/'+METHOD+'/dev.json'
    assert str(r.comparison_result(tmp_path,'qwen32','other'))=='/frozen/other.json'


def test_uncertain_call_cannot_be_reissued(tmp_path):
    (tmp_path/'protocol.json').write_text('{}');fp=sha256(tmp_path/'protocol.json')
    cp=tmp_path/'calls'/METHOD/'q.json';cp.parent.mkdir(parents=True)
    cp.write_text(json.dumps(dict(status='started_may_generate',protocol_sha256=fp,gpu='0')))
    rp=tmp_path/'raw_generations'/METHOD/'q.json';rp.parent.mkdir(parents=True)
    rp.write_text(json.dumps(dict(identity=dict(question_id='q',method=METHOD,protocol_sha256=fp,gpu='0'),
        generation_state=dict(started=True,returned=False))))
    with pytest.raises(ValueError,match='uncertain'):recover(tmp_path,dict(question_id='q'),dict(profile='llava8'))
    assert read_json(cp)['status']=='started_may_generate'


def test_completed_development_import_keeps_physical_ancestry(tmp_path):
    from videoqa_audit.conditional_merge_runtime import parent_call_sources
    ancestor=tmp_path/'ancestor.json';ancestor.write_text('{}')
    result=tmp_path/'result.json';result.write_text('{}')
    old=dict(question_id='q',method=METHOD,imported=True,parent_sources={str(ancestor):sha256(ancestor)})
    paths=parent_call_sources('qwen32',result,old)
    assert paths=={str(result):sha256(result),str(ancestor):sha256(ancestor)}
    ancestor.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='ancestry'):parent_call_sources('qwen32',result,old)


def test_cpu_prompt_observation_uses_expanded_text(monkeypatch):
    import torch
    from PIL import Image
    import videoqa_audit.conditional_merge_runtime as r
    class Task:
        def videomme_doc_to_text(self,*args):return 'question'
    class Processor:
        def apply_chat_template(self,*args,**kwargs):return 'short'
        def __call__(self,**kwargs):
            kwargs['text'][0]='expanded'
            return dict(input_ids=torch.tensor([[1,2]]),pixel_values_videos=torch.zeros(1,3),video_grid_thw=torch.tensor([[1,2,2]]))
    monkeypatch.setattr(r,'load_task',lambda:Task());monkeypatch.setattr(r,'task_doc',lambda row:{})
    image=Image.new('RGB',(2,2),'white')
    try:assert r.cpu_identity('qwen32',(Processor(),None),[image],{})['prompt']=='expanded'
    finally:image.close()
