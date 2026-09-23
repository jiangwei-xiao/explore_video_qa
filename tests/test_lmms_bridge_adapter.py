"""仅在独立框架环境运行的真实API/任务函数验收，不进行模型加载。"""
import pytest
pytest.importorskip('lmms_eval')
from videoqa_lmms_bridge.common import QUESTION_IDS,GENERATION,POST_PROMPT,question_rows,task_doc,load_task,manifest_rows


def test_exact_frozen_rows_and_reference_fields():
    assert len(QUESTION_IDS)==len(set(QUESTION_IDS))==10
    assert len(question_rows())==10
    for r in manifest_rows():
        d=task_doc(r)
        assert all(d[k]==r['original'][k] for k in d)


def test_reference_prompt_without_extra_time_or_subtitle():
    task=load_task();r=question_rows()[0]
    text=task.videomme_doc_to_text(task_doc(r),{'post_prompt':POST_PROMPT})
    assert r['question'] in text and all(o in text for o in r['options'])
    assert 'The video lasts for' not in text and 'subtitles' not in text
    assert text.endswith(POST_PROMPT) and GENERATION['max_new_tokens']==16


def test_framework_generate_is_not_reimplemented():
    from lmms_eval.models.simple.llava_vid import LlavaVid
    from videoqa_lmms_bridge.llava_adapter import FrozenInputLlavaVid
    assert FrozenInputLlavaVid.generate_until is LlavaVid.generate_until


def test_native_indices_match_actual_framework_loader(monkeypatch):
    import numpy as np
    import lmms_eval.models.simple.llava_vid as module
    from videoqa_lmms_bridge.frames import native_indices
    seen=[]
    class Reader:
        def __init__(self,*a,**k):pass
        def __len__(self):return 103
        def get_avg_fps(self):return 30.
        def get_batch(self,ids):
            seen.extend(ids)
            class Batch:
                def asnumpy(self):return np.zeros((len(ids),2,2,3),dtype=np.uint8)
            return Batch()
    monkeypatch.setattr(module,'VideoReader',Reader)
    module.LlavaVid.load_video(None,'synthetic.mp4',16,1,force_sample=True)
    assert seen==native_indices(103)


@pytest.mark.parametrize('raw,expected',[('The best answer is B.','B'),('C','C'),('unknown','')])
def test_real_task_parser(raw,expected):
    assert load_task().extract_characters_regex(raw)==expected
