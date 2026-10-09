"""OCR完整返回先持久化，语义验收失败不丢失原始记录。"""
import pytest
from videoqa_runtime.common import read_json


def test_synthetic_raw_returns_survive_failed_state_acceptance(tmp_path,monkeypatch):
    import videoqa_audit.text_state_experiments as e
    class Process:
        def __init__(self,log):self.n=0
        def infer(self,rgb):
            self.n+=1
            return dict(regions=[] if self.n==1 else [dict(text='same',score=1.0,box=[])])
        def close(self):pass
    monkeypatch.setattr(e,'verify',lambda root:{})
    monkeypatch.setattr(e,'OCRProcess',Process)
    with pytest.raises(ValueError,match='distinction failed'):e.synthetic(tmp_path)
    for name in ('blank','alpha','beta'):
        assert read_json(tmp_path/'synthetic'/(name+'.json'))['status']=='completed'
    assert read_json(tmp_path/'synthetic/beta.json')['ocr']['regions'][0]['text']=='same'
    assert not (tmp_path/'synthetic_acceptance.json').exists()
