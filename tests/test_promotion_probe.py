"""辅助推广分类边界：只允许完整标签，拒绝把原题选项/答案作为输入。"""
import pytest
from videoqa_audit.promotion_probe import classify_output,inspect_frame

@pytest.mark.parametrize('text,label,decision',[('A','A','exclude_candidate'),(' b. ','B','keep'),('C','C','keep'),('D','D','keep')])
def test_complete_labels(text,label,decision):
    r=classify_output(text);assert r['category']==label and r['decision']==decision and r['parse_status']=='valid'

@pytest.mark.parametrize('text',['','A because it is promotional','Answer: A','AB','PROMO'])
def test_malformed_output_kept(text):
    assert classify_output(text)==dict(category='D',decision='keep',parse_status='invalid_fallback')

def test_adapter_single_frame_without_position_or_original_options():
    class Backend:
        def answer(self,frames,times,duration,question,options,generation_state):
            assert frames==['image'] and times==[0.0] and duration==1.0
            assert 'What happens?' in question and 'Do NOT answer' in question and len(options)==4
            generation_state.update(started=True,returned=True)
            return dict(raw_output='C')
    state={};r=inspect_frame(Backend(),'image','What happens?',state)
    assert r['decision']=='keep' and state==dict(started=True,returned=True)
