"""两方法固定协议入口验证，不调用模型、不改历史选择。"""
from pathlib import Path
import pytest
from videoqa_full.state import durable, invoke_once
from videoqa_lmms_baseline.methods import METHODS, GPUS, method_order, selection


def test_scope_and_order():
    assert GPUS == ('4','5','6','7')
    assert METHODS == ('BASE-BLIP-TopK','RD-1.2')
    assert method_order(0)==METHODS and method_order(1)==METHODS[::-1] and method_order(2)==METHODS


@pytest.mark.parametrize('n',[1,11,14,16])
def test_preserve_frozen_selection(tmp_path,n):
    from videoqa_runtime.common import sha256
    s=dict(video=dict(time_base='1/10',start_pts=0,duration_seconds=20),selected_frames=[
        dict(source_pts=i,source_frame_index=i,timestamp_seconds=i/10) for i in range(n)])
    rel='selections/RD-1.2/001-1.json';durable(tmp_path/rel,dict(selection=s))
    spec=dict(selection_exports={rel:sha256(tmp_path/rel)})
    assert selection(tmp_path,spec,'001-1','RD-1.2')==s
    durable(tmp_path/rel,dict(selection=s,changed=True))
    with pytest.raises(AssertionError):selection(tmp_path,spec,'001-1','RD-1.2')


def test_method_call_isolation_and_resume(tmp_path):
    seen=[]
    def call(state):
        seen.append(1);state.update(started=True,returned=True);return {'raw_output':'A'}
    for m in METHODS:
        assert invoke_once(tmp_path,'001-1',m,'hash','4',call)[1] is False
        assert invoke_once(tmp_path,'001-1',m,'hash','4',call)[1] is True
    assert len(seen)==2
    with pytest.raises(RuntimeError):invoke_once(tmp_path,'001-1',METHODS[0],'changed','4',call)


def test_uncertain_generation_never_repeats(tmp_path):
    seen=[]
    def fail(state):
        seen.append(1);state['started']=True;raise RuntimeError('uncertain')
    for _ in range(2):
        with pytest.raises(RuntimeError):invoke_once(tmp_path,'001-1',METHODS[0],'hash','4',fail)
    assert len(seen)==1
