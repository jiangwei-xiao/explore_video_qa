"""相邻粗观测词确认，不让细采样复制提升观测支持。"""
from videoqa_methods.text_state_confirmation import confirm_text_states


def test_coarse_recurrence_suppresses_one_off_clock_and_ocr_noise():
    c=[dict(source_pts=i*2) for i in range(4)]
    states=[['banks','00','40'],['banks','ins','00','41'],['banks','ins','00','42','noise'],['banks','ins','00','43']]
    new,_=confirm_text_states(c,4,states)
    assert new==[['00','banks'],['00','banks','ins'],['00','banks','ins'],['00','banks','ins']]


def test_fine_repeated_tokens_do_not_replace_two_coarse_confirmations():
    c=[dict(source_pts=0),dict(source_pts=4),dict(source_pts=1),dict(source_pts=2),dict(source_pts=3)]
    states=[['true','40'],['true','41'],['true','40'],['true','40'],['true','40']]
    new,_=confirm_text_states(c,2,states)
    assert new[2:]==[['true'],['true'],['true']]


def test_single_observation_and_unbracketed_fine_return_visual_fallback():
    c=[dict(source_pts=2),dict(source_pts=1),dict(source_pts=3)]
    new,_=confirm_text_states(c,1,[['title'],['title'],['title']])
    assert new==[[],[],[]]
