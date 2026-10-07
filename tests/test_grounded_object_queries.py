"""空白框弃权、暗场保护和文本忠实性测试，不运行检测/问答模型。"""
import numpy as np
from PIL import Image
from videoqa_audit.grounded_object_queries import box_structure,parse_object_queries,object_extraction_prompt


def test_white_or_uniform_dark_box_is_abstention_not_positive_object():
    for color in ('white','black','gray'):
        assert not box_structure(Image.new('RGB',(20,20),color),[0,0,20,20])['grounded_visual_structure']


def test_dark_scene_with_real_structure_is_not_filtered():
    a=np.zeros((20,20,3),dtype=np.uint8);a[3:10,5:11]=[8,6,3]
    assert box_structure(Image.fromarray(a),[0,0,20,20])['grounded_visual_structure']


def test_box_canvas_clipping_and_outside_box():
    assert box_structure(Image.new('RGB',(20,20),'white'),[-10,-10,4,4])['clipped_box']==[0,0,4,4]
    assert box_structure(Image.new('RGB',(20,20),'white'),[22,22,30,30])['reason']=='outside_visible_canvas'


def test_parse_licensed_transport_and_no_answer():
    raw='{"known":["person"],"choices":[["person"],["bicycle"],["skateboard"],["vehicle"]]}'
    v=parse_object_queries(raw,'How does the person commute?', ['Walking.','Biking.','Skateboarding.','Driving.'])
    assert v['valid'] and v['choices'][2]==['skateboard']
    assert not parse_object_queries(raw,'What is the color?', ['red','blue','green','white'])['valid']


def test_unlicensed_unknown_fact_and_explanation_rejected():
    assert parse_object_queries('{"known":["fork"],"choices":[[],[],[],[]]}','What is in the box?',['book','coin','hat','key'])['fallback']
    assert parse_object_queries('The answer is C.','What?',['a','b','c','d'])['fallback']


def test_prompt_has_all_choices_and_no_groundtruth_parameter():
    text=object_extraction_prompt('Where is the gun?',['jacket','box','car','bag'])
    assert all(x in text for x in ['jacket','box','car','bag'])
    assert 'Do not answer' in text
