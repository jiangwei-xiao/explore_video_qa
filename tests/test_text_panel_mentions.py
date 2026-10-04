"""选项词面核查边界，不验证语义或回答正确性。"""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('mentions',Path(__file__).parents[1]/'scripts/analyze_text_panel_mentions.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_option_order_symmetry():
    """选项排列只重排结果，不改变内容匹配。"""
    words=[dict(text='SHAVASANA',score=.9)]
    assert module.mentions(['A. Shavasana.','B. Bhramari.'],words)==[True,False]
    assert module.mentions(['A. Bhramari.','B. Shavasana.'],words)==[False,True]


def test_numbers_not_substrings():
    """数字1不能匹配10，避免凭子串扩大数值命中。"""
    assert module.mentions(['A. 1','B. 10'],[dict(text='10',score=.9)])==[False,True]


def test_negation_preserved_and_low_score_ignored():
    """否定词不删除，低分OCR不产生词面命中。"""
    assert module.mentions(['A. Not first'],[dict(text='first',score=.9)])==[False]
    assert module.mentions(['A. First'],[dict(text='first',score=.4)])==[False]
