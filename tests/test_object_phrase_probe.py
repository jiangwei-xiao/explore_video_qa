"""严格文本模板的范围与限定语保留；不涉及答案或视频题型标签。"""
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).parents[1]/'scripts/probe_object_phrase_localization.py'
spec=importlib.util.spec_from_file_location('object_phrase_probe',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('question,expected',[
    ('What color is the makeup bag in the video?','the makeup bag in the video'),
    ('What color are the wheels of the skateboard used by the dog in the video?',
     'the wheels of the skateboard used by the dog in the video'),
    ("What color is the player's sniper rifle?","the player's sniper rifle"),
    ('What colour is the first car after the truck?','the first car after the truck'),
    ('Which color is the bag?',None),('What is the bag?',None),
])
def test_keep_object_constraints(question,expected):
    """只移除已声明的提问前缀和问号，不删时间/对象限定语，不扩大规则。"""
    assert module.object_phrase(question)==expected
