"""CPU索引提前发布的并发验收，不解码真实视频或运行模型。"""
import importlib.util
import json
from pathlib import Path

import pytest


def load():
    """测试接口：只导入独立索引脚本，主协调器和冻结实现不执行。"""
    path=Path(__file__).resolve().parents[1]/'scripts/preindex_longvideo_trial_assets.py'
    spec=importlib.util.spec_from_file_location('preindex_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_publish_complete_record_without_overwriting(tmp_path):
    m=load();result=dict(video_id='one',sha256='fixed',bytes=20,eligible=True,reason=None,fingerprint={'phash':['01']})
    stage=tmp_path/'staged.json';stage.write_text(json.dumps(result))
    assert m.publish_result(tmp_path,stage,result)
    canonical=tmp_path/'identity_checks/one.json';before=canonical.stat().st_ino
    assert not m.publish_result(tmp_path,stage,dict(result,index_path='/different/cache'))
    assert canonical.stat().st_ino==before
    with pytest.raises(ValueError):m.publish_result(tmp_path,stage,dict(result,sha256='mutated'))


def test_unfinished_record_never_published(tmp_path):
    with pytest.raises(FileNotFoundError):load().publish_result(tmp_path,tmp_path/'missing.json',dict(video_id='one'))
    assert not (tmp_path/'identity_checks/one.json').exists()
