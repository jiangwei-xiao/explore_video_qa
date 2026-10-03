"""历史视频占位重建只允许折叠已知连续Token，不改变题目正文。"""
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('native_confidence',Path(__file__).resolve().parents[1]/'scripts/probe_native_answer_confidence.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_expanded_and_single_placeholder():
    assert module.restore_media_placeholder(dict(prompt='x<|video_pad|><|video_pad|>y',visual_tokens=2))=='x<|video_pad|>y'
    assert module.restore_media_placeholder(dict(prompt='x<|video_pad|>y',visual_tokens=100))=='x<|video_pad|>y'


def test_noncontiguous_or_wrong_count_rejected():
    with pytest.raises(ValueError):module.restore_media_placeholder(dict(prompt='<|video_pad|>x<|video_pad|>',visual_tokens=2))
    with pytest.raises(ValueError):module.restore_media_placeholder(dict(prompt='<|video_pad|><|video_pad|>',visual_tokens=3))
