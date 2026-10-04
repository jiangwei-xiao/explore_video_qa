"""OCR账本恢复测试，使用伪返回，不调用GPU或生成模型。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('frame_text_probe', Path(__file__).parents[1] / 'scripts/probe_frame_text.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_completed_skips_processing(tmp_path):
    """完成项直接复用，缺少模型也不触发任何输入处理。"""
    path = tmp_path / 'call.json'
    record = dict(status='completed', fingerprint='fixed', raw_text='text')
    module.durable(path, record)
    assert module.recognize(None, None, None, path, 'fixed') == record


def test_uncertain_generation_not_repeated(tmp_path):
    """调用开始但未确认返回时必须停止，不能追求一次成功输出。"""
    path = tmp_path / 'call.json'
    module.durable(path, dict(status='started', fingerprint='fixed'))
    with pytest.raises(ValueError):
        module.recognize(None, None, None, path, 'fixed')


def test_returned_recovers_only_decode(tmp_path):
    """已有Token时仅完成后处理，伪模型不提供generate接口。"""
    path = tmp_path / 'call.json'
    module.durable(path, dict(status='returned', fingerprint='fixed', input_ids=[10], output_ids=[10, 11, 2]))
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2))
    processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: str(ids)))
    result = module.recognize(model, processor, None, path, 'fixed')
    assert result['generated_ids'] == [11, 2]
    assert result['normal_eos']


def test_changed_fingerprint_rejected(tmp_path):
    """来源变化时拒绝复用旧完成项。"""
    path = tmp_path / 'call.json'
    module.durable(path, dict(status='completed', fingerprint='old'))
    with pytest.raises(ValueError):
        module.recognize(None, None, None, path, 'new')
