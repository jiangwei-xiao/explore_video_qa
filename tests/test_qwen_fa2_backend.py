"""FA2修订范围验收：新后端不能偷偷替换预处理、提示词或生成实现。"""
import pytest
from videoqa_full.qwen_backend import QwenVideoBackend
from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend


@pytest.mark.parametrize('name',['prepare','answer','warmup'])
def test_fa2_reuses_frozen_input_and_generation(name):
    """只允许加载注意力实现不同，其余方法直接复用原类。"""
    assert getattr(QwenFA2VideoBackend,name) is getattr(QwenVideoBackend,name)
