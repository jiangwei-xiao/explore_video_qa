"""诊断标签、人工偏好隔离和不完整返回恢复边界。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from probe_evidence_binding import parse_preference, messages, finish_raw


def test_score_decomposition_has_only_constructed_swap_symmetry():
    """中文说明：只加载纯代数函数，避免导入GPU入口时修改测试进程的可见设备。"""
    import ast
    source=Path(__file__).resolve().parents[1]/'scripts/probe_evidence_binding_scores.py'
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='decompose')
    namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    f=namespace['decompose']
    assert f(-2.,-4.)==dict(antisymmetric_content=1.,symmetric_position=-3.)
    assert f(-4.,-2.)==dict(antisymmetric_content=-1.,symmetric_position=-3.)


def test_exact_labels_only():
    assert parse_preference(' first\n')=='FIRST'
    assert parse_preference('UNCERTAIN')=='UNCERTAIN'
    for text in ['A', 'FIRST.', 'FIRST because it shows a box', '']:
        assert parse_preference(text) is None


def test_prompt_has_only_presented_groups_not_annotation():
    groups=[[dict(timestamp_seconds=1.),dict(timestamp_seconds=2.)],
            [dict(timestamp_seconds=10.),dict(timestamp_seconds=20.)]]
    result=messages('Which object is shown?', ['A. One','B. Two','C. Three','D. Four'], groups)
    content=result[1]['content']
    assert sum(x['type']=='image' for x in content)==4
    text='\n'.join(x.get('text','') for x in content)
    assert 'Group FIRST' in text and 'Group SECOND' in text
    assert 'Group evidence' not in text and 'Group contrast' not in text
    assert '20.000 seconds' in text


def test_recovery_requires_complete_valid_observation():
    class Tokenizer:
        """中文说明：只测试解析，不调用任何实际模型。"""
        def decode(self, *args, **kwargs):
            """中文说明：模拟已完成的标签返回。"""
            return 'FIRST'
    raw=dict(generation_state=dict(started=True,returned=True),generated_token_ids=[1],
             measurements=dict(prefill_calls=1,checked_logit_steps=1),image_grid_thw=[[1,2,2]]*4)
    assert finish_raw(raw,Tokenizer())['parsed_preference']=='FIRST'
    with pytest.raises(AssertionError):
        finish_raw(dict(raw,generation_state=dict(started=True,returned=False)),Tokenizer())
    with pytest.raises(AssertionError):
        finish_raw(dict(raw,measurements=dict(prefill_calls=2,checked_logit_steps=1)),Tokenizer())
