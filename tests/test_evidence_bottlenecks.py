"""结构代价测试：不把命中、可达和语义有效混为一谈。"""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('bottlenecks', Path(__file__).parents[1] / 'scripts/summarize_evidence_bottlenecks.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fact(position, quota=3, cap=8, selected=False):
    """构造独立的单事实结构，不需要视频、GPU或标准答案。"""
    return dict(question_id='test', fact_id='f', description='测试', current_hit=selected,
                paths=[dict(exp12=dict(prefix_position=position, quota=quota,
                                      max_feasible_quota=cap, selected=selected))])


def test_selected():
    assert module.summarize_fact(fact(2, selected=True))['minimum_extra_slots'] == 0


def test_quota_cost():
    assert module.summarize_fact(fact(7))['minimum_extra_slots'] == 4


@pytest.mark.parametrize('position', [None, 9])
def test_order_required(position):
    assert module.summarize_fact(fact(position))['state'] == 'order_or_representation_required'


def test_alternative_witness():
    row = fact(None)
    row['paths'] += fact(4)['paths']
    assert module.summarize_fact(row)['minimum_extra_slots'] == 1


def test_inconsistent_summary_rejected():
    row = fact(2, selected=True)
    row['current_hit'] = False
    with pytest.raises(ValueError):
        module.summarize_fact(row)
