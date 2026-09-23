"""阶段分析分母与配对单元测试；不加载任何模型。"""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('full_stage_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_full_stage_failures.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('u,d,cell',[(True,True,'both_correct'),(True,False,'rd_loss'),(False,True,'rd_gain'),(False,False,'both_wrong')])
def test_pair_cell(u,d,cell):
    assert module.pair_cell(u,d)==cell


def test_counts_do_not_treat_correctness_as_evidence():
    values=dict(region_count=2,window_fraction=.2,new_count=3,new_selected=1,seed_retained=8,uniform_outside_windows=12,max_region_quota=9)
    items=[dict(values,uniform=True,topk=False,rd=False,cell='rd_loss'),dict(values,uniform=False,topk=True,rd=True,cell='rd_gain')]
    result=module.summarize_group(items)
    assert result['n']==2 and result['counts']==dict(uniform=1,topk=1,rd=1)
    assert result['cells']==dict(rd_loss=1,rd_gain=1)
    assert result['new_selected']['mean']==1 and result['refined_questions']==2
