"""全量适配边界测试：不调用模型，冻结算法独立于用于验收的参照配额。"""
import pytest
from videoqa_budget32.prefix_full import partition, reference_allocation


def test_partition_order():
    """接口：导入集合和新调用集合按全量顺序闭合。"""
    data = [{'question_id':q} for q in ['a','b','c']]
    assert partition(data,['c','a']) == (['a','c'],['b'])


@pytest.mark.parametrize('data,dev', [([{'question_id':'a'}],['b']),([{'question_id':'a'},{'question_id':'a'}],[]),([{'question_id':'a'}],['a','a'])])
def test_invalid_partition(data,dev):
    """接口：未知、重复或交叉来源不能混入生成集合。"""
    with pytest.raises(ValueError): partition(data,dev)


@pytest.mark.parametrize('counts,budget,expected', [([2,3],4,[2,2]),([1,1],2,[1,1]),([2,3],5,[2,3]),([40],32,[32])])
def test_feasible_reference(counts,budget,expected):
    """接口：参照只满足容量约束，无模型和历史配额依赖。"""
    assert reference_allocation(counts,budget) == expected


@pytest.mark.parametrize('counts,budget', [([],0),([0,2],2),([1,1],3),([2,2],1)])
def test_invalid_reference(counts,budget):
    """接口：无可行参照时失败，不能伪造候选。"""
    with pytest.raises(ValueError): reference_allocation(counts,budget)


def test_reference_not_objective():
    """步骤1固定覆盖曲线，步骤2确认DP只依赖曲线/权重/预算，不接受历史答案。"""
    from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget
    curves = [[0,.2,.9],[0,.4,.5]]
    assert allocate_prefix_budget(curves,[1,1],3)[0] == [2,1]


def test_arbitrary_reference_does_not_change_selection():
    """接口：不同可行参照配额不得改变冻结exp07的实际输出。"""
    from test_prefix_coverage_budget import fixture
    from videoqa_methods.prefix_coverage_budget import select_prefix_coverage
    args = fixture()
    left = select_prefix_coverage(*(args[:5]+({'0':2,'1':6},args[-1])))
    right = select_prefix_coverage(*(args[:5]+({'0':6,'1':2},args[-1])))
    assert left[0] == right[0]
    assert left[1]['region_budgets'] == right[1]['region_budgets']
    assert left[1]['objective'] == right[1]['objective']
