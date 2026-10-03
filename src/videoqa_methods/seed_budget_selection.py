"""在给定种子计划及合法池上重算exp12，不改变局部或预算公式。"""
from fractions import Fraction
from videoqa_methods.coarse_eligibility import coarse_selection_cells
from videoqa_methods.coarse_only_ablation import feasible_reference
from videoqa_methods.density_selection import source_time
from videoqa_methods.prefix_coverage_budget import select_prefix_coverage
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage


def select_seed_budget(pool,features,frame_budget=32):
    """接口：种子与窗口由输入池冻结；只按原exp12规则生成选择和追踪，不读取答案。"""
    # 步骤1：可行参考容量含粗细候选，仅用于旧函数验收，不作为最终预算。
    cells=coarse_selection_cells(pool['plan'],pool['video'])['regions']
    capacities=[sum(Fraction(r['left_fraction'])<=source_time(c,pool['video'])<=Fraction(r['right_fraction'])
                    for c in pool['candidates']) for r in cells]
    target=min(frame_budget,len(pool['candidates']))
    counts=feasible_reference(capacities,target)
    reference={str(r['region_id']):k for r,k in zip(cells,counts)}
    # 步骤2：复用原加权种子支持局部前缀，再以原问题加权粗参考覆盖执行精确DP。
    _,parent=select_prefix_coverage(pool['candidates'],features,pool['plan'],pool['video'],
        dict(frame_budget=frame_budget),reference,pool['initial_scores'])
    selected,trace=select_relevance_coverage(pool,features,parent)
    assert len(selected)==len(set(selected))==target
    assert set(parent['protected_indices'])<=set(selected)
    trace.update(protected_indices=parent['protected_indices'],prefix_trace=parent,
        seed_budget_effective=pool['plan']['effective_seed_count'],frame_budget=frame_budget)
    return selected,trace
