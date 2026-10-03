"""固定局部前缀的等权区域覆盖对照，只移除预算目标中的检索密度倍率。"""
import math
from videoqa_methods.density_selection import frame_key
from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget


def select_equal_region_budget(candidates,reference_trace):
    """接口：输入已冻结前缀/曲线，输出相同总预算的新配额；不改变上游补查密度。

    目标从Σw_r C_r(n_r)改为ΣC_r(n_r)，不是均分帧数，也不宣称各区真的同样重要。
    等权只是检验检索密度先验是否压制其他已定位内容的无新增参数对照。
    """
    regions=reference_trace['regions']
    original_weights=[r['density_weight'] for r in regions]
    if not regions or any(not math.isfinite(w) or w<=0 for w in original_weights):raise ValueError('Invalid regions')
    target=sum(reference_trace['region_budgets'].values())
    # 步骤1：原前缀、保护锚点和曲线完全冻结，只改DP的区域乘子。
    quotas,value=allocate_prefix_budget([r['coverage_curve'] for r in regions],[1.]*len(regions),target)
    old_quotas=[reference_trace['region_budgets'][str(r['region_id'])] for r in regions]
    reference=sum(r['coverage_curve'][n] for r,n in zip(regions,old_quotas))
    equivalent=all(w==original_weights[0] for w in original_weights)
    # 步骤2：共同正倍率在数学上不影响解，沿用旧离散解避免浮点近同分造成伪变化。
    if equivalent:
        if abs(value-reference)>1e-12:raise ValueError('Common-weight equivalence failed')
        quotas=old_quotas;value=reference
    if value+1e-12<reference:raise ValueError('Worse than feasible old allocation')
    selected=sorted([i for r,n in zip(regions,quotas) for i in r['prefix'][:n]],key=lambda i:frame_key(candidates[i]))
    if len(selected)!=target or len(set(selected))!=target or not set(reference_trace['protected_indices'])<=set(selected):
        raise ValueError('Budget or anchor changed')
    # 步骤3：保留原密度字段作追踪，不覆盖原计划，明确区分旧先验与新预算权重。
    return selected,dict(regions=[dict(r,budget_weight=1.) for r in regions],
        region_budgets={str(r['region_id']):n for r,n in zip(regions,quotas)},
        protected_indices=reference_trace['protected_indices'],eligible_indices=reference_trace['eligible_indices'],
        objective=value,reference_objective=reference,old_density_weights=original_weights,
        exact_equal_weight_equivalence=equivalent,local_prefixes_unchanged=True,
        rule='Equal regional empirical coverage utility; original density still used upstream')
