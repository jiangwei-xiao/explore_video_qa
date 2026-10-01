"""exp12的跨区域检索强度对照：仅缩放预算效用，保护锚点和局部前缀不变。"""
import numpy as np
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget
from videoqa_methods.density_selection import frame_key


def select_seed_strength_budget(pool,features,reference_trace):
    """接口：区域平均原种子ITM作相对强度，缩放exp12覆盖曲线后执行原DP。"""
    parent_selected,parent=select_relevance_coverage(pool,features,reference_trace)
    scores=pool['initial_scores'];regions=pool['plan']['regions']
    means={r['region_id']:float(np.mean([scores[i] for i in r['seed_indices']],dtype=np.float64)) for r in regions}
    peak=max(means.values());seed_scores=[scores[i] for r in regions for i in r['seed_indices']]
    all_equal=all(x==seed_scores[0] for x in seed_scores);records=[];parent_value=0.
    # 步骤1：只比较同一问题内的种子强度，既不跨问题比较概率，也不声称语义可信度。
    for r in parent['regions']:
        factor=1. if peak==0 or all_equal else means[r['region_id']]/peak
        curve=[float(x*factor) for x in r['coverage_curve']]
        row=dict(r,coverage_curve=curve,query_strength_factor=factor,mean_seed_score=means[r['region_id']],
                 effective_density_weight=r['density_weight']*factor)
        records.append(row);parent_value+=r['density_weight']*curve[parent['region_budgets'][str(r['region_id'])]]
    # 步骤2：将缩放放在曲线中以复用原DP接口；等价于密度乘强度，不取消每区最低一帧。
    target=len(parent_selected)
    quotas,value=allocate_prefix_budget([r['coverage_curve'] for r in records],[r['density_weight'] for r in records],target)
    selected=sorted([i for r,k in zip(records,quotas) for i in r['prefix'][:k]],key=lambda i:frame_key(pool['candidates'][i]))
    if len(selected)!=len(set(selected)) or len(selected)!=target:raise ValueError('Budget mismatch')
    if value+1e-12<parent_value:raise ValueError('DP worse than feasible parent allocation')
    # 步骤3：同强度时精确退化为exp12；记录旧新预算供单因素归因，不附用旧答案。
    if all_equal and selected!=parent_selected:raise ValueError('Uniform-strength equivalence lost')
    return selected,dict(regions=records,region_budgets={str(r['region_id']):k for r,k in zip(records,quotas)},
        parent_selected_indices=parent_selected,parent_region_budgets=parent['region_budgets'],
        objective=value,parent_allocation_objective=parent_value,all_seed_scores_equal=all_equal,
        rule='exp12 reference weighting times mean regional seed ITM / maximum regional mean; same prefixes and protected anchors',
        limitation='Relative localization strength, not calibrated evidence confidence; weakly scored answer regions can be undervalued')
