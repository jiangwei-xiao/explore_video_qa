"""exp12条件补查消融：仅移除细帧，复用冻结前缀和预算公式。"""
import numpy as np
from videoqa_methods.prefix_coverage_budget import select_prefix_coverage
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage


def feasible_reference(capacities,budget):
    """接口：构造旧函数所需的可行对照配额；仅用于目标下界验收，不决定最终配额。"""
    if not capacities or min(capacities)<1 or not len(capacities)<=budget<=sum(capacities):
        raise ValueError('Invalid reference capacity')
    counts=[1]*len(capacities);left=budget-len(counts)
    # 步骤1：按原区域顺序填充，最终DP仍独立枚举全部可行前缀组合。
    for j,cap in enumerate(capacities):
        take=min(left,cap-1);counts[j]+=take;left-=take
    assert left==0
    return counts


def select_coarse_only(pool,features,parent_trace,frame_budget=32):
    """接口：原粗池/特征前缀与原窗口不变，输出新选择、前缀及预算；不读答案。"""
    n=pool['initial_count'];f=np.asarray(features,dtype=np.float32)
    if not 1<=n<=len(pool['candidates']) or len(f)!=len(pool['candidates']):
        raise ValueError('Invalid frozen coarse prefix')
    # 步骤1：只创建浅层独立视图，不改写原池、原候选编号或原特征。
    coarse=dict(pool,candidates=pool['candidates'][:n]);target=min(frame_budget,n)
    capacities=[len(r['reference_coarse_indices']) for r in parent_trace['regions']]
    if sum(capacities)!=n:raise ValueError('Coarse ownership mismatch')
    counts=feasible_reference(capacities,target)
    reference={str(r['region_id']):k for r,k in zip(parent_trace['regions'],counts)}
    # 步骤2：复用原L乘sqrt差异局部次序。可行reference只验证旧函数的DP下界。
    _,coarse_trace=select_prefix_coverage(coarse['candidates'],f[:n],coarse['plan'],coarse['video'],
        dict(frame_budget=frame_budget),reference,coarse['initial_scores'])
    selected,trace=select_relevance_coverage(coarse,f[:n],coarse_trace)
    # 步骤3：所有粗参考、密度、锚点保持，重新分配名额而非从旧32帧中简单删除细帧。
    assert len(selected)==len(set(selected))==target and all(i<n for i in selected)
    assert set(parent_trace['protected_indices'])<=set(selected)
    for old,new in zip(parent_trace['regions'],trace['regions']):
        assert old['region_id']==new['region_id'] and old['reference_coarse_indices']==new['reference_coarse_indices']
        assert old['density_weight']==new['density_weight']
    trace.update(ablation='remove refinement candidates; same seeds/windows/references/formulas',
        initial_count=n,frame_budget=frame_budget,actual_budget=target,coarse_prefix_trace=coarse_trace,
        feasibility_reference_only=reference,protected_indices=parent_trace['protected_indices'])
    return selected,trace
