"""固定exp08配额的窗内竞争诊断，不重算预算或改写正式算法。"""
from fractions import Fraction
from videoqa_methods.density_selection import source_time
from videoqa_methods.weighted_seed_support import select_weighted_seed_support


def select_inside_windows(pool,features,quotas,cfg):
    """接口：只收缩候选资格，保留种子、锚点、分数、配额及局部排序公式。"""
    # 步骤1：容量不足必须显式失败，不能扩窗、借额或少选来掩盖不可比条件。
    for r in pool['plan']['regions']:
        count=sum(Fraction(r['left_fraction'])<=source_time(c,pool['video'])<=Fraction(r['right_fraction']) for c in pool['candidates'])
        if count<quotas[str(r['region_id'])]:raise ValueError('Insufficient fixed-window capacity')
    selected,trace=select_weighted_seed_support(pool['candidates'],features,pool['plan'],pool['video'],cfg,quotas,pool['initial_scores'])
    assert trace['region_budgets']==quotas
    # 步骤2：保留可重放的精简轨迹，标明配额仍来自未改动的exp08，不宣称联合最优。
    compact=dict(region_budgets=trace['region_budgets'],protected_indices=trace['protected_indices'],
        eligible_indices=trace['eligible_indices'],region_selected=trace['region_selected'],
        supports=trace['supports'],rounds=[dict(region_id=r['region_id'],selected_before=r['selected_before'],winner=r['winner']) for r in trace['rounds']],
        quota_source='frozen exp08 global-domain DP; diagnostic only',eligibility='original retrieval windows only',new_model_calls=0)
    return selected,compact
