"""RD-1.2-exp04：全局粗候选资格与局部细采样窗口解耦，原预算/支持公式不变。"""
import copy
from fractions import Fraction
import math
from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_methods.density_selection import source_time
from videoqa_methods.weighted_seed_support import select_weighted_seed_support


def coarse_selection_cells(plan,video):
    """接口：将原检索区域按到区间的时间距离扩成唯一选择单元，不作为解码窗口。"""
    regions=plan['regions'];tick=Fraction(video['time_base']);end=Fraction(video['duration_fraction'])
    if not regions or tick<=0:raise ProtocolError('Invalid regions or source tick')
    # 步骤1：间隙平分即到左右原窗口距离相等的位置；同距源PTS归较早区域。
    cuts=[]
    for left,right in zip(regions,regions[1:]):
        b,a=Fraction(left['right_fraction']),Fraction(right['left_fraction'])
        if b>=a:raise ProtocolError('Original refinement windows must be ordered and disjoint')
        midpoint=(b+a)/2
        cuts.append(math.floor(midpoint/tick)*tick)
    # 步骤2：用相邻整数PTS刻度隔开闭区间，避免边界帧重复或浮点epsilon。
    # 所有源帧相对起点均为time_base的整数倍，因此这不会遗漏可表示的源帧。
    result=copy.deepcopy(plan)
    for i,r in enumerate(result['regions']):
        r['original_window']=[r['left_fraction'],r['right_fraction']]
        r['left_fraction']=str(Fraction(0) if i==0 else cuts[i-1]+tick)
        r['right_fraction']=str(end if i==len(regions)-1 else cuts[i])
        if Fraction(r['left_fraction'])>Fraction(r['right_fraction']):raise ProtocolError('Empty temporal cell')
    result['scope']='selection eligibility only; NEVER pass these cells to video refinement/decode'
    return result


def select_global_coarse(candidates,features,plan,video,cfg,quotas,initial_scores):
    """接口：现成粗+原细池共同竞争；不产生新帧，不改补查/种子/配额/局部评分。"""
    cells=coarse_selection_cells(plan,video);n=len(initial_scores)
    if not 1<=n<=len(candidates):raise ProtocolError('Invalid coarse prefix')
    membership={};original_membership={}
    # 步骤1：核对所有源帧恰属一个选择单元；原窗内帧仍属原区域。
    for i,f in enumerate(candidates):
        t=source_time(f,video)
        owners=[r['region_id'] for r in cells['regions'] if Fraction(r['left_fraction'])<=t<=Fraction(r['right_fraction'])]
        old=[r['region_id'] for r in plan['regions'] if Fraction(r['left_fraction'])<=t<=Fraction(r['right_fraction'])]
        if len(owners)!=1 or (old and old!=owners):raise ProtocolError('Eligibility partition changed original membership')
        if i>=n and not old:raise ProtocolError('Fine candidate outside original refinement windows')
        membership[i]=owners[0];original_membership[i]=old
    for r in cells['regions']:
        r['coarse_indices']=[i for i in range(n) if membership[i]==r['region_id']]
    # 步骤2：直接复用exp03局部选择；第一遍配额仍来自冻结RD-1.2，不在此重算。
    selected,trace=select_weighted_seed_support(candidates,features,cells,video,cfg,quotas,initial_scores)
    trace.update(eligibility_rule='all existing coarse candidates; nearest original window in time; no new refinement',
        selection_cells=[dict(region_id=r['region_id'],left_fraction=r['left_fraction'],right_fraction=r['right_fraction'],
            original_window=r['original_window']) for r in cells['regions']],
        newly_eligible_coarse=[i for i in range(n) if not original_membership[i]],
        selected_outside_original_windows=[i for i in selected if not original_membership[i]])
    return selected,trace


def select_global_coarse_two_pass(candidates,features,plan,video,cfg,original_quotas,initial_scores):
    """接口：统一两遍的候选资格，按原第一遍公式重算配额，再按exp03规则选帧。"""
    from videoqa_audit.density_reward_decay import select_density_decay
    # 步骤1：只把可用候选域传给原预算函数；种子、密度权重和公式均保持原值。
    cells=coarse_selection_cells(plan,video)
    _,allocation=select_density_decay(candidates,features,cells,video,cfg)
    # 步骤2：细帧仍只能来自原窗口。复用同一资格校验，不对扩大单元执行补查。
    selected,trace=select_global_coarse(candidates,features,plan,video,cfg,allocation['region_budgets'],initial_scores)
    trace.update(budget_rule='original RD-1.2 first pass recomputed on the same eligible pool as the second pass',
        original_quotas=original_quotas,budget_trace=allocation,
        quota_changes={rid:allocation['region_budgets'][rid]-original_quotas[rid] for rid in original_quotas})
    return selected,trace
