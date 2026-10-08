"""exp07单预算入口：K同时确定定位种子上限与最终输入上限。"""
from fractions import Fraction

from videoqa_runtime.common import ROOT, read_json
from videoqa_methods.density_selection import build_regions, source_time
from videoqa_methods.coarse_eligibility import coarse_selection_cells
from videoqa_methods.coarse_only_ablation import feasible_reference
from videoqa_methods.prefix_coverage_budget import select_prefix_coverage


def budget_config(frame_budget):
    """接口：仅接收K，返回冻结函数所需的兼容配置；不修改历史配置文件。"""
    if isinstance(frame_budget, bool) or not isinstance(frame_budget, int) or frame_budget < 1:
        raise ValueError('frame_budget must be a positive integer')
    # 步骤1：复制原固定规则，内部两个兼容字段共同由唯一公开预算派生。
    cfg = read_json(ROOT / 'configs/retrieval_density_v1.json')
    cfg.update(seed_budget=frame_budget, frame_budget=frame_budget)
    return cfg


def select_exp07(pool, features, frame_budget):
    """接口：合法冻结粗细池与特征→源索引和轨迹；只有K，不接收答案或独立M。

    本接口不采样或计算特征。输入池必须已经按同一个K定位和补查，
    因而不允许把旧M32池直接用于K8，也不把细帧来源当作保留优先级。
    """
    # 步骤1：从原粗池重建计划，确认输入确实遵循Top-K种子规则。
    cfg = budget_config(frame_budget)
    coarse = pool['candidates'][:pool['initial_count']]
    plan = build_regions(coarse, pool['initial_scores'], pool['video'], cfg)
    if plan != pool['plan']:
        raise ValueError('Frozen pool does not follow the single-budget seed plan')
    # 步骤2：可行参照只满足旧函数的目标下界检查，不决定实际DP配额。
    cells = coarse_selection_cells(plan, pool['video'])['regions']
    capacities = [sum(Fraction(r['left_fraction']) <= source_time(c, pool['video'])
                      <= Fraction(r['right_fraction']) for c in pool['candidates']) for r in cells]
    target = min(frame_budget, len(pool['candidates']))
    counts = feasible_reference(capacities, target)
    reference = {str(r['region_id']): n for r, n in zip(cells, counts)}
    # 步骤3：原exp07前缀、等权覆盖与DP完全复用，仅改变外部配置契约。
    selected, trace = select_prefix_coverage(pool['candidates'], features, plan, pool['video'],
                                            cfg, reference, pool['initial_scores'])
    trace['budget_contract'] = dict(frame_budget=frame_budget, seed_count=len(plan['seed_indices']),
                                    actual_frame_count=len(selected))
    return selected, trace
