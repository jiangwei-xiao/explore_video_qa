"""RD-1.2-exp08：只替换入口种子集合，其余复用冻结exp07定义。"""
from fractions import Fraction
from videoqa_audit.option_rank_selection import select_option_ranks
from videoqa_audit.option_coarse_replay import plan_from_seeds
from videoqa_methods.coarse_eligibility import coarse_selection_cells
from videoqa_methods.density_selection import source_time
from videoqa_methods.prefix_coverage_budget import select_prefix_coverage
from videoqa_budget32.prefix_full import reference_allocation


def option_seed_plan(coarse,original_scores,option_scores,video,cfg):
    """接口：四路仅决定种子集合；原问题分数继续决定锚点和后续支持。"""
    # 步骤1：复用已验收的对称名次融合，不选择正确选项或增加种子名额。
    seeds,ranking_trace=select_option_ranks(coarse,option_scores,cfg['seed_budget'])
    # 步骤2：复用原几何与锚点规则，不把名次编码写成新的ITM概率。
    return plan_from_seeds(coarse,original_scores,seeds,video,cfg),ranking_trace


def select_option_pool(pool,features,cfg):
    """接口：新窗口粗细池→原exp07局部前缀与DP选择，无新增评分公式。"""
    candidates=pool['candidates'];cells=coarse_selection_cells(pool['plan'],pool['video'])
    # 步骤1：只构造可行参考配额供原冻结函数检验下界，不参与最终DP解。
    counts=[sum(Fraction(r['left_fraction'])<=source_time(c,pool['video'])<=Fraction(r['right_fraction'])
                for c in candidates) for r in cells['regions']]
    target=min(cfg['frame_budget'],len(candidates))
    reference={str(r['region_id']):n for r,n in zip(cells['regions'],reference_allocation(counts,target))}
    # 步骤2：种子支持仍使用原问题ITM相对值，覆盖参照仍只用初始粗帧。
    return select_prefix_coverage(candidates,features,pool['plan'],pool['video'],cfg,reference,pool['initial_scores'])
