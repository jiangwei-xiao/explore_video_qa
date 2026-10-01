"""仅粗池的入口替换重放；不解码、不补查、不产生问答成绩。"""
import math
from videoqa_methods.density_selection import build_regions,frame_key
from videoqa_methods.coarse_eligibility import coarse_selection_cells
from videoqa_audit.density_reward_decay import select_density_decay
from videoqa_methods.prefix_coverage_budget import select_prefix_coverage


def plan_from_seeds(candidates, original_scores, seeds, video, cfg):
    """接口：显式种子集合→原区域结构；锚点仍按原问题分数决定，不融合跨文本概率。"""
    n=len(candidates)
    if len(original_scores)!=n or any(not math.isfinite(s) or not 0<=s<=1 for s in original_scores):
        raise ValueError('Invalid original-question scores')
    if len(seeds)!=min(cfg['seed_budget'],n) or len(seeds)!=len(set(seeds)) or any(not 0<=i<n for i in seeds):
        raise ValueError('Invalid explicit seed set')
    # 步骤1：仅将集合和原分数次序编码给冻结几何函数；这些数不是ITM，不写回资产。
    ordered=sorted(seeds,key=lambda i:(-original_scores[i],*frame_key(candidates[i])))
    priority=[0.]*n
    for rank,i in enumerate(ordered):priority[i]=(len(ordered)-rank)/(len(ordered)+1)
    plan=build_regions(candidates,priority,video,cfg)
    assert plan['seed_indices']==ordered
    # 步骤2：原最高分锚点、密度权重、核心连接与扩窗均保持原定义。
    for region in plan['regions']:
        assert region['anchor_index']==min(region['seed_indices'],key=lambda i:(-original_scores[i],*frame_key(candidates[i])))
    return plan


def replay_coarse(candidates,features,original_scores,seeds,video,cfg):
    """接口：同一完整粗池下执行原局部前缀和DP；补查标志只作轨迹记录，不触发读取。"""
    plan=plan_from_seeds(candidates,original_scores,seeds,video,cfg)
    # 步骤1：为原DP校验构造同池可行参考配额；它不决定DP最终最优解。
    _,reference=select_density_decay(candidates,features,coarse_selection_cells(plan,video),video,cfg)
    # 步骤2：初始分数始终是原问题ITM；不把四路名次当成支持置信度。
    selected,trace=select_prefix_coverage(candidates,features,plan,video,cfg,reference['region_budgets'],original_scores)
    return dict(plan=plan,selected_indices=selected,trace=trace,new_model_calls=0,refinement_performed=False,
                scope='Same coarse-only pool; cannot reuse any historical QA prediction')
