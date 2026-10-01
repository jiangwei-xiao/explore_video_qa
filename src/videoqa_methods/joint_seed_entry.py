"""RD-1.2-exp09：原题独立视角与四条候选描述共同提议种子。"""
from videoqa_audit.joint_query_views import select_joint_views
from videoqa_audit.option_coarse_replay import plan_from_seeds


def joint_seed_plan(coarse,original_scores,option_scores,video,cfg):
    """接口：仅增加原题独立排名，原分数锚点、支持、区域及预算规则不变。"""
    # 步骤1：五路竞争同一M预算，不保留五组各M帧，也不根据正确选项选分支。
    seeds,trace=select_joint_views(coarse,original_scores,option_scores,cfg['seed_budget'])
    # 步骤2：排序编码仅限定集合，继续由原题分数决定区域锚点和后续支持。
    return plan_from_seeds(coarse,original_scores,seeds,video,cfg),trace
