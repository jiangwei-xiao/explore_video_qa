"""四路候选陈述的入口对照；不接收标准答案，不执行RD后续机制。"""
import numpy as np
from videoqa_audit.query_input_probe import ranking


def select_option_ranks(candidates, scores, budget=32):
    """接口：四个查询内名次取最小值，稳定排序后返回时间有序候选及完整追踪。"""
    if len(scores) != 4 or not isinstance(budget, int) or budget < 1:
        raise ValueError('Expected four routes and positive budget')
    # 步骤1：仅比较查询内名次，避免把跨文本ITM概率当成校准的事实置信度。
    orders = [ranking(candidates, vector) for vector in scores]
    ranks = np.empty((4, len(candidates)), dtype=np.int64)
    for branch, order in enumerate(orders):
        ranks[branch, order] = np.arange(1, len(candidates) + 1)
    best = ranks.min(axis=0)
    # 步骤2：名次相同按源PTS、编号，不使用正确选项、视觉差异或时间扩窗。
    order = sorted(range(len(candidates)), key=lambda i: (
        int(best[i]), candidates[i]['source_pts'], candidates[i]['candidate_index']))
    selected = sorted(order[:min(budget, len(candidates))], key=lambda i: (
        candidates[i]['source_pts'], candidates[i]['candidate_index']))
    return selected, dict(per_route_orders=orders, best_rank=best.tolist(), fused_order=order,
                          budget=budget, actual_frames=len(selected))
