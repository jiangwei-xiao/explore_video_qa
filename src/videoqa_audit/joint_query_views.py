"""原题独立视角与四个候选描述共存的固定种子诊断。"""
import numpy as np
from videoqa_audit.query_input_probe import ranking


def select_joint_views(candidates,question_scores,option_scores,budget=32):
    """接口：五路取最佳查询内名次，不增加总预算、不比较跨文本绝对概率。"""
    if len(option_scores)!=4 or not isinstance(budget,int) or budget<1:raise ValueError('Expected four option routes and positive budget')
    orders=[ranking(candidates,s) for s in [question_scores,*option_scores]]
    ranks=np.empty((5,len(candidates)),dtype=np.int64)
    # 步骤1：原题成为独立的一路，文字被包含在其他查询中不等于其原排名得到保留。
    for k,order in enumerate(orders):ranks[k,order]=np.arange(1,len(candidates)+1)
    best=ranks.min(axis=0)
    ordered=sorted(range(len(candidates)),key=lambda i:(int(best[i]),candidates[i]['source_pts'],candidates[i]['candidate_index']))
    # 步骤2：稳定取最多M个唯一源帧；五路争用同一预算，不保证四路旧种子全被保留。
    selected=sorted(ordered[:min(budget,len(candidates))],key=lambda i:(candidates[i]['source_pts'],candidates[i]['candidate_index']))
    return selected,dict(view_names=['question_only','question_plus_A','question_plus_B','question_plus_C','question_plus_D'],
                          per_view_orders=orders,best_rank=best.tolist(),fused_order=ordered,budget=budget)
