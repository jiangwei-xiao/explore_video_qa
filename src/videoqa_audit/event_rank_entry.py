"""原问题与事件查询的联合排序入口；固定启发式，不把排名当作证据概率。"""
import math
from videoqa_methods.density_selection import frame_key


def joint_rank_scores(candidates, question_scores, event_scores):
    """接口：同一粗池两套分数→联合入口分数及逐帧名次；不读取答案、题型或人工事实。

    最小化两名次乘积等价于最小化log名次之和；两查询按同样的相对名次
    代价参与，无混合系数。它只检验排序共识，不保证互补证据或人物条件。
    """
    n = len(candidates)
    if not n or len(question_scores) != n or len(event_scores) != n:
        raise ValueError('Two rankings must cover the same nonempty coarse population')
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in list(question_scores) + list(event_scores)):
        raise ValueError('Invalid original/event ITM scores')
    # 步骤1：概率只用于各查询自己的稳定排序，不跨查询比较绝对大小。
    ranks = []
    for scores in (question_scores, event_scores):
        ordered = sorted(range(n), key=lambda i: (-scores[i], *frame_key(candidates[i])))
        rank = [0] * n
        for position, i in enumerate(ordered, 1): rank[i] = position
        ranks.append(rank)
    # 步骤2：正名次乘积定义单一联合代价；倒数仅适配旧入口0—1接口。
    products = [a * b for a, b in zip(*ranks)]
    return [1.0 / value for value in products], dict(
        question_ranks=ranks[0], event_ranks=ranks[1], rank_products=products,
        rule='minimize question_rank * event_rank; no probability interpretation or fitted coefficient')
