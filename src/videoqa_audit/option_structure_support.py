"""选项结构与支持权重的独立诊断；不读取答案选择规则，不组合两项干预。"""
import re
import numpy as np
from videoqa_methods.coarse_eligibility import select_global_coarse


def option_structure(question,options):
    """接口：只识别严格数字计数及显式事件引用排列；其他文本不推断语义类型。"""
    if len(options)!=4 or not isinstance(question,str):raise ValueError('Expected question and four options')
    bodies=[re.sub(r'^[A-D][.)]\s*','',s.strip()).strip() for s in options]
    # 步骤1：只处理How many与纯整数选项；不扩展到年份、单位、英文数词等未知情况。
    if re.search(r'\bhow many\b',question,re.I) and all(re.fullmatch(r'\d+\.?',s) for s in bodies):
        return 'integer_count_options'
    # 步骤2：引用必须能在题面找到独立事件定义，且各选项都是相同标签集合的排列。
    labels=re.findall(r'(?:^|\n)\s*\(([a-z])\)\s+\S',question.lower())
    if len(labels)>=2 and len(labels)==len(set(labels)):
        valid=True
        for body in bodies:
            refs=re.findall(r'\(([a-z])\)',body.lower())
            if not re.fullmatch(r'(?:\([a-z]\)\s*)+\.?',body.lower()) or len(refs)!=len(labels) or set(refs)!=set(labels):valid=False
        if valid:return 'event_reference_permutations'
    return 'other'


def fixed_quota_support(pool,features,quotas,cfg,equal=False):
    """接口：固定exp08候选、种子、锚点及配额，仅取消相对种子分数权重。"""
    # 步骤1：派生等权向量只传给支持函数，绝不覆盖池中保存的真实ITM分数。
    scores=np.ones(len(pool['initial_scores']),dtype=np.float64).tolist() if equal else pool['initial_scores']
    selected,trace=select_global_coarse(pool['candidates'],features,pool['plan'],pool['video'],cfg,quotas,scores)
    # 步骤2：保存必要的重放轨迹，不将旧函数的历史配额标签误当本轮来源。
    compact=dict(region_budgets=trace['region_budgets'],protected_indices=trace['protected_indices'],
        supports=trace['supports'],rounds=[dict(region_id=r['region_id'],selected_before=r['selected_before'],winner=r['winner']) for r in trace['rounds']],
        quota_source='frozen exp08 DP allocation',support_weighting='equal_selected_seeds' if equal else 'original_question_relative',new_model_calls=0)
    return selected,compact
