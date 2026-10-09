"""事件查询的来源条件验收；保留诊断表达，不修改已冻结的原文提取器。"""
import re
from videoqa_audit.event_clause import event_clause, WH_AUX

NEUTRAL_PREFIX = re.compile(r'\s*(?:(?:in|according to)\s+the\s+video\s*,?\s*)?', re.I)


def checked_event_clause(question):
    """接口：仅原问题输入；提取会丢前置内容限定时完整回退，并记录具体原因。

    通用人物暂移出的身份风险仍保留为诊断限制，不能称为语义等价改写。
    不接收题号、选项、答案或人工见证，回退不取决于模型分数或历史对错。
    """
    result = event_clause(question)
    # 步骤1：只允许移出无内容定位的任务引导词；时间/顺序/空间限定不能消失。
    match = WH_AUX.search(question)
    prefix = question[:match.start()] if match else ''
    lost = bool(match and NEUTRAL_PREFIX.fullmatch(prefix) is None)
    if lost:
        original = result['event_query']
        result.update(event_query=question, source_span=[0, len(question)],
            supported=False, removed_subject='', actor_information_ablation=False,
            rejected_event_query=original, guard_reason='nonneutral_leading_condition',
            guard_status='fallback', preserved_leading_text=prefix)
    else:
        result.update(guard_status='accepted' if result['supported'] else 'fallback',
            guard_reason=None if result['supported'] else 'unsupported_grammar')
    # 步骤2：来源约束只保护已定义的条件，不能证明多人身份或答案证据充分。
    start, end = result['source_span']
    if question[start:end] != result['event_query']:
        raise ValueError('Query is not an exact original source span')
    return result
