"""原题事件子句的定位能力对照：只取原文，不生成新事件或未知事实。"""
import re

WH_AUX=re.compile(r'(?:at\s+)?what\s+time\s+(?:do|does|did)\s+',re.I)
GENERIC_ACTOR=re.compile(r'(?:the|a)\s+(?:(?:male|female)\s+)?(?:protagonist|hero|heroine|man|woman|boy|girl|person)\s+in\s+the\s+video\s+',re.I)


def event_clause(question):
    """接口：原问题→精确事件子串、来源跨度及移出的通用人物角色；不接收选项或答案。

    这是定位能力诊断，暂时移出通用主体会损失身份约束，不能直接替换正式
    问题或部署到多人视频。限定词/专名仍在原文子串内；不支持的句式回退原题。
    """
    if not isinstance(question,str) or not question.strip(): raise ValueError('Original question required')
    # 步骤1：只识别明确时刻句式，保留所有非模板限定，不做自由改写。
    start=0; end=len(question.rstrip()); match=WH_AUX.search(question); subject=''
    if match:
        start=match.end(); actor=GENERIC_ACTOR.match(question[start:])
        if actor: subject=actor.group(); start+=actor.end()
        if question[end-1]=='?': end-=1
        while end>start and question[end-1].isspace(): end-=1
    query=question[start:end]
    if not query.strip(): raise ValueError('Empty locating event clause')
    # 步骤2：子串校验是信息来源约束，不保证语义任务或人物身份完整。
    return dict(original_question=question,event_query=query,source_span=[start,end],removed_subject=subject,
                supported=match is not None,actor_information_ablation=bool(subject),
                scope='Exact-source event-only retrieval diagnostic; not final QA text or deployed method')
