"""单帧推广资格能力探针；只输出建议，不接入冻结选帧流程。"""
import re

INSTRUCTION='''Inspect this single frame as a possible evidence source for the video question below.
Do NOT answer the video question. Classify the role of the visible frame instead.

A: Clear promotional-only content unrelated to answering this question, such as an app advertisement, purchase page, purchase link, or channel subscription/recommendation page.
B: Substantive video content or potentially useful evidence/context; no clear unrelated promotion.
C: Mixed promotion/branding/credits and substantive content that could help answer the question. Keep the frame.
D: Uncertain from this frame alone. Keep the frame.

Do not infer promotion merely from its position, a logo, text, dark lighting, credits, or a product being visible.
Preserve in-story advertisements or objects when they are part of the events asked about.
If the main visual content could provide evidence or context, choose B or C rather than A.
When uncertain, choose D. Use only visible content and the question, not assumed narration.

Video question (context only, do not answer):
{question}

Which category describes this frame?'''
OPTIONS=['Clear unrelated promotional-only content.',
         'Substantive content or potentially useful evidence/context.',
         'Mixed content with potentially useful evidence; keep.',
         'Uncertain; keep.']

def classify_output(raw):
    """接口：仅完整A/B/C/D标签有效；格式异常保守保留，不重试。"""
    match=re.fullmatch(r'([ABCD])[.]?',raw.strip(),re.I)
    if not match:return dict(category='D',decision='keep',parse_status='invalid_fallback')
    letter=match.group(1).upper()
    return dict(category=letter,decision='exclude_candidate' if letter=='A' else 'keep',parse_status='valid')

def inspect_frame(backend,image,question,state):
    """接口：复用已验收模型，以一个画面和原问题做辅助分类，不输入原选项/答案/排名。"""
    # 步骤1：使用虚拟单帧片段时间，避免原视频末端位置成为分类捷径；真实PTS由调用记录保存。
    result=backend.answer([image],[0.0],1.0,INSTRUCTION.format(question=question),OPTIONS,generation_state=state)
    # 步骤2：仅A提出排除建议，其余保留；本能力实验不实际删除候选。
    parsed=classify_output(result['raw_output'])
    return dict(**parsed,inference=result,task='auxiliary_promotion_classification_not_original_QA')
