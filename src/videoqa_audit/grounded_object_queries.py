"""物体定位候选的文本忠实性与可见结构检查；不预测答案。"""
import json
import re
import numpy as np

TRANSPORT_ALIASES={'biking':'bicycle','cycling':'bicycle','skateboarding':'skateboard','driving':'vehicle','walking':'person'}
FUNCTION_WORDS=set('a an the of with in on to from his her their its this that and or person people man woman girl boy'.split())


def box_structure(image,box):
    """接口：仅判返回框是否有可见像素结构，原8bit变化≤2时弃权，不判语义正确。

    这不是暗场过滤；框外画面及原置信度保持不变，误报有纹理物体仍需独立核查。
    """
    pixels=np.asarray(image.convert('RGB'),dtype=np.uint8);h,w=pixels.shape[:2]
    values=np.asarray(box,dtype=np.float64)
    if values.shape!=(4,) or not np.isfinite(values).all():raise ValueError('Invalid detector box')
    # 步骤1：仅裁到源画布进行可见性检查，不改模型输入或框置信度。
    x0,y0,x1,y1=values;left=max(0,int(np.floor(x0)));top=max(0,int(np.floor(y0)))
    right=min(w,int(np.ceil(x1)));bottom=min(h,int(np.ceil(y1)))
    if right<=left or bottom<=top:return dict(grounded_visual_structure=False,reason='outside_visible_canvas')
    region=pixels[top:bottom,left:right].reshape(-1,3)
    ranges=np.ptp(region,axis=0).astype(int).tolist()
    # 步骤2：均匀ROI无法在单帧中支持物体边界；非均匀并不保证对象或关系正确。
    return dict(grounded_visual_structure=max(ranges)>2,channel_ranges=ranges,
        clipped_box=[left,top,right,bottom],reason='nonuniform_pixels_not_semantic_truth' if max(ranges)>2 else 'uniform_roi_abstain')


def parse_object_queries(raw,question,options):
    """接口：严格JSON known/choices，物体词必须来自输入或预置动作词形映射，失败空回退。"""
    # 步骤1：拒绝解释/代码块；不依赖正确答案、数据集类型或人工物体名单。
    try:
        value=json.loads(raw.strip())
        if set(value)!={'known','choices'} or not isinstance(value['known'],list) or not isinstance(value['choices'],list) or len(value['choices'])!=4:
            raise ValueError('Expected known and four choices')
        if len(value['known'])>6 or any(not isinstance(a,list) or len(a)>3 for a in value['choices']):raise ValueError('Object query bound exceeded')
        groups=[value['known']]+value['choices'];out=[]
        # 步骤2：词表只检查未知名词漂移，不宣称自动保证语义/对象可见性。
        for group,source in zip(groups,[question]+list(options)):
            original=re.findall(r"[a-z]+(?:'[a-z]+)?",source.lower())
            vocabulary=set(original)|FUNCTION_WORDS
            vocabulary|={w[:-1] for w in original if w.endswith('s') and len(w)>3}
            vocabulary|={v for k,v in TRANSPORT_ALIASES.items() if k in original}
            normalized=[]
            for phrase in group:
                if not isinstance(phrase,str) or not phrase.strip() or len(phrase)>80 or '\n' in phrase:raise ValueError('Invalid object phrase')
                words=re.findall(r'[a-z]+',phrase.lower())
                if not words or any(word not in vocabulary for word in words):raise ValueError('Unlicensed content word')
                canonical=' '.join(words)
                if canonical not in normalized:normalized.append(canonical)
            out.append(normalized)
        return dict(known=out[0],choices=out[1:],valid=True,fallback=False,reason=None)
    except (ValueError,TypeError,json.JSONDecodeError) as exc:
        return dict(known=[],choices=[[],[],[],[]],valid=False,fallback=True,reason=str(exc))


def object_extraction_prompt(question,options):
    """接口：只有原题和四备选描述，提取物体/可定位活动，不回答或推断未知事实。"""
    return '''Extract visually localizable objects or activities explicitly mentioned in the question and each of the four alternatives.
Do not answer the question. Do not infer which alternative is true.
Use short English phrases copied from the input; singular forms are allowed.
For transport only, these fixed conversions are allowed: biking/cycling -> bicycle; skateboarding -> skateboard; driving -> vehicle; walking -> person.
Abstract themes, numbers, logical references, and event-order symbols should produce empty lists, not invented objects.
Return JSON only: {"known": [objects from question], "choices": [[objects from first alternative], [second], [third], [fourth]]}.
At most 6 known phrases and 3 phrases per choice. Never add an object not licensed by the input.

Question:
'''+question+'\nAlternatives:\n'+'\n'.join(options)
