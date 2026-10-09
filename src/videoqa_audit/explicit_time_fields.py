"""显式时间字段离线提议：识别可见格式，不把字段或上下文分数当作回答价值。"""
from fractions import Fraction
import math
import re
from videoqa_methods.density_selection import source_time, frame_key

TIME_QUESTION = re.compile(r'how long|how much time|how many (?:minutes|seconds)|(?:at )?what time',re.I)
NUMERIC_TIME = re.compile(r'(?<![\d:.])(?P<a>\d{1,2})\s*(?P<sep>[:.])\s*(?P<b>\d{2})(?:\s*:\s*(?P<c>\d{2}))?\s*(?:(?P<amp>[ap]\.?m\.?)(?!\w))?(?![\d:.])',re.I)
HOUR_TIME = re.compile(r'(?<![\d:.])(?P<a>\d{1,2})\s*(?P<amp>[ap]\.?m\.?)(?!\w)',re.I)
DURATION = re.compile(r'(?<![\d.:])(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b',re.I)


def question_mode(question):
    """接口：原题字面线索→观察模式；不读选项、答案或官方题型，不声称完整语义分类。"""
    if not TIME_QUESTION.search(question): return None
    if re.search(r'what time of (?:the )?day',question,re.I): return 'qualitative'
    if re.search(r'how long|how much time|how many (?:minutes|seconds)',question,re.I): return 'duration'
    return 'clock'


def extract_time_fields(regions, mode):
    """接口：原始OCR框/文字/置信度→显式字段；固定0.5默认识别门槛，不补未知数字。

    小数点格式可能是价格/数值，单列歧义；没有单位或日期上下文时，不推断
    它究竟是墙钟还是视频计时。不跨框补AM/PM，不纠正错误数字。
    """
    if mode not in ('clock','duration','qualitative'): raise ValueError('Invalid observation mode')
    fields=[]
    # 步骤1：原输出全部保留，字段消费沿用OCR默认分数，不搜索置信阈值。
    for r in regions:
        score=float(r['score'])
        if not math.isfinite(score) or not 0<=score<=1: raise ValueError('Invalid OCR confidence')
        if score<0.5 or mode=='qualitative': continue
        text=r['text'].replace('：',':'); covered=[]
        for pattern,kind in ((NUMERIC_TIME,'numeric_time'),(HOUR_TIME,'hour_period')):
            for match in pattern.finditer(text):
                if any(match.start()<b and a<match.end() for a,b in covered): continue
                h=int(match['a']); amp=match['amp']; amp=amp.lower().replace('.','') if amp else None
                minute=int(match['b']) if kind=='numeric_time' else None
                second=int(match['c']) if kind=='numeric_time' and match['c'] else None
                if (mode=='clock' and h>23) or (amp is not None and not 1<=h<=12) or (minute is not None and minute>59) or (second is not None and second>59): continue
                # 小数点无需被擅自认定为时钟：保留歧义；时长模式只接受冒号或明确单位。
                dot=kind=='numeric_time' and match['sep']=='.'
                if dot and mode=='duration': continue
                if mode=='duration' and amp is not None: continue
                if re.search(r'[$€£]\s*$',text[:match.start()]): continue
                key=f'time:{h:02d}:'+('?' if minute is None else f'{minute:02d}')
                if second is not None: key+=f':{second:02d}'
                key+=':'+str(amp or '?')
                fields.append(dict(key=key,kind=kind,displayed_text=match.group().strip(),full_region_text=r['text'],
                    score=score,box=r['box'],decimal_ambiguity=dot and amp is None,semantic_attribution_verified=False))
                covered.append(match.span())
        if mode=='duration':
            for match in DURATION.finditer(text):
                unit=match['unit'].lower()
                unit='second' if unit.startswith('sec') else 'minute' if unit.startswith('min') else 'hour' if unit.startswith(('hour','hr')) else unit.rstrip('s')
                fields.append(dict(key='duration:'+match['number']+':'+unit,kind='duration_unit',
                    displayed_text=match.group(),full_region_text=r['text'],score=score,box=r['box'],
                    decimal_ambiguity=False,semantic_attribution_verified=False))
    # 步骤2：同框重复模式去重，不把词序、识别置信度或时间值当作证据充分性。
    return fields


def propose_time_regions(candidates, scores, fields, video):
    """接口：完整粗候选/既有原题ITM/可见字段→区域、唯一载体提议及相邻语境。

    同字段在连续粗观测中连接；上下文沿用RD的±4秒尺度，不改变原方法窗口。
    最大邻域ITM只是语境定位代理，不识别所问事件，不决定最终32/8帧。
    """
    if len(candidates)!=len(scores) or len(fields)!=len(scores): raise ValueError('Field/source correspondence differs')
    if any(not math.isfinite(float(s)) or not 0<=float(s)<=1 for s in scores): raise ValueError('Invalid frozen ITM score')
    times=[source_time(c,video) for c in candidates]
    if times!=sorted(set(times)): raise ValueError('Coarse sources not unique/time ordered')
    # 步骤1：用字段值整理重复观测；漏检会断开区域，不自动填补语义状态。
    by_key={}
    for i,observations in enumerate(fields):
        for value in observations:
            key=value['key']; per_frame=by_key.setdefault(key,{})
            if i not in per_frame or value['score']>per_frame[i]['score']: per_frame[i]=value
    regions=[]; duration=Fraction(video['duration_fraction'])
    for key,observations in sorted(by_key.items()):
        groups=[]
        for i in sorted(observations):
            if groups and i==groups[-1][-1]+1: groups[-1].append(i)
            else: groups.append([i])
        for members in groups:
            left=max(Fraction(0),times[members[0]]-4); right=min(duration,times[members[-1]]+4)
            context=[i for i,t in enumerate(times) if left<=t<=right]
            representative=min(members,key=lambda i:(-observations[i]['score'],*frame_key(candidates[i])))
            anchor=min(context,key=lambda i:(-float(scores[i]),*frame_key(candidates[i])))
            regions.append(dict(field_key=key,members=members,representative_index=representative,
                displayed_field=observations[representative],context_indices=context,context_anchor_index=anchor,
                context_score=float(scores[anchor]),left_fraction=str(left),right_fraction=str(right)))
    # 步骤2：相同源载体只提议一次，完整保留各字段；无人工事实或标准答案输入。
    proposals={}
    for region in regions:
        i=region['representative_index']
        p=proposals.setdefault(i,dict(candidate_index=i,frame=candidates[i],regions=[],context_score=0.,confidence=0.))
        p['regions'].append(region); p['context_score']=max(p['context_score'],region['context_score'])
        p['confidence']=max(p['confidence'],region['displayed_field']['score'])
    ordered=sorted(proposals.values(),key=lambda p:(-p['context_score'],-p['confidence'],*frame_key(p['frame'])))
    # 步骤3：返回全部提议及排名，观察前4项仅为审计预算，不是新的种子/帧预算参数。
    return dict(regions=regions,proposals=[dict(p,rank=i+1) for i,p in enumerate(ordered)],
        context_radius_seconds=4,selection_writes=0,
        limitation='Visible formats plus neighboring original ITM; not event binding or final frame selection')
