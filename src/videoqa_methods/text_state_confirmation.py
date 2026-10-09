"""exp30：用相邻粗观测确认词集合，再复用exp29的有方向重复抑制。"""
from bisect import bisect_left


def confirm_text_states(candidates,initial_count,raw_states):
    """接口：唯一源粗细候选与OCR词集合→两粗观测确认的状态，不重新识别文字。

    粗帧词须在前/后粗邻居之一复现；细帧词须在夹住它的两个粗帧都出现。
    确认依据固定1FPS粗观测，避免补查密度改变可靠性门槛；不把重现当语义真值。
    不给只有一帧可见的短文字额外奖励，也不删除其候选，仍退回原视觉规则。
    """
    if not 1<=initial_count<=len(candidates) or len(raw_states)!=len(candidates):raise ValueError('Invalid text bank correspondence')
    coarse=candidates[:initial_count];pts=[int(c['source_pts']) for c in coarse]
    if pts!=sorted(pts) or len(set(pts))!=len(pts):raise ValueError('Coarse source order differs')
    raw=[set(x) for x in raw_states];result=[];trace=[]
    # 步骤1：仅用相邻唯一粗来源确认，不按平均FPS或新增细帧数量推导置信度。
    for i,c in enumerate(candidates):
        if i<initial_count:
            neighbors=[j for j in (i-1,i+1) if 0<=j<initial_count]
            repeated=set().union(*(raw[j] for j in neighbors)) if neighbors else set()
            tokens=raw[i]&repeated;refs=[i]+neighbors;kind='coarse_neighbor_recurrence'
        else:
            right=bisect_left(pts,int(c['source_pts']));left=right-1
            refs=[left,right] if 0<=left<right<initial_count else []
            tokens=raw[i]&raw[left]&raw[right] if refs else set();kind='fine_bracketing_two_coarse'
        result.append(sorted(tokens))
        # 步骤2：留痕被去掉的词，前后粗时间间隔仍可追踪；确认不代表整个区间恒定。
        trace.append(dict(candidate_index=i,confirmation=kind,coarse_references=refs,
                          raw_tokens=sorted(raw[i]),confirmed_tokens=sorted(tokens),removed_tokens=sorted(raw[i]-tokens)))
    return result,trace
