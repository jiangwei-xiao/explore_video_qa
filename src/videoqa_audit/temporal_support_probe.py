"""时间支持单项离线探针，不接入正式方法或读取标准答案。"""
import math
import numpy as np


def temporal_support(times,coarse_scores,tau=4.):
    """接口：有序时间与粗帧分数映射为指数衰减最大支持，细帧用None，不新增ITM。"""
    if not math.isfinite(tau) or tau<=0 or len(times)!=len(coarse_scores) or not times:
        raise ValueError('Invalid temporal support input')
    if any(not math.isfinite(float(t)) for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Times must be finite and strictly increasing')
    raw=[0. if r is None else float(r) for r in coarse_scores]
    if any(not math.isfinite(r) or not 0<=r<=1 for r in raw):raise ValueError('Invalid ITM')
    peak=max(raw)
    if peak==0:return np.ones(len(times),dtype=np.float64)
    base=np.asarray(raw,dtype=np.float64)/peak
    forward=base.copy();backward=base.copy()
    # 步骤1：左右两次扫描精确计算max_j r_j exp(-|t_i-t_j|/tau)，不构造平方距离矩阵。
    for i in range(1,len(times)):
        forward[i]=max(base[i],forward[i-1]*math.exp(-float(times[i]-times[i-1])/tau))
    for i in range(len(times)-2,-1,-1):
        backward[i]=max(base[i],backward[i+1]*math.exp(-float(times[i+1]-times[i])/tau))
    # 步骤2：支持只表示邻域检索强度，不是事件联系或证据充分性概率。
    return np.maximum(forward,backward)


def local_prefix(features,ids,anchor,support,frame_keys,budget):
    """接口：保留原锚点和sqrt差异规则，仅替换支持输入；输出可复算前缀和每步记录。"""
    if len(ids)!=len(set(ids)) or anchor not in ids or len(support)!=len(ids):raise ValueError('Invalid members')
    if budget<1 or budget>len(ids):raise ValueError('Invalid prefix budget')
    selected=[anchor];trace=[]
    # 步骤1：同段已选帧决定最大截断余弦相似度，新旧候选无来源优先级。
    redundancy=np.clip(features[ids]@features[anchor],0,1)
    while len(selected)<budget:
        scores=[float(support[j])*math.sqrt(1-float(redundancy[j])) for j in range(len(ids))]
        j=min((j for j,i in enumerate(ids) if i not in selected),key=lambda j:(-scores[j],*frame_keys[ids[j]]))
        i=ids[j];trace.append(dict(candidate_index=i,support=float(support[j]),redundancy=float(redundancy[j]),score=scores[j]))
        selected.append(i);redundancy=np.maximum(redundancy,np.clip(features[ids]@features[i],0,1))
    return selected,trace
