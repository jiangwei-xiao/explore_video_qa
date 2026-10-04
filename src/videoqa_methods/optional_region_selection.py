"""更大定位提议池的可选区域预算；独立实现，不改exp12冻结算法。"""
from fractions import Fraction
import math
import numpy as np
from videoqa_methods.density_selection import build_regions,validate_candidates,source_time,frame_key
from videoqa_methods.coarse_eligibility import coarse_selection_cells
from videoqa_methods.weighted_seed_support import weighted_support


def build_optional_regions(candidates,scores,video,cfg):
    """接口：区域是候选提议，不在构建阶段强迫消耗一帧；最终K由选择器约束。

    原构建函数只有容量检查使用frame_budget；暂以种子上限允许所有提议输出，
    不改变核心连接、扩窗、合并、触发或密度，也不将此配置用于最终帧数。
    """
    planning=dict(cfg,frame_budget=max(cfg['frame_budget'],cfg['seed_budget']))
    return build_regions(candidates,scores,video,planning)


def allocate_optional(curves,weights,budget):
    """接口：在原加权覆盖目标下允许n_r=0；同分沿用较早区域更多名额。"""
    if not curves or len(curves)!=len(weights) or budget<0:raise ValueError('预算范围错误')
    for curve,w in zip(curves,weights):
        if len(curve)<2 or curve[0]!=0 or not math.isfinite(w) or w<=0:
            raise ValueError('覆盖曲线或权重错误')
        if any(not math.isfinite(x) or not 0<=x<=1 for x in curve) or any(b<a for a,b in zip(curve,curve[1:])):
            raise ValueError('覆盖曲线必须有限且单调')
    # 步骤1：只放宽配额下界，不添加阈值、复杂度惩罚或新的归一化。
    states={0:(0.,())}
    for curve,w in zip(curves,weights):
        next_states={}
        for used,(value,allocation) in states.items():
            for n in range(min(len(curve)-1,budget-used)+1):
                proposal=(value+w*curve[n],allocation+(n,))
                if used+n not in next_states or proposal>next_states[used+n]:next_states[used+n]=proposal
        states=next_states
    # 步骤2：精确闭合总预算，不能靠少选或重复帧满足约束。
    if budget not in states:raise ValueError('候选容量不足')
    value,counts=states[budget]
    return list(counts),value


def select_optional_regions(pool,features,frame_budget=32):
    """接口：原L×sqrt差异局部前缀和原问题加权粗参考不变，只允许零区域配额。

    新旧帧同等竞争；细帧不进入覆盖参考。正配额区域才保护其最高分种子。
    覆盖仍是视觉代理，不保证答题证据充分。
    """
    c=pool['candidates'];v=pool['video'];n=pool['initial_count']
    validate_candidates(c,v)
    f=np.asarray(features,dtype=np.float32);scores=np.asarray(pool['initial_scores'],dtype=np.float64)
    if frame_budget<1 or len(scores)!=n or not 1<=n<=len(c):raise ValueError('预算或初始池不符')
    if not np.isfinite(scores).all() or np.any(scores<0) or np.any(scores>1):raise ValueError('分数不合法')
    if f.ndim!=2 or len(f)!=len(c) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5,rtol=0):
        raise ValueError('归一化特征不符')
    target=min(frame_budget,len(c));plan=pool['plan'];cells=coarse_selection_cells(plan,v)
    times=[source_time(x,v) for x in c];keys=[frame_key(x) for x in c];records=[];assigned=set()
    # 步骤1：全局粗资格保持，细帧仍仅来自原补查窗口；选择单元不扩大解码范围。
    for pos,r in enumerate(cells['regions']):
        left,right=Fraction(r['left_fraction']),Fraction(r['right_fraction'])
        ids=sorted([i for i,t in enumerate(times) if left<=t<=right],key=lambda i:keys[i])
        refs=[i for i in ids if i<n];original=plan['regions'][pos];anchor=original['anchor_index']
        if assigned.intersection(ids) or not refs or anchor not in ids or not set(original['seed_indices'])<=set(ids):
            raise ValueError('选择单元身份冲突')
        for i in ids:
            if i>=n and not Fraction(original['left_fraction'])<=times[i]<=Fraction(original['right_fraction']):
                raise ValueError('细帧越出实际窗口')
        assigned.update(ids);support,_,_=weighted_support(f,ids,original['seed_indices'],scores)
        chosen=[anchor];redundant=np.clip(f[ids]@f[anchor],0,1);steps=[]
        # 步骤2：逐项复用原局部规则，即使该区域最终为零配额也完整评估可行前缀。
        while len(chosen)<min(target,len(ids)):
            proposals=[(float(support[j])*math.sqrt(1-float(redundant[j])),j,i) for j,i in enumerate(ids) if i not in chosen]
            gain,j,i=min(proposals,key=lambda p:(-p[0],*keys[p[2]]))
            steps.append(dict(candidate_index=i,support=float(support[j]),redundancy=float(redundant[j]),local_score=gain))
            chosen.append(i);redundant=np.maximum(redundant,np.clip(f[ids]@f[i],0,1))
        raw=scores[refs];mass=float(raw.sum());uniform=bool(np.all(raw==raw[0]))
        weights=np.full(len(refs),1/len(refs),dtype=np.float64) if mass==0 else raw/mass
        covered=np.zeros(len(refs),dtype=np.float32);curve=[0.]
        for i in chosen:
            covered=np.maximum(covered,np.clip(f[refs]@f[i],0,1))
            value=float(covered.mean(dtype=np.float64)) if uniform else min(1.,max(curve[-1],float(weights@covered)))
            curve.append(value)
        records.append(dict(region_id=r['region_id'],prefix=chosen,local_steps=steps,
            reference_coarse_indices=refs,reference_weights=weights.tolist(),coverage_curve=curve,density_weight=r['weight']))
    if assigned!=set(range(len(c))):raise ValueError('候选资格遗漏')
    # 步骤3：候选区域可以不入选，但正配额必须包含原锚点，总帧数仍严格受K约束。
    counts,value=allocate_optional([r['coverage_curve'] for r in records],[r['density_weight'] for r in records],target)
    selected=sorted([i for r,k in zip(records,counts) for i in r['prefix'][:k]],key=lambda i:keys[i])
    if len(selected)!=target or len(set(selected))!=target:raise ValueError('最终预算不闭合')
    return selected,dict(regions=records,region_budgets={str(r['region_id']):k for r,k in zip(records,counts)},
        objective=value,frame_budget=frame_budget,effective_frame_budget=target,
        active_anchors=[r['prefix'][0] for r,k in zip(records,counts) if k>0],
        zero_quota_regions=[r['region_id'] for r,k in zip(records,counts) if k==0],
        limitation='原覆盖目标可行域扩展，不代表证据充分性或问答最优')
