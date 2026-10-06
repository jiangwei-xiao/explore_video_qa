"""问题加权的真实观测时间覆盖：不把外观相似等同于状态重复。"""
import numpy as np
from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget
from videoqa_methods.density_selection import frame_key,validate_candidates


def temporal_kernel(reference_times,candidate_times):
    """接口：已观测PTS时间点的平方距离覆盖核，不声称帧代表连续时间区间。

    用当前单元真实候选时间跨度统一单位；C=1−加权最近平方距离/span²。
    """
    a=np.asarray(reference_times,dtype=np.float64);b=np.asarray(candidate_times,dtype=np.float64)
    if a.ndim!=1 or b.ndim!=1 or not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid observed time coordinates')
    if a.min()<b.min()-1e-8 or a.max()>b.max()+1e-8:
        raise ValueError('Reference observation outside candidate span')
    if len(a)*len(b)*8>512*1024**2:
        raise ValueError('Temporal workspace exceeds512MiB; pause without altering objective')
    # 步骤1：只对离散粗观测赋权，补查帧不增加参考质量或时间区间权重。
    span=float(b.max()-b.min())
    if span==0:return np.ones((len(a),len(b)),dtype=np.float64),span
    # 步骤2：对应归一化平方量化误差，核有限且0–1；没有半径/温度/指数搜索。
    return np.clip(1-((a[:,None]-b[None,:])/span)**2,0,1),span


def select_temporal_coverage(pool,parent_trace,budget=32):
    """接口：固定原池/种子/锚点/密度，以时间覆盖贪心前缀和原DP选择唯一帧。

    唯一替换为选择目标的时间几何；不调用视觉/文本模型、不接受答案或人工见证。
    DP仍仅在确定的前缀族中最优，不宣称证据充分性或QA最优。
    """
    c=pool['candidates'];validate_candidates(c,pool['video'])
    scores=np.asarray(pool['initial_scores'],dtype=np.float64)
    if len(scores)!=pool['initial_count'] or not np.isfinite(scores).all() or np.any(scores<0) or np.any(scores>1):
        raise ValueError('Invalid original ITM reference weights')
    target=min(budget,len(c))
    if target!=sum(parent_trace['region_budgets'].values()):raise ValueError('Parent actual budget differs')
    records=[];seen=set()
    # 步骤1：复用父版本资格单元和实际粗参考，不重新分段、加密或改变局部窗口。
    for original,region in zip(parent_trace['regions'],pool['plan']['regions']):
        if original['region_id']!=region['region_id']:raise ValueError('Region order differs')
        members=original['members'];refs=original['reference_coarse_indices'];anchor=region['anchor_index']
        if not refs or not set(refs)<=set(members) or anchor not in members or seen&set(members):
            raise ValueError('Temporal reference/member identity differs')
        seen.update(members)
        times=np.array([c[i]['timestamp_seconds'] for i in members],dtype=np.float64)
        reference=np.array([c[i]['timestamp_seconds'] for i in refs],dtype=np.float64)
        kernel,span=temporal_kernel(reference,times)
        weights=scores[refs];mass=float(weights.sum())
        weights=weights/mass if mass else np.full(len(refs),1/len(refs),dtype=np.float64)
        positions={i:j for j,i in enumerate(members)};chosen=[anchor]
        covered=kernel[:,positions[anchor]].copy();curve=[0.,float(weights@covered)];steps=[]
        # 步骤2：增量只奖励降低加权最近时间误差，不奖励视觉离群或罚相似状态。
        while len(chosen)<min(target,len(members)):
            gains=np.empty(len(members),dtype=np.float64)
            for start in range(0,len(members),256):
                gains[start:start+256]=weights@np.maximum(kernel[:,start:start+256]-covered[:,None],0)
            winner=min((i for i in members if i not in chosen),key=lambda i:(-float(gains[positions[i]]),*frame_key(c[i])))
            chosen.append(winner);covered=np.maximum(covered,kernel[:,positions[winner]])
            value=min(1.,max(curve[-1],float(weights@covered)));curve.append(value)
            steps.append(dict(candidate_index=winner,weighted_temporal_gain=float(gains[positions[winner]]),
                coverage_after=value,normalized_nearest_squared_time_error=1-value))
        records.append(dict(region_id=region['region_id'],members=members,reference_coarse_indices=refs,
            reference_weights=weights.tolist(),reference_times=reference.tolist(),candidate_time_span_seconds=span,
            prefix=chosen,coverage_curve=curve,density_weight=original['density_weight'],local_steps=steps))
    if seen!=set(range(len(c))):raise ValueError('Global coarse/fine candidate eligibility changed')
    # 步骤3：保留原正配额/密度和总预算，以新前缀实际覆盖曲线执行相同DP。
    quotas,value=allocate_prefix_budget([r['coverage_curve'] for r in records],[r['density_weight'] for r in records],target)
    selected=sorted([i for r,k in zip(records,quotas) for i in r['prefix'][:k]],key=lambda i:frame_key(c[i]))
    fixed=sorted([i for r in records for i in r['prefix'][:parent_trace['region_budgets'][str(r['region_id'])]]],key=lambda i:frame_key(c[i]))
    protected=[r['anchor_index'] for r in pool['plan']['regions']]
    if len(selected)!=len(set(selected)) or len(selected)!=target or not set(protected)<=set(selected):
        raise ValueError('Budget/anchor invariant failed')
    return selected,dict(regions=records,region_budgets={str(r['region_id']):k for r,k in zip(records,quotas)},
        protected_indices=protected,fixed_parent_quota_indices=fixed,objective=value,
        rule='Original-question ITM-weighted normalized squared temporal quantization, greedy prefixes and unchanged positive-quota DP',
        limitation='Temporal coverage of observed coarse samples is a proxy, not distinct semantic states, event boundaries or answer sufficiency')
