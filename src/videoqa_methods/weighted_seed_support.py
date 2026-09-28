"""冻结配额下的相关性加权种子支持；只改变第二遍支持函数，不修改RD-1.2。"""
import math
from fractions import Fraction
import numpy as np
from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_methods.density_selection import validate_candidates,source_time,frame_key


def weighted_support(features,members,seeds,initial_scores):
    """接口：把原问题ITM的相对种子强度传递给邻域新旧帧，不补算细帧ITM。"""
    scores=np.asarray(initial_scores,dtype=np.float64)
    if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():
        raise ProtocolError('Invalid initial ITM scores')
    if not seeds or min(seeds)<0 or max(seeds)>=len(scores):raise ProtocolError('Seeds must be initial candidates')
    # 步骤1：区域内最高分归一化，不把不同问题的BLIP分数比较为证据置信度。
    peak=float(max(scores[j] for j in seeds))
    weights={j:float(scores[j]/peak) if peak>0 else 1.0 for j in seeds}
    # 步骤2：保留原max支持结构，仅将每个种子的单位权重改为其相对ITM分数。
    # 这是外观邻域的检索支持代理，不声称估计了回答充分性或语义因果关系。
    support=np.maximum.reduce([weights[j]*np.clip(features[members]@features[j],0,1) for j in seeds])
    return support,weights,peak==0


def select_weighted_seed_support(candidates,features,plan,video,cfg,quotas,initial_scores):
    """接口：固定窗口、池、锚点与配额，按L_r×sqrt(1-D)填满唯一源帧预算。"""
    validate_candidates(candidates,video);f=np.asarray(features,dtype=np.float32)
    if f.ndim!=2 or len(f)!=len(candidates) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5,rtol=0):
        raise ProtocolError('Invalid normalized features')
    if set(quotas)!={str(r['region_id']) for r in plan['regions']}:raise ProtocolError('Quota scope mismatch')
    if len(initial_scores)>len(candidates):raise ProtocolError('Scores extend beyond candidate pool')
    picked={};eligible=[];trace=[];support_trace={}
    # 步骤1：外部原RD-1.2配额逐个冻结，不能为新支持规则重分配或借用名额。
    for region in plan['regions']:
        rid=region['region_id'];left,right=Fraction(region['left_fraction']),Fraction(region['right_fraction'])
        members=sorted([i for i,r in enumerate(candidates) if left<=source_time(r,video)<=right],key=lambda i:frame_key(candidates[i]))
        budget=quotas[str(rid)];anchor=region['anchor_index']
        if anchor not in members or not set(region['seed_indices'])<=set(members) or set(members)&set(eligible):raise ProtocolError('Invalid region membership')
        if not isinstance(budget,int) or not 1<=budget<=len(members):raise ProtocolError('Invalid fixed quota')
        eligible.extend(members);selected=[anchor];picked[str(rid)]=selected
        support,weights,fallback=weighted_support(f,members,region['seed_indices'],initial_scores)
        support_trace[str(rid)]=dict(seed_weights={str(k):v for k,v in weights.items()},all_zero_fallback=fallback,
            values=dict(zip(map(str,members),map(float,support))))
        # 步骤2：新旧帧同式竞争；只使用已有视觉特征，来源标记不产生优先级。
        while len(selected)<budget:
            redundant=np.maximum.reduce([np.clip(f[members]@f[j],0,1) for j in selected])
            proposals=[]
            for pos,i in enumerate(members):
                if i in selected:continue
                L=float(support[pos]);D=float(redundant[pos]);u=1-D
                proposals.append(dict(candidate_index=i,support=L,redundancy=D,squared_distance=u,
                    distance=math.sqrt(u),local_score=L*math.sqrt(u)))
            winner=min(proposals,key=lambda p:(-p['local_score'],*frame_key(candidates[p['candidate_index']])))
            trace.append(dict(region_id=rid,selected_before=list(selected),winner=winner,proposals=proposals))
            selected.append(winner['candidate_index'])
    # 步骤3：恢复源时间顺序；零增益仍按稳定规则填满，最终预算与原配额一致。
    indices=sorted([i for group in picked.values() for i in group],key=lambda i:frame_key(candidates[i]))
    if len(indices)!=min(cfg['frame_budget'],len(eligible)) or len(set(indices))!=len(indices):raise ProtocolError('Fixed total budget mismatch')
    return indices,dict(local_rule='max_j((r_j/max_region_seed_r)*cos_+(i,j))*sqrt(1-D)',
        budget_rule='frozen RD-1.2 first-pass quotas; no reallocation',
        protected_indices=[r['anchor_index'] for r in plan['regions']],region_selected=picked,
        region_budgets={k:len(v) for k,v in picked.items()},eligible_indices=sorted(eligible),supports=support_trace,rounds=trace)
