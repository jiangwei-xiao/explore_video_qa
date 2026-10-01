"""RD-1.2-exp10：严格恒定RGB排除，其他exp07前缀与预算目标不变。"""
import math
from fractions import Fraction
import numpy as np
from videoqa_methods.density_selection import frame_key,source_time,validate_candidates
from videoqa_methods.coarse_eligibility import coarse_selection_cells
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.prefix_coverage_budget import allocate_prefix_budget
from videoqa_audit.option_coarse_replay import plan_from_seeds


def pixel_statistics(image):
    """接口：8bit RGB整图统计；仅所有数值相同才无效，不把低照度或近似画面过滤。"""
    a=np.asarray(image)
    if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or not a.size:
        raise ValueError('Expected nonempty uint8 RGB')
    # 步骤1：精确比较，无可调阈值；单个不同像素也视为非恒定。
    lo=int(a.min());hi=int(a.max())
    return dict(min=lo,max=hi,constant_rgb=lo==hi)


def valid_seed_plan(coarse,scores,video,cfg,inspect):
    """接口：原问题稳定排名依次检查，返回非恒定种子、原区域规则及排除记录。"""
    validate_candidates(coarse,video)
    if len(scores)!=len(coarse) or any(not math.isfinite(s) or not 0<=s<=1 for s in scores):
        raise ValueError('Invalid original scores')
    order=sorted(range(len(coarse)),key=lambda i:(-scores[i],*frame_key(coarse[i])))
    accepted=[];rejected=[];target=min(cfg['seed_budget'],len(coarse))
    # 步骤1：检查次序只由原问题评分决定，不能用答案、事实或其他查询补位。
    for offset in range(0,len(order),max(1,target)):
        batch=order[offset:offset+target];stats=inspect(batch)
        for i in batch:
            if stats[i]['constant_rgb']:rejected.append(i)
            else:accepted.append(i)
            if len(accepted)==target:break
        if len(accepted)==target:break
    if not accepted:raise ValueError('No nonconstant seed candidates')
    # 步骤2：有效数量不足不重复补齐；几何仍基于原PTS与2秒/4秒规则。
    effective=dict(cfg,seed_budget=len(accepted))
    plan=plan_from_seeds(coarse,scores,accepted,video,effective)
    return plan,dict(original_order=order,accepted=accepted,rejected=rejected,
                     requested_seed_budget=cfg['seed_budget'],effective_seed_budget=len(accepted))


def select_eligible(pool,features,cfg,banned):
    """接口：只禁用选择资格，覆盖参照仍为原粗帧；无禁用时应精确复现exp07。"""
    c=pool['candidates'];video=pool['video'];validate_candidates(c,video)
    f=np.asarray(features,dtype=np.float32);n=pool['initial_count'];banned=set(banned)
    if f.ndim!=2 or len(f)!=len(c) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5,rtol=0):
        raise ValueError('Invalid normalized features')
    if not banned<=set(range(len(c))):raise ValueError('Invalid banned IDs')
    cells=coarse_selection_cells(pool['plan'],video);target=min(cfg['frame_budget'],len(c)-len(banned))
    if target<len(cells['regions']):raise ValueError('Insufficient valid frame capacity')
    records=[];all_members=set()
    # 步骤1：粗细候选归属、种子支持与覆盖参照保持原定义，不把资格过滤变成新覆盖目标。
    for pos,r in enumerate(cells['regions']):
        left,right=Fraction(r['left_fraction']),Fraction(r['right_fraction'])
        members=sorted([i for i,frame in enumerate(c) if left<=source_time(frame,video)<=right],key=lambda i:frame_key(c[i]))
        if all_members&set(members):raise ValueError('Overlapping cells')
        all_members.update(members);ids=[i for i in members if i not in banned];refs=[i for i in members if i<n]
        anchor=r['anchor_index'];original=pool['plan']['regions'][pos]
        if anchor not in ids or not set(r['seed_indices'])<=set(ids) or not refs:raise ValueError('Invalid protected seed')
        for i in members:
            if i>=n and not Fraction(original['left_fraction'])<=source_time(c[i],video)<=Fraction(original['right_fraction']):
                raise ValueError('Fine frame outside original window')
        support,_,_=weighted_support(f,ids,r['seed_indices'],pool['initial_scores'])
        chosen=[anchor];d=np.clip(f[ids]@f[anchor],0,1);covered=np.clip(f[refs]@f[anchor],0,1)
        curve=[0.,float(np.mean(covered,dtype=np.float64))];steps=[]
        # 步骤2：保持局部L×sqrt(u)和PTS/编号同分规则，禁用帧不进入前缀。
        while len(chosen)<min(target,len(ids)):
            j=min((j for j,i in enumerate(ids) if i not in chosen),key=lambda j:(-float(support[j])*math.sqrt(1-float(d[j])),*frame_key(c[ids[j]])))
            i=ids[j];steps.append(dict(candidate_index=i,support=float(support[j]),redundancy=float(d[j]),local_score=float(support[j])*math.sqrt(1-float(d[j]))))
            chosen.append(i);d=np.maximum(d,np.clip(f[ids]@f[i],0,1));covered=np.maximum(covered,np.clip(f[refs]@f[i],0,1))
            curve.append(float(np.mean(covered,dtype=np.float64)))
        records.append(dict(region_id=r['region_id'],members=members,eligible_members=ids,reference_coarse_indices=refs,
            prefix=chosen,coverage_curve=curve,density_weight=r['weight'],local_steps=steps))
    if all_members!=set(range(len(c))):raise ValueError('Candidate ownership lost')
    # 步骤3：仍用固定前缀族的精确DP；实际有效预算不足时如实少选。
    quotas,value=allocate_prefix_budget([r['coverage_curve'] for r in records],[r['density_weight'] for r in records],target)
    selected=sorted([i for r,k in zip(records,quotas) for i in r['prefix'][:k]],key=lambda i:frame_key(c[i]))
    if len(selected)!=len(set(selected)) or len(selected)!=target or set(selected)&banned:raise ValueError('Invalid final selection')
    return selected,dict(regions=records,region_budgets={str(r['region_id']):k for r,k in zip(records,quotas)},
        objective=value,banned_indices=sorted(banned),effective_frame_budget=target,
        limitation='Selection eligibility only; original coarse coverage references unchanged')


def select_checked(pool,features,cfg,inspect,known_banned=()):
    """接口：惰性检查与重选达到固定点，保证最终源帧无恒定像素；不宣称全池均已扫描。"""
    banned=set(known_banned);rounds=[]
    # 步骤1：每轮至少新增一个禁用项，故有限池内必终止或显式容量失败。
    for step in range(len(pool['candidates'])+1):
        selected,trace=select_eligible(pool,features,cfg,banned);stats=inspect(selected)
        newly={i for i in selected if stats[i]['constant_rgb']}
        rounds.append(dict(step=step,selected_indices=selected,newly_banned=sorted(newly)))
        if not newly:
            trace['eligibility_rounds']=rounds;return selected,trace
        if newly&banned:raise ValueError('Rejected frame returned')
        banned.update(newly)
    raise RuntimeError('Eligibility did not converge')
