"""exp21：只扩展局部支持参考，不改变原窗口、候选池、锚点或配额。"""
from fractions import Fraction
import numpy as np
from videoqa_methods.density_selection import source_time,frame_key,validate_candidates
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_audit.temporal_support_probe import local_prefix


def propose_references(pool):
    """接口：原问题分数与真实时间提议至多4帧；不读取题号、题型、选项或答案。"""
    c=pool['candidates'];scores=pool['initial_scores'];video=pool['video'];windows=pool['plan']['regions']
    # 步骤1：原窗口闭区间排除，正分候选稳定排序；4秒以内不能重复提议。
    eligible=[i for i in range(pool['initial_count']) if scores[i]>0 and not any(
        Fraction(w['left_fraction'])<=source_time(c[i],video)<=Fraction(w['right_fraction']) for w in windows)]
    ordered=sorted(eligible,key=lambda i:(-scores[i],*frame_key(c[i])))
    chosen=[];suppressed=[]
    for i in ordered:
        previous=next((s for s in chosen if abs(source_time(c[i],video)-source_time(c[s],video))<=4),None)
        if previous is not None:suppressed.append(dict(candidate_index=i,suppressor=previous));continue
        chosen.append(i)
        if len(chosen)==4:break
    return chosen,suppressed


def select_external_support(pool,features,prefix_trace,budget_trace):
    """接口：原exp07前缀和exp12预算为固定控制，输出最终选择与完整局部追踪。"""
    c=pool['candidates'];validate_candidates(c,pool['video']);f=np.asarray(features,dtype=np.float32)
    if f.ndim!=2 or len(f)!=len(c) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5,rtol=0):
        raise ValueError('Invalid frozen visual features')
    proposals,suppressed=propose_references(pool);keys=[frame_key(x) for x in c]
    plans={r['region_id']:r for r in pool['plan']['regions']};selected=[];before=[];regions=[]
    # 步骤1：原支持必须复现冻结前缀，避免将实现差异误计为支持参考收益。
    for r in prefix_trace['regions']:
        rid=r['region_id'];members=r['members'];seeds=plans[rid]['seed_indices'];horizon=len(r['prefix'])
        original,_,zero0=weighted_support(f,members,seeds,pool['initial_scores'])
        control,_=local_prefix(f,members,r['prefix'][0],original,keys,horizon)
        if control!=r['prefix']:raise ValueError('Parent prefix replay mismatch')
        # 步骤2：只扩展支持集合；峰值归一化不变，去重仍依据同单元已选帧。
        additions=[i for i in proposals if i in members]
        scores=pool['initial_scores']
        assert max(scores[i] for i in seeds)==max(scores[i] for i in seeds+additions)
        support,_,zero1=weighted_support(f,members,seeds+additions,scores)
        assert zero0==zero1 and np.all(support+1e-7>=original)
        prefix,steps=local_prefix(f,members,r['prefix'][0],support,keys,horizon)
        quota=budget_trace['region_budgets'][str(rid)];assert 1<=quota<=horizon
        selected.extend(prefix[:quota]);before.extend(control[:quota])
        regions.append(dict(region_id=rid,quota=quota,extra_support=additions,prefix=prefix,steps=steps))
    # 步骤3：名额与原锚点严格保留；不承诺最多换4帧或最终覆盖目标单调增加。
    selected.sort(key=lambda i:keys[i]);before.sort(key=lambda i:keys[i])
    assert len(selected)==len(set(selected))==len(before)
    assert set(prefix_trace['protected_indices'])<=set(selected)
    return selected,dict(proposals=proposals,suppressed=suppressed,regions=regions,old_indices=before,
        region_budgets=dict(budget_trace['region_budgets']),protected_indices=prefix_trace['protected_indices'])
