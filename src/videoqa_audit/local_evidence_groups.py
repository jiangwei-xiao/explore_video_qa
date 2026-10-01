"""局部证据组合的零问答对照：固定exp07窗口、池、配额，仅补缺失侧。"""
import math
from fractions import Fraction
import numpy as np
from videoqa_methods.density_selection import source_time,frame_key,validate_candidates
from videoqa_methods.weighted_seed_support import weighted_support


def complete_anchor_sides(pool,features,reference_trace):
    """接口：输入冻结候选/视觉特征/原前缀预算，输出固定配额选择及最多两次局部替换。"""
    c=pool['candidates'];video=pool['video'];validate_candidates(c,video)
    f=np.asarray(features,dtype=np.float32)
    if f.ndim!=2 or len(f)!=len(c) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5,rtol=0):
        raise ValueError('Invalid normalized features')
    records=[];all_before=[];all_after=[]
    # 步骤1：原配额与锚点冻结；只有原补查条件成立且能容纳三帧的区域可尝试。
    for region in reference_trace['regions']:
        rid=region['region_id'];original=next(r for r in pool['plan']['regions'] if r['region_id']==rid)
        quota=reference_trace['region_budgets'][str(rid)];base=region['prefix'][:quota]
        anchor=original['anchor_index'];assert base[0]==anchor and len(base)==quota
        members=region['members'];left,right=Fraction(original['left_fraction']),Fraction(original['right_fraction'])
        instant=source_time(c[anchor],video)
        sides={'before':{i for i in members if left<=source_time(c[i],video)<instant},
               'after':{i for i in members if instant<source_time(c[i],video)<=right}}
        eligible=bool(original['refine'] and quota>=3 and all(sides.values()))
        selected=list(base);added=[];changes=[]
        if eligible:
            support,_,_=weighted_support(f,members,original['seed_indices'],pool['initial_scores'])
            support=dict(zip(members,map(float,support)))
            # 步骤2：最多修补两侧。保留锚点、已补帧和每侧唯一见证，不破坏已有两侧覆盖。
            for _ in range(2):
                missing=[name for name,ids in sides.items() if not ids&set(selected)]
                if not missing:break
                removable=[i for i in reversed(base) if i in selected and i!=anchor and i not in added
                    and all(not (i in ids and len(ids&set(selected))==1) for ids in sides.values())]
                if not removable:raise ValueError('Missing side but no removable slot')
                victim=removable[0];kept=[i for i in selected if i!=victim]
                proposals=[]
                for side in missing:
                    for i in sides[side]-set(selected):
                        redundancy=float(np.max(np.clip(f[kept]@f[i],0,1)))
                        proposals.append(dict(candidate_index=i,side=side,support=support[i],redundancy=redundancy,
                            score=support[i]*math.sqrt(1-redundancy)))
                if not proposals:raise ValueError('Side candidates unexpectedly empty')
                winner=min(proposals,key=lambda x:(-x['score'],*frame_key(c[x['candidate_index']])))
                selected=kept+[winner['candidate_index']];added.append(winner['candidate_index'])
                changes.append(dict(removed_index=victim,added_index=winner['candidate_index'],winner=winner,
                    reference_indices=kept,removal_rule='last original local-prefix item not protected or sole side witness'))
        # 步骤3：约束只保证时间两侧有画面，不宣称事件正确、关系完整或答案可得。
        if len(selected)!=quota or len(set(selected))!=quota or anchor not in selected:raise ValueError('Quota/anchor violation')
        if eligible and not all(ids&set(selected) for ids in sides.values()):raise ValueError('Side repair incomplete')
        if len(set(base)-set(selected))>2:raise ValueError('More than two replacements')
        all_before.extend(base);all_after.extend(selected)
        records.append(dict(region_id=rid,quota=quota,eligible=eligible,anchor_index=anchor,
            old_indices=base,new_indices=selected,changes=changes,
            old_side_coverage={name:bool(ids&set(base)) for name,ids in sides.items()},
            new_side_coverage={name:bool(ids&set(selected)) for name,ids in sides.items()}))
    if len(set(all_after))!=len(all_before) or len(set(all_before))!=len(all_before):raise ValueError('Global identity mismatch')
    selected=sorted(all_after,key=lambda i:frame_key(c[i]))
    return selected,dict(rule='minimum missing-side repair at fixed exp07 quotas',regions=records,
        region_budgets=reference_trace['region_budgets'],replacement_count=len(set(all_before)-set(all_after)),
        model_calls=0,limitation='Temporal geometry only; no semantic event detector or QA evidence guarantee')
