"""定位种子预算单因素：由冻结M32映射重建更小种子数的精确补查池。"""
from fractions import Fraction
import numpy as np
from videoqa_methods.density_selection import build_regions,refinement_targets,source_time,validate_candidates


def retarget_pool(pool,features,cfg):
    """接口：返回新池、同序特征与原索引映射；不解码、不计算新特征、不修改旧资产。"""
    n=pool['initial_count'];old=pool['candidates'];f=np.asarray(features)
    if len(f)!=len(old) or cfg['seed_budget']>pool['plan']['effective_seed_count'] and pool['plan']['effective_seed_count']<n:
        raise ValueError('Input feature identity or seed subset invalid')
    plan=build_regions(old[:n],pool['initial_scores'],pool['video'],cfg)
    if not set(plan['seed_indices'])<=set(pool['plan']['seed_indices']):raise ValueError('Seeds are not a frozen subset')
    # 步骤1：旧映射按有理数采样目标索引；不按候选编号或平均FPS猜测新位置。
    mappings={}
    for r in pool['refinement']:
        for m in r['mappings']:
            key=Fraction(m['target_fraction'])
            if key in mappings and mappings[key]!=m:raise ValueError('Conflicting frozen target mapping')
            mappings[key]=m
    candidates=[dict(c) for c in old[:n]];indices=list(range(n));seen={c['source_pts'] for c in candidates};records=[]
    # 步骤2：只保留新条件触发区域的2FPS目标，再按实际PTS执行新窗口裁剪及源帧去重。
    for region in plan['regions']:
        targets=refinement_targets(region,pool['video'],cfg) if region['refine'] else []
        kept=[];checks=[]
        for target in targets:
            if target not in mappings:raise ValueError('Missing frozen refinement target; no silent new inference')
            m=mappings[target];status=m['status']
            if status=='no_source_in_window':
                checks.append(dict(target_fraction=str(target),status='frozen_no_source'));continue
            if status!='added':raise ValueError('Unreviewed frozen mapping status')
            index=m['candidate_index'];frame=old[index]
            if (frame['source_pts'],frame['source_frame_index'])!=(m['source_pts'],m['source_frame_index']):
                raise ValueError('Frozen source identity mismatch')
            t=source_time(frame,pool['video'])
            if not Fraction(region['left_fraction'])<=t<=Fraction(region['right_fraction']):
                checks.append(dict(target_fraction=str(target),status='outside_new_window',original_index=index));continue
            if frame['source_pts'] in seen:
                checks.append(dict(target_fraction=str(target),status='duplicate_source',original_index=index));continue
            new_index=len(candidates)
            candidates.append(dict(frame,candidate_index=new_index,region_id=region['region_id'],
                requested_fraction=str(target),requested_seconds=float(target)))
            indices.append(index);seen.add(frame['source_pts']);kept.append(new_index)
            checks.append(dict(target_fraction=str(target),status='reused',original_index=index,candidate_index=new_index))
        records.append(dict(region_id=region['region_id'],enabled=region['refine'],targets=list(map(str,targets)),mappings=checks,new_indices=kept))
    # 步骤3：新编号必须连续，源身份仍与旧池对应；特征只按索引取行，不能近似插值。
    validate_candidates(candidates,pool['video'])
    result=dict(video=pool['video'],candidates=candidates,initial_count=n,initial_scores=list(pool['initial_scores']),
        plan=plan,refinement=records,new_candidate_count=len(candidates)-n,new_frames_have_itm_scores=False,
        original_candidate_indices=indices,asset_policy='Exact frozen M32 source mappings and feature rows; no new decode or model call')
    return result,f[indices].copy()
