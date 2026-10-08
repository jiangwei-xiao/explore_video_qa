"""exp27：触发核心独立决定补查范围，选择组织与exp07一致。"""
from fractions import Fraction
import numpy as np
from videoqa_methods.budget_contract import budget_config
from videoqa_methods.density_selection import build_regions, source_time, refinement_targets, validate_candidates


def refinement_windows(plan, video, frame_budget):
    """接口：冻结核心→触发窗口并集；只决定新增采样，不改选择单元或核心统计。"""
    cfg=budget_config(frame_budget);end=Fraction(video['duration_fraction']);windows=[]
    # 步骤1：沿用已有扩窗及3种子条件，每个触发核心先产生自己的闭窗口。
    for core in plan['cores']:
        if core['seed_count']<cfg['refinement_min_core_seeds']:continue
        left=max(Fraction(0),Fraction(core['left_fraction']))
        right=min(end,Fraction(core['right_fraction']))
        if left>right:raise ValueError('Invalid triggering core window')
        windows.append(dict(left=left,right=right,core_ids=[core['core_id']]))
    # 步骤2：只合并触发窗口，共享端点也合并；稀疏核心不继承补查。
    merged=[]
    for item in sorted(windows,key=lambda w:(w['left'],w['right'])):
        if merged and item['left']<=merged[-1]['right']:
            merged[-1]['right']=max(merged[-1]['right'],item['right'])
            merged[-1]['core_ids'].extend(item['core_ids'])
        else:merged.append(dict(item))
    return [dict(scope_id=i,left_fraction=str(w['left']),right_fraction=str(w['right']),
                 core_ids=w['core_ids']) for i,w in enumerate(merged)]


def restrict_refinement_pool(pool, features, frame_budget):
    """接口：冻结粗细池/特征→合法子池、同序特征、完整映射；不读取视频或模型。

    粗帧全部保留。新细帧同时满足目标和源PTS落在同一补查窗口；
    原选择计划逐字段不变，候选编号重新连续排列，跨池身份使用源帧号与PTS。
    """
    old=pool['candidates'];n=pool['initial_count'];f=np.asarray(features)
    if len(f)!=len(old) or not 1<=n<=len(old):raise ValueError('Invalid parent feature rows')
    cfg=budget_config(frame_budget)
    if build_regions(old[:n],pool['initial_scores'],pool['video'],cfg)!=pool['plan']:
        raise ValueError('Parent pool is not a single-budget plan')
    # 步骤1：建立原冻结目标映射，兼容M32 added与M8 reused记录；不猜测缺失来源。
    bank={}
    for region in pool['refinement']:
        for item in region['mappings']:
            target=Fraction(item['target_fraction'])
            if target in bank and bank[target]!=item:raise ValueError('Conflicting source mappings')
            bank[target]=item
    globals_=pool.get('original_candidate_indices',list(range(len(old))))
    if len(globals_)!=len(old) or len(set(globals_))!=len(old):raise ValueError('Invalid parent global mapping')
    inverse={v:i for i,v in enumerate(globals_)}
    windows=refinement_windows(pool['plan'],pool['video'],frame_budget)
    candidates=[dict(c) for c in old[:n]];kept=list(range(n));seen={c['source_pts'] for c in candidates};records=[]
    # 步骤2：使用原对齐网格逐目标筛回；实际源帧越过新边界时记录丢弃。
    for window in windows:
        left,right=Fraction(window['left_fraction']),Fraction(window['right_fraction'])
        checks=[];added=[]
        for target in refinement_targets(window,pool['video'],cfg):
            if target not in bank:raise ValueError('Missing frozen target; new inference forbidden')
            item=bank[target];status=item['status'];entry=dict(target_fraction=str(target),parent_status=status)
            if status in ('no_source_in_window','frozen_no_source','outside_window','outside_new_window'):
                entry['status']='parent_unavailable';checks.append(entry);continue
            if status not in ('added','reused','duplicate_source'):
                raise ValueError('Unknown frozen mapping status')
            index=item.get('candidate_index')
            if index is None:index=inverse.get(item.get('original_index'))
            if index is None or not 0<=index<len(old):raise ValueError('Source mapping missing from parent pool')
            frame=old[index]
            if 'source_pts' in item and (item['source_pts'],item['source_frame_index'])!=(frame['source_pts'],frame['source_frame_index']):
                raise ValueError('Frozen mapping source identity differs')
            instant=source_time(frame,pool['video']);entry['parent_candidate_index']=index
            if not left<=instant<=right:entry['status']='outside_new_scope'
            elif frame['source_pts'] in seen:entry['status']='duplicate_source'
            else:
                # 步骤3：保留旧特征行，不重算ITM/视觉特征；选择归属仍使用原区域ID。
                cid=len(candidates)
                candidates.append(dict(frame,candidate_index=cid,requested_fraction=str(target),requested_seconds=float(target)))
                kept.append(index);seen.add(frame['source_pts']);added.append(cid)
                entry.update(status='retained',candidate_index=cid)
            checks.append(entry)
        records.append(dict(**window,targets=[str(t) for t in refinement_targets(window,pool['video'],cfg)],
                            mappings=checks,new_indices=added))
    validate_candidates(candidates,pool['video'])
    result=dict(video=pool['video'],candidates=candidates,initial_count=n,
                initial_scores=list(pool['initial_scores']),plan=pool['plan'],refinement=records,
                parent_candidate_indices=kept,original_candidate_indices=[globals_[i] for i in kept],
                new_candidate_count=len(candidates)-n,new_frames_have_itm_scores=False,
                refinement_scope_policy='triggering original-core window union; original selection organization')
    trace=dict(windows=windows,parent_fine_count=len(old)-n,fine_count=len(candidates)-n,
               retained_parent_indices=kept,removed_parent_indices=sorted(set(range(n,len(old)))-set(kept)))
    return result,f[kept].copy(),trace
