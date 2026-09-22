"""推广入口人工修复诊断：仅既有粗候选，两遍选择不新增补查或模型调用。"""
import numpy as np
from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_methods.density_selection import build_regions,validate_candidates
from .density_reward_decay import select_density_decay
from .local_distance_scale import select_local_distance

def replay_coarse(candidates,scores,features,video,cfg,excluded):
    """接口：显式人工排除集合→重排种子/区域/配额/粗选；返回原始身份映射及完整轨迹。"""
    validate_candidates(candidates,video)
    if len(scores)!=len(candidates) or len(features)!=len(candidates):raise ProtocolError('Coarse arrays not aligned')
    if len(excluded)!=len(set(excluded)) or any(not isinstance(i,int) or i<0 or i>=len(candidates) for i in excluded):
        raise ProtocolError('Invalid or duplicate exclusion IDs')
    # 步骤1：保留真实源帧与原采样目标；仅内部连续索引重映射，不能压缩时间轴。
    blocked=set(excluded);mapping=[i for i in range(len(candidates)) if i not in blocked]
    if not mapping:raise ProtocolError('No retained candidates')
    rows=[dict(candidates[i],candidate_index=j) for j,i in enumerate(mapping)]
    values=[scores[i] for i in mapping];vectors=np.asarray(features,dtype=np.float32)[mapping]
    # 步骤2：从完整剩余排序补种子，不直接使用人工事实或答案决定新种子。
    plan=build_regions(rows,values,video,cfg)
    _,budget=select_density_decay(rows,vectors,plan,video,cfg)
    selected,trace=select_local_distance(rows,vectors,plan,video,cfg,budget['region_budgets'])
    # 步骤3：结果恢复原编号；refine标记只表示潜在触发，本诊断明确不执行。
    original=[mapping[i] for i in selected]
    if blocked&set(original):raise ProtocolError('Excluded frame reentered selection')
    return dict(excluded_original_indices=list(excluded),local_to_original=mapping,
        plan=plan,budget_trace=budget,selection_trace=trace,
        seed_original_indices=[mapping[i] for i in plan['seed_indices']],selected_original_indices=original,
        selected_frames=[candidates[i] for i in original],
        regions_original=[dict(region_id=r['region_id'],left=r['left_fraction'],right=r['right_fraction'],
            anchor_index=mapping[r['anchor_index']],seed_indices=[mapping[i] for i in r['seed_indices']],
            quota=trace['region_budgets'][str(r['region_id'])],potential_refine=r['refine']) for r in plan['regions']],
        actual_refinement=False,new_candidates=0,new_model_calls=0,
        limitation='Human-targeted coarse-only diagnostic, not RD-2P formal inputs or a deployed detector')
