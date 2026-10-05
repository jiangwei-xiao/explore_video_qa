"""编码单位感知的离线候选：16个代表位置各配一个直接邻帧，输出仍为32帧。"""
from fractions import Fraction
import numpy as np
from videoqa_methods.density_selection import frame_key,source_time
from videoqa_methods.coarse_eligibility import coarse_selection_cells
from videoqa_methods.optional_region_selection import select_optional_regions


def pair_representatives(candidates,features,representatives,owners):
    """接口：在同选择单元的前/后直接邻帧中求最大总余弦的不重叠配对。

    不接收题目、答案或人工见证。邻帧不能也是代表帧；时间排序后每对必须
    对应同一模型时间patch，不能用两组交错帧冒充配对。无解返回None并说明原因。
    """
    f=np.asarray(features,dtype=np.float32);n=len(candidates);reps=set(representatives)
    if f.ndim!=2 or len(f)!=n or len(owners)!=n or not np.isfinite(f).all():raise ValueError('Invalid feature or owner shape')
    if not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5,rtol=0):raise ValueError('Expected normalized features')
    if len(reps)!=len(representatives) or not reps or min(reps)<0 or max(reps)>=n:raise ValueError('Invalid representatives')
    # 步骤1：按真实源时间建立邻接关系，不依赖粗帧/细帧在数组中的存储顺序。
    order=sorted(range(n),key=lambda i:frame_key(candidates[i]));position={i:j for j,i in enumerate(order)}
    sequence=sorted(reps,key=lambda i:position[i]);choices=[]
    for anchor in sequence:
        p=position[anchor];edges=[]
        for neighbour in [p-1,p+1]:
            if not 0<=neighbour<n:continue
            j=order[neighbour]
            if j in reps or owners[j]!=owners[anchor]:continue
            lo,hi=sorted([p,neighbour]);coherence=float(np.clip(f[anchor]@f[j],-1,1))
            edges.append((lo,hi,coherence))
        if not edges:return None,dict(reason='representative_has_no_free_same_region_neighbour',anchor=anchor)
        choices.append(edges)
    # 步骤2：动态规划只解决相邻配对冲突；不借此重新排序代表位置或调整预算。
    states={-1:(0.,())}
    for edges in choices:
        updated={}
        for previous,(score,pairs) in states.items():
            for lo,hi,value in edges:
                if lo<=previous:continue
                proposed=(score+value,pairs+((lo,hi),))
                incumbent=updated.get(hi)
                if incumbent is None or proposed[0]>incumbent[0] or (proposed[0]==incumbent[0] and proposed[1]<incumbent[1]):
                    updated[hi]=proposed
        if not updated:return None,dict(reason='overlapping_neighbour_pairs_have_no_solution')
        states=updated
    score,positions=min(states.values(),key=lambda p:(-p[0],p[1]))
    pairs=[(order[a],order[b]) for a,b in positions];flat=[i for pair in pairs for i in pair]
    # 步骤3：核验真实最终顺序与配对边界，禁止重复源帧、跨区域或丢代表帧。
    if flat!=sorted(flat,key=lambda i:frame_key(candidates[i])) or len(set(flat))!=2*len(reps) or not reps<=set(flat):
        raise ValueError('Pairing invariant violated')
    if any(len(set(pair)&reps)!=1 or owners[pair[0]]!=owners[pair[1]] for pair in pairs):
        raise ValueError('Pair contains wrong representatives or regions')
    return flat,dict(pairs=[list(pair) for pair in pairs],coherence_sum=score,
                     rule='one representative plus immediate temporal neighbour; ordered nonoverlapping pairs')


def select_temporal_pairs(pool,features,original_indices,budget=32):
    """接口：冻结M32池、原覆盖目标与局部规则，代表预算K/2；无法配对则整题回退原exp12。

    候选数量不足预算时保持原输入，不额外凑源帧；不改变Qwen编码器、Token或FPS。
    代表配额允许0，正配额仍以原区域锚点开始；最终每个代表占用两个源帧名额。
    """
    if budget!=32:raise ValueError('This bounded prototype only defines K32')
    n=len(pool['candidates'])
    if len(original_indices)!=min(budget,n) or len(set(original_indices))!=len(original_indices):
        raise ValueError('Fallback budget or uniqueness differs')
    if any(not 0<=i<n for i in original_indices) or list(original_indices)!=sorted(original_indices,key=lambda i:frame_key(pool['candidates'][i])):
        raise ValueError('Fallback source identity or order invalid')
    if len(pool['candidates'])<=budget:
        return list(original_indices),dict(fallback=True,reason='short_or_full_pool_unchanged')
    if len(original_indices)!=budget:raise ValueError('Original K32 input is required')
    # 步骤1：复用已验证的可选区域DP，减少的是代表预算，不是定位种子数量。
    representatives,trace=select_optional_regions(pool,features,frame_budget=budget//2)
    cells=coarse_selection_cells(pool['plan'],pool['video']);owners=[]
    for c in pool['candidates']:
        t=source_time(c,pool['video'])
        matches=[r['region_id'] for r in cells['regions'] if Fraction(r['left_fraction'])<=t<=Fraction(r['right_fraction'])]
        if len(matches)!=1:raise ValueError('Candidate owner is not unique')
        owners.append(matches[0])
    # 步骤2：以物理时间邻接与视觉兼容性补齐，不用人工线索挑选搭配。
    result,pair_trace=pair_representatives(pool['candidates'],features,representatives,owners)
    if result is None:
        return list(original_indices),dict(fallback=True,reason=pair_trace,representatives=representatives,
                                          representative_trace=trace)
    if len(result)!=budget:raise ValueError('Final source budget changed')
    # 步骤3：代表配额与实际源帧配额分别保存，不把16个位置称为32个独立位置。
    return result,dict(fallback=False,representatives=representatives,representative_trace=trace,
                       pair_trace=pair_trace,source_budget=budget,representative_budget=budget//2,
                       limitation='邻近且视觉相近不保证语义有效；覆盖广度可能下降')
