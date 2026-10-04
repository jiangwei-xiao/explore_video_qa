"""均匀与冻结RD输入的等权时间分布融合，不读取问题答案或重新评分。"""
from fractions import Fraction
from math import lcm


def select_equal_temporal_mixture(uniform,rd,frame_budget=32):
    """接口：两个有序源帧集各占1/2质量，以有序唯一帧逼近等质量分位点。

    这是固定工程假设，不是语义充分性目标；不强制保留原RD锚点。
    动态规划仅解决重复源帧和分位点匹配，不能解释成QA最优。
    """
    if frame_budget<1 or not uniform['selected_frames'] or not rd['selected_frames']:
        raise ValueError('预算或输入为空')
    for key in ['path','time_base','start_pts','duration_fraction']:
        if uniform['video'][key]!=rd['video'][key]:raise ValueError('视频身份不一致')
    # 步骤1：按真实源身份合并，重合帧获得两路质量，但仍只占一个输入位置。
    merged={};pts_to_index={};index_to_pts={}
    counts=[len(x['selected_frames']) for x in [uniform,rd]]
    denominator=lcm(*(2*n for n in counts))
    for name,selection,n in zip(['BASE-Uniform','RD-1.2-exp12'],[uniform,rd],counts):
        frames=selection['selected_frames']
        keys=[(f['source_pts'],f['source_frame_index']) for f in frames]
        if keys!=sorted(set(keys)):raise ValueError('来源帧不唯一或无序')
        for f in frames:
            pts,idx=f['source_pts'],f['source_frame_index']
            if pts in pts_to_index and pts_to_index[pts]!=idx:raise ValueError('同PTS源编号冲突')
            if idx in index_to_pts and index_to_pts[idx]!=pts:raise ValueError('同源编号PTS冲突')
            index_to_pts[idx]=pts
            pts_to_index[pts]=idx;key=(pts,idx)
            if key not in merged:merged[key]=dict(frame=dict(f),mass=0,sources=[])
            merged[key]['mass']+=denominator//(2*n)
            merged[key]['sources'].append(name)
    items=[merged[k] for k in sorted(merged)]
    if sum(x['mass'] for x in items)!=denominator:raise ValueError('混合质量不闭合')
    target=min(frame_budget,len(items));prefix=0;centers=[]
    for ordinal,item in enumerate(items):
        item['frame']['candidate_index']=ordinal
        centers.append(2*prefix+item['mass']);prefix+=item['mass']
    # 步骤2：最小化CDF中心与(j+1/2)/K的L1距离；统一分母后用整数精确比较。
    states={0:(0,())}
    for i,center in enumerate(centers):
        for count in range(min(target,i+1),0,-1):
            if count-1 not in states:continue
            cost,chosen=states[count-1]
            proposal=(cost+abs(center*target-(2*count-1)*denominator),chosen+(i,))
            if count not in states or proposal<states[count]:states[count]=proposal
    cost,indices=states[target]
    # 步骤3：按时间输出唯一源帧，记录来源；不重用历史答案或宣称输入信息必然更好。
    selected=[items[i]['frame'] for i in indices]
    trace=dict(rule='equal source measures; unique ordered midpoint-quantile L1 matching',
        mass_denominator=denominator,union=items,selected_indices=list(indices),
        exact_cost=str(Fraction(cost,2*denominator*target)),actual_budget=target,
        source_counts={name:sum(name in items[i]['sources'] for i in indices) for name in ['BASE-Uniform','RD-1.2-exp12']},
        shared_selected=sum(len(items[i]['sources'])==2 for i in indices))
    return dict(video=uniform['video'],selected_frames=selected),trace
