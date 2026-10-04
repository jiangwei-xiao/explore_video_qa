"""用冻结完整源PTS索引构造可选区域候选元数据，不解码或计算特征。"""
from fractions import Fraction
import math
import numpy as np
from videoqa_methods.optional_region_selection import build_optional_regions
from videoqa_methods.density_selection import refinement_targets,validate_candidates


def map_optional_pool(old,cfg,source_pts):
    """接口：复用粗候选/分数，返回新区域与2FPS精确源映射；旧细帧不混入初始池。

    source_pts必须是该视频冻结的完整解码序列，数组位置就是真实源帧号。
    无平均FPS推算；取不早于目标的首帧，并在真实窗口内检查、去重。
    """
    pts=np.asarray(source_pts)
    if pts.ndim!=1 or not len(pts) or pts.dtype.kind not in 'iu' or np.any(pts[1:]<=pts[:-1]):
        raise ValueError('完整PTS索引必须为严格递增整数')
    n=old['initial_count'];coarse=[dict(r) for r in old['candidates'][:n]];video=old['video']
    base=Fraction(video['time_base']);origin=video['start_pts']
    # 步骤1：逐个粗帧核对源序号，不能混用另一视频或另一解码序列。
    for row in coarse:
        index=row['source_frame_index']
        if not 0<=index<len(pts) or int(pts[index])!=row['source_pts']:raise ValueError('粗帧与完整索引不符')
    plan=build_optional_regions(coarse,old['initial_scores'],video,cfg)
    candidates=list(coarse);seen={r['source_pts']:r['candidate_index'] for r in coarse};reports=[]
    # 步骤2：在新窗口全局网格上找精确源帧，先检查范围，再排除所有已用源帧。
    for region in plan['regions']:
        targets=refinement_targets(region,video,cfg) if region['refine'] else []
        left,right=Fraction(region['left_fraction']),Fraction(region['right_fraction'])
        report=dict(region_id=region['region_id'],enabled=region['refine'],targets=list(map(str,targets)),mappings=[])
        for target in targets:
            index=int(np.searchsorted(pts,origin+math.ceil(target/base),side='left'))
            mapping=dict(target_fraction=str(target))
            if index>=len(pts):
                mapping['status']='no_source_in_window'
            else:
                p=int(pts[index]);t=(p-origin)*base
                if not left<=t<=right:
                    mapping['status']='no_source_in_window'
                elif p in seen:
                    mapping.update(status='duplicate_source',candidate_index=seen[p],source_pts=p,source_frame_index=index)
                else:
                    cid=len(candidates)
                    row=dict(candidate_index=cid,source_frame_index=index,source_pts=p,source_seconds=float(p*base),
                        timestamp_seconds=float(t),requested_seconds=float(target),requested_fraction=str(target),region_id=region['region_id'])
                    candidates.append(row);seen[p]=cid
                    mapping.update(status='added',candidate_index=cid,source_pts=p,source_frame_index=index)
            report['mappings'].append(mapping)
        reports.append(report)
    # 步骤3：只准备元数据；后续必须按源PTS解码、核对RGB并计算新细帧纯视觉特征。
    validate_candidates(candidates,video)
    return dict(video=video,candidates=candidates,initial_count=n,initial_scores=list(old['initial_scores']),
        plan=plan,refinement=reports,new_candidate_count=len(candidates)-n,new_frames_have_itm_scores=False,
        mapping_policy='Frozen complete source-PTS index, ceil target tick, unique source identity')
