"""R：保护Top-K连续区域的首尾/峰值，仅用少量边界邻域替换内部帧。"""
from fractions import Fraction
import time
from videoqa_runtime.baseline_selection import select_topk, ProtocolError
from .refinement import decode_hotspots

CONFIG = dict(version='topk-region-boundary-v1', frame_budget=16, maximum_replacements=4,
              connection_seconds=1, halo_seconds=1, refine_fps=4,
              maximum_new_frames_per_hotspot=3, maximum_new_frames=24)


def make_replacement_plan(candidates, scores, video):
    """接口：仅由候选/原问题分数生成保护、替换和取证窗口计划；不读取题目答案或类型。"""
    # 步骤1：固定Top-K主体；短候选池直接保留全部，不做补查。
    if [r['candidate_index'] for r in candidates] != list(range(len(candidates))):
        raise ProtocolError('Expected contiguous initial candidate indices')
    if len({r['source_pts'] for r in candidates}) != len(candidates):
        raise ProtocolError('Initial source PTS must be unique')
    top = [r['candidate_index'] for r in select_topk(candidates, scores, CONFIG['frame_budget'])]
    result = dict(topk_indices=top, effective_budget=len(top), regions=[], protected_indices=[],
                  potential_removals=[], windows=[], short_pool=len(candidates)<CONFIG['frame_budget'])
    if result['short_pool']:
        result['protected_indices'] = top
        return result
    regions = []
    for i in top:
        if (not regions or Fraction(str(candidates[i]['requested_seconds'])) -
            Fraction(str(candidates[regions[-1][-1]]['requested_seconds'])) > CONFIG['connection_seconds']):
            regions.append([])
        regions[-1].append(i)
    # 步骤2：首帧、末帧与最高分帧受保护；从其他内部帧全局选最多4个替换位置。
    protected, eligible = set(), []
    for rid, ids in enumerate(regions):
        peak = min(ids, key=lambda i:(-scores[i],candidates[i]['source_pts'],i))
        keep = {ids[0],ids[-1],peak};protected.update(keep)
        remaining = [i for i in ids if i not in keep];eligible.extend(remaining)
        result['regions'].append(dict(region_id=rid, indices=ids, peak_index=peak,
            peak_score=scores[peak], protected_indices=sorted(keep), eligible_indices=remaining))
    order = lambda i:(scores[i],-candidates[i]['source_pts'],-i)
    removals = sorted(eligible,key=order)[:CONFIG['maximum_replacements']]
    result.update(protected_indices=sorted(protected),potential_removals=removals)
    # 步骤3：按区域最高分排序处理，窗口边界依据真实源PTS，以有理数避免浮点同分漂移。
    tb,origin = Fraction(video['time_base']),video['start_pts'];duration=Fraction(str(video['duration_seconds']))
    active = sorted([g for g in result['regions'] if set(g['indices'])&set(removals)],
                    key=lambda g:(-g['peak_score'],candidates[g['indices'][0]]['source_pts'],g['indices'][0]))
    result['region_execution_order'] = [g['region_id'] for g in active]
    for g in active:
        g['replacement_slots'] = sorted(set(g['indices'])&set(removals),key=order)
        for side,i in [('before',g['indices'][0]),('after',g['indices'][-1])]:
            boundary=(candidates[i]['source_pts']-origin)*tb
            left=max(Fraction(0),boundary-1) if side=='before' else boundary
            right=boundary if side=='before' else min(duration,boundary+1)
            if right<=left:continue
            result['windows'].append(dict(region_id=g['region_id'],side=side,source_pts=candidates[i]['source_pts'],
                left_fraction=str(left),right_fraction=str(right),center_fraction=str(boundary)))
    return result


def select_from_neighborhoods(candidates, scores, video, plan, additions):
    """纯选择接口：按距边界距离补位；每成功加入一帧才移除对应区域的一个最低分内部帧。"""
    rows=list(candidates)+list(additions)
    if [r['candidate_index'] for r in rows] != list(range(len(rows))) or len({r['source_pts'] for r in rows}) != len(rows):
        raise ProtocolError('New candidates must extend indices with unique PTS')
    if len(additions)>CONFIG['maximum_new_frames']:raise ProtocolError('New candidate cap exceeded')
    tb,origin=Fraction(video['time_base']),video['start_pts']
    selected=set(plan['topk_indices']);swaps=[];traces=[]
    # 步骤1：只在本区域边界外窗口寻找候选，保留已有初始候选，新帧无需BLIP评分。
    for rid in plan.get('region_execution_order',[]):
        region=next(g for g in plan['regions'] if g['region_id']==rid)
        eligible={}
        for window in [w for w in plan['windows'] if w['region_id']==rid]:
            left,right,boundary=(Fraction(window[k]) for k in ('left_fraction','right_fraction','center_fraction'))
            for i,row in enumerate(rows):
                instant=(row['source_pts']-origin)*tb
                outside=instant<boundary if window['side']=='before' else instant>boundary
                if left<=instant<=right and outside:
                    distance=abs(instant-boundary)
                    if i not in eligible or distance<eligible[i]:eligible[i]=distance
        ranking=sorted(eligible,key=lambda i:(eligible[i],rows[i]['source_pts'],rows[i]['candidate_index']))
        added=[]
        # 步骤2：名额不能跨区域借用；空邻域保留原输入，不填重复帧。
        for old in region['replacement_slots']:
            new=next((i for i in ranking if i not in selected),None)
            if new is None:break
            selected.remove(old);selected.add(new);added.append(new)
            swaps.append(dict(region_id=rid,removed_index=old,added_index=new,
                              boundary_distance_fraction=str(eligible[new]),is_new_source=new>=len(candidates)))
        traces.append(dict(region_id=rid,eligible_indices=ranking,added_indices=added,
                           distances={str(i):str(eligible[i]) for i in ranking}))
    # 步骤3：检查保护、预算、唯一性，恢复真实时间顺序。
    indices=sorted(selected,key=lambda i:(rows[i]['source_pts'],rows[i]['candidate_index']))
    if len(indices)!=plan['effective_budget'] or not set(plan['protected_indices'])<=selected:
        raise ProtocolError('Region protection or final budget violated')
    if len(swaps)>4:raise ProtocolError('Replacement cap exceeded')
    return dict(video=video,candidates=rows,initial_scores=list(scores),initial_count=len(candidates),
                new_candidate_count=len(additions),plan=plan,replacements=swaps,region_trace=traces,
                selected_indices=indices,selected_frames=[rows[i] for i in indices],
                new_candidate_scoring='not scored; selection uses boundary distance only')


def read_neighborhoods_and_select(pool):
    """接口：生成固定计划、精确解码邻域并完成R选择；返回记录、最终选帧和分阶段计时。"""
    start=time.perf_counter()
    plan=make_replacement_plan(pool['candidates'],pool['scores'],pool['video'])
    times={'region_plan_seconds':time.perf_counter()-start}
    # 步骤1：复用已有锚点解码；仅参数为每侧3帧/总24帧，不改旧C实现。
    if plan['windows']:
        additions,images,decode_trace,decode_times=decode_hotspots(pool['video'],pool['candidates'],plan['windows'],CONFIG)
        del images
    else:
        additions,decode_trace,decode_times=[],[],dict(fine_decode_seconds=0.,fine_rgb_seconds=0.)
    times.update(decode_times)
    # 步骤2：冻结候选身份后补位，新帧不会被相关性分数重新筛掉。
    start=time.perf_counter()
    result=select_from_neighborhoods(pool['candidates'],pool['scores'],pool['video'],plan,additions)
    result['fine_decode_trace']=decode_trace
    times['boundary_selection_seconds']=time.perf_counter()-start
    return result,times


def read_exact_frames(video,selected):
    """按精确PTS读取最终帧，保持双线程解码；不按平均FPS推算，也不使用应用缓存。"""
    import av
    pts=[r['source_pts'] for r in selected]
    if len(set(pts))!=len(pts) or pts!=sorted(pts):raise ProtocolError('Selected PTS must be unique and sorted')
    images=[]
    for wanted in pts:
        with av.open(video['path']) as container:
            stream=container.streams.video[0];stream.codec_context.thread_count=2
            container.seek(wanted,stream=stream,backward=True)
            for frame in container.decode(stream):
                if frame.pts==wanted:images.append(frame.to_image());break
                if frame.pts is not None and frame.pts>wanted:raise ProtocolError('Exact selected PTS missed')
            else:raise ProtocolError('Selected source frame missing')
    return images
