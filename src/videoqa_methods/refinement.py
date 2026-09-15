"""Fine sampling recovers absolute source indices from an exact coarse-frame anchor."""
import math
import time
from fractions import Fraction
from videoqa_runtime.baseline_selection import ProtocolError


def fine_targets(hotspot, video):
    """接口：在裁剪后的热点窗口构造全局四分之一秒网格，排除原1 FPS网格及视频终点之外的时间。"""
    left, right = Fraction(hotspot['left_fraction']), Fraction(hotspot['right_fraction'])
    duration = Fraction(str(video['duration_seconds']))
    return [Fraction(k, 4) for k in range(math.ceil(left * 4), math.floor(right * 4) + 1)
            if k % 4 != 1 and Fraction(k, 4) < duration]


def decode_hotspots(video, candidates, hotspots, cfg):
    """核心接口：输入视频、粗候选和热点，返回去重的新候选、RGB图像、逐窗记录和计时。以精确粗帧锚点恢复绝对源帧号，禁止按平均FPS推算。"""
    import av
    tb, origin = Fraction(video['time_base']), video['start_pts']
    existing = {r['source_pts'] for r in candidates}
    anchors = {r['source_pts']: r['source_frame_index'] for r in candidates}
    additions, images_by_pts, reports = [], {}, []
    rgb_seconds = 0.0
    started = time.perf_counter()
    # 步骤1：逐窗生成新增网格，并找到窗口左侧已知绝对帧号的粗候选锚点。
    for number, h in enumerate(hotspots):
        left, right = Fraction(h['left_fraction']), Fraction(h['right_fraction'])
        targets = fine_targets(h, video)
        before = [r for r in candidates if (r['source_pts'] - origin) * tb <= left]
        anchor = before[-1] if before else None
        seen = set(existing)
        found, drops, cursor = [], [], 0
        with av.open(video['path']) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            index, anchored = -1, anchor is None
            if anchor is not None:
                container.seek(anchor['source_pts'], stream=stream, backward=True)
            previous = None
            # 步骤2：先精确对齐锚点PTS，再逐帧计数；窗口靠近起点时直接从文件开头计数。
            for frame in container.decode(stream):
                pts = frame.pts
                if pts is None or (previous is not None and pts <= previous):
                    raise ProtocolError('Invalid PTS during local decoding')
                previous = pts
                if not anchored:
                    if pts < anchor['source_pts']:
                        continue
                    if pts != anchor['source_pts']:
                        raise ProtocolError('Local decoder missed its exact coarse anchor')
                    index, anchored = anchor['source_frame_index'], True
                else:
                    index += 1
                if pts in anchors and anchors[pts] != index:
                    raise ProtocolError('Local and full-stream source frame numbers disagree')
                instant = (pts - origin) * tb
                if instant > right:
                    break
                # 步骤3：映射到不早于网格点的第一源帧，剔除越界及与粗候选重复的帧。
                while cursor < len(targets) and instant >= targets[cursor]:
                    target = targets[cursor]
                    cursor += 1
                    if instant < left or instant >= Fraction(str(video['duration_seconds'])):
                        drops.append(dict(target_seconds=float(target), reason='outside_window'))
                        continue
                    if pts in seen:
                        drops.append(dict(target_seconds=float(target), reason='duplicate_source'))
                        continue
                    seen.add(pts)
                    row = dict(source_frame_index=index, source_pts=pts, source_seconds=float(pts * tb),
                               timestamp_seconds=float(instant), requested_seconds=float(target),
                               hotspot_index=number, hotspot_source_pts=h.get('source_pts'),
                               hotspot_segment_id=h.get('segment_id'), requested_fraction=str(target))
                    t = time.perf_counter()
                    image = frame.to_image()
                    rgb_seconds += time.perf_counter() - t
                    found.append((row, image))
                if cursor == len(targets):
                    break
            if not anchored:
                raise ProtocolError('No coarse anchor found during local decoding')
        for target in targets[cursor:]:
            drops.append(dict(target_seconds=float(target), reason='no_valid_source_in_window'))
        # 步骤4：少数变帧率偏移会令闭区间包含13个非粗网格点。
        # 源帧去重后按距热点近优先、同距取更早PTS，固定截到最多12个，不补足被剔除帧。
        center = Fraction(h['center_fraction'])
        found.sort(key=lambda x: (abs((x[0]['source_pts'] - origin) * tb - center), x[0]['source_pts']))
        for row, _ in found[cfg['maximum_new_frames_per_hotspot']:]:
            drops.append(dict(source_pts=row['source_pts'], reason='per_hotspot_cap'))
        retained = found[:cfg['maximum_new_frames_per_hotspot']]
        for row, image in retained:
            if row['source_pts'] in existing:
                raise ProtocolError('Hotspot ranges produced duplicate source frames')
            existing.add(row['source_pts'])
            additions.append(row)
            images_by_pts[row['source_pts']] = image
        reports.append(dict(hotspot_index=number, hotspot_source_pts=h.get('source_pts'),
                            hotspot_segment_id=h.get('segment_id'), targets=[float(t) for t in targets],
                            anchor=None if anchor is None else dict(source_pts=anchor['source_pts'], source_frame_index=anchor['source_frame_index']),
                            retained=len(retained), dropped=drops))
    # 步骤5：跨窗口按源时间排序，接在原候选编号之后，最后核对每题最多24帧。
    additions.sort(key=lambda r: r['source_pts'])
    for i, row in enumerate(additions, len(candidates)):
        row['candidate_index'] = i
    if len(additions) > cfg['maximum_new_frames']:
        raise ProtocolError('Exceeded the global refinement candidate cap')
    return additions, [images_by_pts[r['source_pts']] for r in additions], reports, dict(
        fine_decode_seconds=time.perf_counter() - started - rgb_seconds, fine_rgb_seconds=rgb_seconds)
