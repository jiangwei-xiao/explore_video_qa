"""Source-PTS sampling; never infer timestamps from an average frame rate."""
from fractions import Fraction
from pathlib import Path
import time

from .common import sha256


def candidate_rows(timestamps, time_base, start_pts, phase=Fraction(1, 4), fps=Fraction(1)):
    if phase < 0 or fps <= 0:
        raise ValueError('Invalid sampling phase or FPS')
    target = phase
    previous = None
    for source_index, pts in enumerate(timestamps):
        if pts is None or (previous is not None and pts <= previous):
            raise ValueError('Missing or non-increasing source PTS')
        previous = pts
        timestamp = (pts - start_pts) * time_base
        if timestamp < target:
            continue
        yield {'candidate_index': None, 'source_frame_index': source_index,
               'source_pts': pts, 'source_seconds': float(pts * time_base),
               'timestamp_seconds': float(timestamp), 'requested_seconds': float(target)}
        # Low-FPS sources can cross several grid points; keep each source frame once.
        target += (int((timestamp - target) * fps) + 1) / fps


def index_video(video_path):
    import av
    start = time.perf_counter()
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 4
        time_base = Fraction(stream.time_base)
        start_pts = stream.start_time or 0
        if stream.duration is None or stream.duration <= 0:
            raise ValueError('Missing positive video-stream duration')
        duration = float(stream.duration * time_base)
        candidates = list(candidate_rows((frame.pts for frame in container.decode(stream)), time_base, start_pts))
        for i, row in enumerate(candidates):
            row['candidate_index'] = i
        return {'path': str(Path(video_path).resolve()), 'sha256': sha256(video_path),
                'duration_seconds': duration, 'time_base': str(time_base), 'start_pts': start_pts,
                'candidate_fps': 1, 'phase_seconds': 0.25,
                'mapping': 'first source PTS at or after each grid point; deduplicated by source frame',
                'candidates': candidates, 'index_seconds': time.perf_counter() - start}


def uniform_selection(candidates, count=16):
    """均匀选帧：count 为最大预算（2026-09-17 修订），候选不足时取全部候选等间隔选取；
    候选数≥预算时与原固定预算行为完全一致。"""
    import numpy as np
    if count < 1 or not candidates:
        raise ValueError('Frame budget must be positive and candidates non-empty')
    effective = min(count, len(candidates))
    positions = np.rint(np.linspace(0, len(candidates) - 1, effective)).astype(int).tolist()
    return [candidates[i] for i in positions]


def decode_selected(index, selected):
    import av
    pts_list = [row['source_pts'] for row in selected]
    if len(set(pts_list)) != len(pts_list) or pts_list != sorted(pts_list):
        raise ValueError('Selected frames must be unique and in source-time order')
    frames = []
    for pts in pts_list:
        with av.open(index['path']) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            container.seek(pts, stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts == pts:
                    frames.append(frame.to_image())
                    break
                if frame.pts is not None and frame.pts > pts:
                    raise ValueError(f'Seek missed the exact selected source PTS {pts}')
            else:
                raise ValueError(f'Cannot decode selected source PTS {pts}')
    return frames
