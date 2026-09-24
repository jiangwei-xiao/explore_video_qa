"""显式预算的源帧接口；源身份与模型内部时间补齐分开记录。"""
from fractions import Fraction
import hashlib
import numpy as np


def padded_count(count, budget=32):
    """接口：最多budget个唯一源帧，奇数仅在模型内部复制末帧。"""
    if not 1 <= count <= budget:
        raise ValueError('Source frame budget exceeded')
    return count + count % 2


def validate_frames(selection, budget=32):
    """接口：校验唯一源帧、真实PTS与有界预算，不依赖答案或题型。"""
    # 步骤1：源编号与PTS均须唯一，并保持时间顺序。
    video, frames = selection['video'], selection['selected_frames']
    padded_count(len(frames), budget)
    ids = [(f['source_pts'], f['source_frame_index']) for f in frames]
    assert ids == sorted(ids)
    assert len({p for p, _ in ids}) == len({i for _, i in ids}) == len(ids)
    # 步骤2：使用有理数时间校验，不按平均FPS反推。
    for frame in frames:
        instant = (frame['source_pts'] - video['start_pts']) * Fraction(video['time_base'])
        assert 0 <= instant < Fraction(video['duration_fraction'])
        assert abs(float(instant) - frame['timestamp_seconds']) < 1e-8


def native_frames(pts, video, budget=32):
    """接口：在完整源编号域进行参考均匀采样，不经过1 FPS池。"""
    # 步骤1：截断取整与参考加载器一致，源不足时不重复。
    indices = np.linspace(0, len(pts)-1, min(budget, len(pts)), dtype=int)
    frames = []
    for ordinal, index in enumerate(indices):
        p = int(pts[index]); t = (p-video['start_pts'])*Fraction(video['time_base'])
        frames.append(dict(candidate_index=ordinal, source_frame_index=int(index), source_pts=p,
            source_seconds=float(p*Fraction(video['time_base'])), timestamp_seconds=float(t), requested_seconds=float(t)))
    validate_frames(dict(video=video, selected_frames=frames), budget)
    return frames


class FrameProvider:
    """接口：32帧独立提供器，精确读取指定源帧，无隐式重采样。"""
    def __init__(self, selection):
        validate_frames(selection); self.selection = selection

    def decode(self):
        """步骤1精确解码，步骤2返回逐帧RGB指纹供实际输入核验。"""
        from videoqa_methods.region_context import read_exact_frames
        images = read_exact_frames(self.selection['video'], self.selection['selected_frames'])
        hashes = [hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest() for im in images]
        return images, hashes
