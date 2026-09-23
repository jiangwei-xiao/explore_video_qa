"""全量任务的持久化、身份校验与严格一次生成账本。"""
import datetime
import fcntl
import hashlib
import json
import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from videoqa_runtime.common import ROOT, read_json, sha256, write_json

METHODS = ('BASE-Uniform', 'BASE-BLIP-TopK', 'RD-1.2')
MODELS = ('llava', 'qwen25vl')
MANIFEST = 'data/manifests/video_mme_full_2700.json'
MANIFEST_HASH = 'eca56f202afcf3cf7e3e3d608be6f95ec1980a7967c117ff379832ce17f3bdd0'
QWEN_PATH = '/home/models/Qwen/Qwen2.5-VL-7B-Instruct'
PYTHONS = {m: '/App/conda/envs/' + e + '/bin/python' for m, e in
           [('llava', 'explore_video_qa'), ('qwen25vl', 'explore_video_qa_qwen25vl')]}


def utc():
    """接口：审计使用UTC，耗时使用单调时钟。"""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def durable(path, value):
    """接口：原子JSON落盘后同步文件与目录，先记账再调用模型。"""
    write_json(path, value)
    with Path(path).open('rb') as stream:
        os.fsync(stream.fileno())
    fd = os.open(str(Path(path).parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def digest(value):
    """以规范JSON计算内容指纹，不依赖展示缩进。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


@contextmanager
def exclusive(path):
    """接口：同阶段/同题的并发执行必须互斥，避免重复生成。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def rows():
    """核对冻结2700题、900视频及每视频三题，不重新抽样。"""
    from collections import Counter
    if sha256(ROOT / MANIFEST) != MANIFEST_HASH:
        raise ValueError('Frozen full manifest changed')
    data = read_json(ROOT / MANIFEST)['rows']
    assert len(data) == len({r['question_id'] for r in data}) == 2700
    assert Counter(r['stratum'] for r in data) == {'short': 900, 'medium': 900, 'long': 900}
    counts = Counter(r['video_id'] for r in data)
    assert len(counts) == 900 and set(counts.values()) == {3}
    return data


def pilot_ids(data):
    """固定首5题及两个短视频边界题，不按预测或正确率挑选。"""
    ids = list(dict.fromkeys([r['question_id'] for r in data[:5]] + ['133-1', '134-1']))
    assert len(ids) == 7
    return ids


def stage_methods(model):
    return ('RD-1.2',) if model == 'llava' else METHODS


def method_order(model, ordinal):
    """同题同GPU，三种方法按冻结题序循环轮换。"""
    methods = stage_methods(model)
    shift = ordinal % len(methods)
    return methods[shift:] + methods[:shift]


def invoke_once(stage, qid, method, fingerprint, gpu, function):
    """接口：固定题号×方法调用ID；任何未确认状态均禁止再次调用。"""
    path = Path(stage) / 'calls' / method / (qid + '.json')
    # 步骤1：完成账本复用返回值；未完成调用不能凭观察超时再次执行。
    if path.exists():
        record = read_json(path)
        if record['protocol_sha256'] != fingerprint or record['status'] != 'completed':
            raise RuntimeError('Uncertain generation; repetition forbidden: ' + str(path))
        return record['answer'], True, 0.0
    record = dict(question_id=qid, method=method, model=Path(stage).name,
                  call_id=f'{Path(stage).name}:{method}:{qid}', protocol_sha256=fingerprint,
                  status='started_may_generate', gpu=gpu, pid=os.getpid(), started_at=utc())
    started = time.perf_counter()
    durable(path, record)
    checkpoint_seconds = time.perf_counter() - started
    state = {}
    # 步骤2：调用失败也占用该ID；不因解析失败或答错重试。
    try:
        answer = function(state)
        assert state == {'started': True, 'returned': True}
        record.update(status='completed', answer=answer, generation_state=state, completed_at=utc())
    except BaseException as exc:
        record.update(status='failed_or_uncertain', generation_state=state,
                      error=repr(exc), traceback=traceback.format_exc(), failed_at=utc())
        durable(path, record)
        raise
    # 步骤3：答案与完整生成状态一同落盘，后续恢复只重建结果。
    durable(path, record)
    return answer, False, checkpoint_seconds


def frame_identity(selection):
    """跨模型仅共享源身份，不把来源模型答案传给新后端。"""
    return [(r['source_pts'], r['source_frame_index']) for r in selection['selected_frames']]


def validate_frames(selection):
    """真实时间、有界唯一源帧和展示时间一一对应。"""
    from fractions import Fraction
    video, frames = selection['video'], selection['selected_frames']
    assert 1 <= len(frames) <= 16
    ids = frame_identity(selection)
    assert len({i[0] for i in ids}) == len({i[1] for i in ids}) == len(ids)
    assert ids == sorted(ids)
    for f in frames:
        t = float((int(f['source_pts']) - int(video['start_pts'])) * Fraction(video['time_base']))
        assert 0 <= t < video['duration_seconds'] and abs(t - f['timestamp_seconds']) < 1e-8


def save_array(path, value):
    """特征原子保存；指纹同时绑定到选帧检查点。"""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256(path)
