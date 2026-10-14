"""MLVU原问题、原生问答文字和计分字段隔离；七类数据身份冻结。"""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import importlib.util
import time
import numpy as np
from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, save_array, utc

DATA = ROOT / 'data/videos/mlvu/MLVU'
REVISION = '96207eb9aa7101e2a495dd147684a7e618c79e12'
TASKS = ('plotQA', 'needle', 'ego', 'count', 'order', 'anomaly_reco', 'topic_reasoning')
COUNTS = dict(zip(TASKS, (539, 355, 352, 206, 259, 200, 263)))
METHODS = ('BASE-Uniform', 'BASE-BLIP-TopK', 'RD-1.2-exp07', 'RD-1.2-exp12')
PROFILES = dict(llava8=8, qwen32=32)
POST_PROMPT = '\nOnly give the best option.\nBest option: ('
PYTHONS = dict(prepare='/App/conda/envs/explore_video_qa/bin/python',
              llava8='/App/conda/envs/explore_video_qa_lmms/bin/python',
              qwen32='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python')
_TASK = None


def load_task():
    """接口：直接加载固定框架MLVU计分实现，禁止手写另一套宽松解析。"""
    global _TASK
    if _TASK is None:
        from videoqa_lmms_bridge.common import LMMS
        spec = importlib.util.spec_from_file_location('mlvu_frozen_native_task', LMMS / 'lmms_eval/tasks/mlvu/utils.py')
        _TASK = importlib.util.module_from_spec(spec); spec.loader.exec_module(_TASK)
    return _TASK


def task_doc(row):
    """接口：仅框架计分对象带答案；问答文字已与原检索问题分别保存。"""
    return dict(question=row['qa_question'], answer='ABCD'[row['answer_index']],
                task_type=row['task_type'], video_name=row.get('video_path', 'synthetic.mp4'))


def context(row):
    """接口：官方任务文字＋默认后缀；选项只出现一次，不加入时间或答案说明。"""
    return load_task().mlvu_doc_to_text(task_doc(row), dict(pre_prompt='', post_prompt=POST_PROMPT))


def rows(root):
    """接口：恢复按清单摘要核验全部2174题，不重新排序或删歧义题。"""
    root = Path(root); record = read_json(root / 'manifest.json')
    if len(record['rows']) != 2174 or len({r['question_id'] for r in record['rows']}) != 2174:
        raise ValueError('MLVU question denominator differs')
    if Counter(r['task_type'] for r in record['rows']) != COUNTS:
        raise ValueError('MLVU task denominator differs')
    if (root / 'protocol.json').exists() and sha256(root / 'manifest.json') != read_json(root / 'protocol.json')['manifest_sha256']:
        raise ValueError('MLVU manifest changed')
    return record['rows']


def _hash_file(path):
    """接口：独立进程读取原视频字节，不写模型、视频或标注。"""
    return str(path), sha256(path)


def prepare_manifest(root, official_path):
    """接口：本地JSON与指定revision的转换数据逐字段对应；视频内容摘要形成跨任务簇。"""
    import pyarrow.parquet as pq
    root = Path(root); official_path = Path(official_path)
    if (root / 'manifest.json').exists():
        old = read_json(root / 'manifest.json')
        if old['official_revision'] != REVISION or old['official_sha256'] != sha256(official_path):
            raise ValueError('Official MLVU source changed')
        for p, h in old['json_sources'].items():
            if sha256(p) != h: raise ValueError('Local MLVU annotations changed')
        return rows(root)
    # 步骤1：匹配完整原题和选项，不以答案正确性筛题；重复选项沿用官方字母。
    official = pq.read_table(official_path).to_pylist()
    if len(official) != 2174: raise ValueError('Official row count differs')
    bank = {}
    for r in official:
        original = r['question'].split('\n(A) ', 1)[0].strip()
        key = (r['task_type'], r['video_name'], original, tuple(r['candidates']))
        if key in bank: raise ValueError('Ambiguous official row identity')
        bank[key] = r
    data = []; sources = {}; matched = set()
    for pos, file in enumerate(sorted((DATA / 'json').glob('*.json'))[:7]):
        sources[str(file)] = sha256(file)
        for index, raw in enumerate(read_json(file)):
            question = raw['question']; options = list(raw['candidates'])
            key = (TASKS[pos], raw['video'], question.strip(), tuple(options))
            if key not in bank: raise ValueError('Local/official question differs: ' + str((file.name, index)))
            ref = bank[key]; q = 'mlvu-dev-' + ref['question_id']
            if q in matched: raise ValueError('Repeated official question ID')
            matched.add(q); answer = 'ABCD'.index(ref['answer'])
            if options[answer] != raw['answer']: raise ValueError('Official answer mapping differs')
            path = DATA / 'video' / file.stem / raw['video']
            if not path.is_file(): raise FileNotFoundError(path)
            data.append(dict(question_id=q, official_question_id=ref['question_id'], ordinal=len(data),
                question=question, qa_question=ref['question'], options=options, answer_index=answer,
                task_type=TASKS[pos], source_json=str(file), source_row=index,
                original_video_path=str(path), declared_duration_seconds=float(raw['duration']),
                ambiguous_correct_options=options.count(raw['answer']) > 1))
    if len(matched) != 2174 or sum(r['ambiguous_correct_options'] for r in data) != 5:
        raise ValueError('Official mapping/ambiguity denominator differs')
    # 步骤2：内容相同的跨任务视频共用簇和解码入口，保留各题原始路径不覆盖数据。
    paths = sorted({r['original_video_path'] for r in data}); hashes = {}
    with ProcessPoolExecutor(max_workers=8) as executor:
        for i, (p, h) in enumerate(executor.map(_hash_file, paths), 1):
            hashes[p] = h
            if i % 20 == 0: durable(root / 'prepare_status.json', dict(stage='video_hash', completed=i, total=len(paths), utc=utc()))
    canonical = {}
    for p in paths: canonical.setdefault(hashes[p], p)
    for r in data:
        r['video_id'] = hashes[r['original_video_path']]; r['video_path'] = canonical[r['video_id']]
    durable(root / 'manifest.json', dict(rows=data, json_sources=sources, video_hashes=hashes,
        official_path=str(official_path), official_revision=REVISION, official_sha256=sha256(official_path),
        physical_video_paths=len(paths), content_video_clusters=len(canonical), utc=utc()))
    return rows(root)


def index_one(job):
    """接口：完整原片PTS索引使用int64，检查全片解码、帧数及严格时间顺序。"""
    import av
    root, vid, path = Path(job[0]), job[1], job[2]
    dest = root / 'indices' / (vid + '.json'); pts_path = dest.with_suffix('.npy')
    if dest.exists():
        old = read_json(dest)
        if old['video_sha256'] != vid or sha256(pts_path) != old['pts_sha256']:
            raise ValueError('Completed video index changed')
        return vid
    start = time.perf_counter(); pts = []
    # 步骤1：不按平均FPS构造源编号，完整解码确认中段没有损坏或截断。
    with av.open(path) as container:
        s = container.streams.video[0]; s.codec_context.thread_count = 2
        base = Fraction(s.time_base); origin = s.start_time or 0
        if s.duration is None or s.duration <= 0: raise ValueError('Missing source duration')
        for frame in container.decode(s):
            if frame.pts is None or (pts and frame.pts <= pts[-1]): raise ValueError('Invalid full source PTS')
            pts.append(frame.pts)
        if not pts or (s.frames and s.frames != len(pts)): raise ValueError('Source count/EOF differs')
        video = dict(path=path, time_base=str(base), start_pts=origin,
            duration_seconds=float(s.duration * base), duration_fraction=str(s.duration * base),
            candidate_fps=1, phase_seconds=0.25)
    # 步骤2：先保存紧凑索引和摘要再发布完成记录；新问答只能读取已完成资产。
    save_array(pts_path, np.asarray(pts, dtype=np.int64))
    durable(dest, dict(video=video, video_sha256=vid, frame_count=len(pts),
        pts_path=str(pts_path), pts_sha256=sha256(pts_path), wall_seconds=time.perf_counter()-start, utc=utc()))
    return vid


def prepare_indices(root):
    """接口：每个内容视频只解码一次；索引阶段无评分、语言或视觉模型调用。"""
    data = rows(root); unique = {r['video_id']: r['video_path'] for r in data}
    with ProcessPoolExecutor(max_workers=8) as executor:
        for i, vid in enumerate(executor.map(index_one, [(str(root), v, p) for v, p in unique.items()]), 1):
            durable(Path(root) / 'prepare_status.json', dict(stage='full_pts_index', completed=i, total=len(unique), utc=utc()))
    durable(Path(root) / 'indices_complete.json', dict(status='completed', videos=len(unique), new_model_calls=0, utc=utc()))
