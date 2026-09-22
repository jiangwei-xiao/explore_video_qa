"""检索密度v1视频与特征接口：复用粗扫描，补查只解码，新帧只做视觉编码。"""
from fractions import Fraction
from pathlib import Path
import os
import time
import numpy as np
from videoqa_runtime.common import sha256
from videoqa_runtime.baseline_selection import ProtocolError, iter_candidate_frames
from .model_ops import FeatureScorer
from .density_selection import source_time, duration, build_regions, refinement_targets, select_joint


class DensityScorer(FeatureScorer):
    """同一BLIP的粗帧ITM+CLS与新帧纯视觉接口；两类计算分别计数。"""
    def __init__(self, path):
        """接口：加载现有冻结模型，不引入新权重或文本判断器。"""
        super().__init__(path)
        self.counts = dict(itm_batches=0, itm_frames=0, visual_batches=0, visual_frames=0)

    def score_batch_features(self, encoded, images):
        """粗帧一次前向复用ITM与CLS，调用开始即计入计算次数。"""
        self.counts['itm_batches'] += 1
        self.counts['itm_frames'] += len(images)
        return super().score_batch_features(encoded, images)

    def visual_batch(self, images):
        """接口：新增帧只经视觉编码器，返回归一化CLS与同步分阶段耗时。"""
        torch = self.torch
        if not 1 <= len(images) <= 16:
            raise ProtocolError('Invalid pure-visual batch size')
        self.counts['visual_batches'] += 1
        self.counts['visual_frames'] += len(images)
        # 步骤1：使用与粗帧一致的官方预处理和FP32，不构造文本输入。
        torch.cuda.synchronize(); started = time.perf_counter()
        pixels = self.processor(images=images, return_tensors='pt')['pixel_values'].to('cuda', torch.float32)
        torch.cuda.synchronize(); preprocessing = time.perf_counter() - started
        started = time.perf_counter()
        # 步骤2：直接复用同一视觉编码器的last_hidden_state，禁止使用投影后的检索向量。
        with torch.inference_mode():
            cls = self.model.vision_model(pixel_values=pixels, return_dict=True).last_hidden_state[:, 0, :].float()
            if not torch.isfinite(cls).all() or torch.any(torch.linalg.vector_norm(cls, dim=-1) <= 1e-12):
                raise ProtocolError('Invalid new-frame visual CLS')
            features = torch.nn.functional.normalize(cls, p=2, dim=-1).cpu().numpy()
        torch.cuda.synchronize()
        return features, preprocessing, time.perf_counter() - started

    def synthetic_acceptance(self):
        """合成输入核对两条CLS路径和问题不变性，不产生任何真实题答案。"""
        from PIL import Image
        images = [Image.new('RGB', (384, 384), (i * 13, i * 7, i * 3)) for i in range(16)]
        _, a, _, _ = self.score_batch_features(self.tokenize('Which synthetic color is visible?'), images)
        _, b, _, _ = self.score_batch_features(self.tokenize('Is the synthetic background blue?'), images)
        c, _, _ = self.visual_batch(images)
        error = max(float(np.max(np.abs(a - b))), float(np.max(np.abs(a - c))))
        for im in images:
            im.close()
        if error > 1e-6:
            raise ProtocolError(f'Pure visual paths differ: {error}')
        return dict(status='passed',maximum_abs_error=error,absolute_tolerance=1e-6,
                    itm_forwards=2,pure_visual_forwards=1,synthetic_only=True)


def scan_original(path, question, scorer):
    """接口：从原视频独立扫描1FPS粗池，同次ITM前向保留分数和视觉CLS。"""
    import av
    started = time.perf_counter()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is None or stream.duration <= 0:
            raise ProtocolError('Missing exact stream duration')
        exact_duration = str(stream.duration * Fraction(stream.time_base))
    timing = dict(coarse_decode_seconds=time.perf_counter() - started, coarse_rgb_seconds=0.,
                  coarse_preprocess_seconds=0., coarse_forward_seconds=0., coarse_misc_seconds=0.)
    started = time.perf_counter(); encoded = scorer.tokenize(question)
    timing['coarse_preprocess_seconds'] += time.perf_counter() - started
    rows, scores, features, images, video = [], [], [], [], {}
    iterator = iter_candidate_frames(path, video)
    def flush():
        """步骤2：消费有界RGB批次并释放画面，只保存紧凑特征。"""
        values, vectors, pre, forward = scorer.score_batch_features(encoded, images)
        scores.extend(values); features.append(vectors)
        timing['coarse_preprocess_seconds'] += pre; timing['coarse_forward_seconds'] += forward
        for im in images:
            im.close()
        images.clear()
    # 步骤1：复用既有粗网格，确保与基线候选逐源帧一致。
    while True:
        started = time.perf_counter()
        try:
            row, frame = next(iterator)
        except StopIteration:
            timing['coarse_decode_seconds'] += time.perf_counter() - started
            break
        timing['coarse_decode_seconds'] += time.perf_counter() - started
        rows.append(row)
        started = time.perf_counter(); images.append(frame.to_image())
        timing['coarse_rgb_seconds'] += time.perf_counter() - started
        if len(images) == 16:
            flush()
    if images:
        flush()
    if not rows:
        raise ProtocolError('No distinct coarse candidates')
    started = time.perf_counter()
    video['duration_fraction'] = exact_duration
    vectors = np.concatenate(features, axis=0)
    timing['coarse_misc_seconds'] += time.perf_counter() - started
    return dict(video=video,candidates=rows,scores=scores,blip_question_tokens=int(encoded['input_ids'].shape[1])), vectors, timing


def decode_regions(video, coarse, plan, cfg, asset_directory):
    """接口：对获准区域实际2FPS补查，将新帧无损保存以供下一步编码；不调用模型。"""
    import av
    directory = Path(asset_directory); directory.mkdir(parents=True, exist_ok=True)
    base, origin = Fraction(video['time_base']), video['start_pts']
    known = {r['source_pts']: r['source_frame_index'] for r in coarse}
    existing = {r['source_pts']: r['candidate_index'] for r in coarse}
    additions, assets, reports = [], [], []
    rgb_seconds = asset_seconds = 0.
    begin = time.perf_counter()
    # 步骤1：最终区域按源时间排列，选择窗口左侧的精确粗帧作为绝对帧号锚点。
    for region in plan['regions']:
        report = dict(region_id=region['region_id'],enabled=region['refine'],core_seed_counts=region['core_seed_counts'],
                      targets=[],mappings=[],anchor=None,new_indices=[])
        reports.append(report)
        if not region['refine']:
            continue
        targets = refinement_targets(region, video, cfg)
        report['targets'] = list(map(str, targets))
        if not targets:
            continue
        left, right = Fraction(region['left_fraction']), Fraction(region['right_fraction'])
        before = [r for r in coarse if source_time(r, video) <= left]
        anchor = before[-1] if before else None
        if anchor:
            report['anchor'] = dict(source_pts=anchor['source_pts'],source_frame_index=anchor['source_frame_index'])
        cursor, index, anchored = 0, -1, anchor is None
        with av.open(video['path']) as container:
            stream = container.streams.video[0]; stream.codec_context.thread_count = 2
            if Fraction(stream.time_base) != base or (stream.start_time or 0) != origin:
                raise ProtocolError('Video clock changed during refinement')
            if anchor:
                container.seek(anchor['source_pts'], stream=stream, backward=True)
            previous = None
            # 步骤2：先对齐已知PTS再逐帧计数；视频开头从文件起点计数。
            for frame in container.decode(stream):
                pts = frame.pts
                if pts is None or (previous is not None and pts <= previous):
                    raise ProtocolError('Invalid fine decoding PTS')
                previous = pts
                if not anchored:
                    if pts < anchor['source_pts']:
                        continue
                    if pts != anchor['source_pts']:
                        raise ProtocolError('Fine decoder missed exact coarse anchor')
                    index, anchored = anchor['source_frame_index'], True
                else:
                    index += 1
                if pts in known and known[pts] != index:
                    raise ProtocolError('Fine source index differs from coarse full scan')
                instant = (pts - origin) * base
                if instant > right:
                    break
                while cursor < len(targets) and instant >= targets[cursor]:
                    target = targets[cursor]; cursor += 1
                    mapping = dict(target_fraction=str(target),source_pts=pts,source_frame_index=index)
                    if instant < left or instant > duration(video):
                        mapping['status'] = 'outside_window'
                    elif pts in existing:
                        mapping.update(status='duplicate_source',candidate_index=existing[pts])
                    else:
                        # 步骤3：新帧按解码时间顺序稳定编号，原ID不变；每次仅暂存一张RGB。
                        cid = len(coarse) + len(additions)
                        row = dict(candidate_index=cid,source_frame_index=index,source_pts=pts,
                                   source_seconds=float(pts * base),timestamp_seconds=float(instant),
                                   requested_seconds=float(target),requested_fraction=str(target),region_id=region['region_id'])
                        started = time.perf_counter(); image = frame.to_image()
                        rgb_seconds += time.perf_counter() - started
                        started = time.perf_counter(); path = directory / f'{cid}_{pts}.png'
                        temporary = path.with_suffix('.png.tmp')
                        image.save(temporary,format='PNG'); image.close(); os.replace(temporary,path)
                        assets.append(dict(candidate_index=cid,path=str(path),sha256=sha256(path)))
                        asset_seconds += time.perf_counter() - started
                        existing[pts] = cid; additions.append(row); report['new_indices'].append(cid)
                        mapping.update(status='added',candidate_index=cid)
                    report['mappings'].append(mapping)
                if cursor == len(targets):
                    break
            if not anchored:
                raise ProtocolError('No exact anchor during local decoding')
        report['mappings'].extend(dict(target_fraction=str(t),status='no_source_in_window') for t in targets[cursor:])
    if [r['source_pts'] for r in additions] != sorted(r['source_pts'] for r in additions):
        raise ProtocolError('New IDs not stable in chronological order')
    return additions, assets, reports, dict(fine_decode_seconds=time.perf_counter()-begin-rgb_seconds-asset_seconds,
                                           fine_rgb_seconds=rgb_seconds,fine_asset_seconds=asset_seconds)


def prepare_density(path, question, scorer, cfg, asset_directory):
    """核心接口：从原视频独立执行六环节，返回可复算选择池、特征及不重叠阶段耗时。"""
    from PIL import Image
    # 步骤1～4：粗扫描评分复用视觉前向，冻结种子与最终区域。
    pool, coarse_features, timing = scan_original(path, question, scorer)
    started = time.perf_counter(); plan = build_regions(pool['candidates'],pool['scores'],pool['video'],cfg)
    timing['region_build_seconds'] = time.perf_counter() - started
    # 步骤5：只补查解码，不计算新帧ITM，不改原种子/窗口。
    added, assets, reports, fine_time = decode_regions(pool['video'],pool['candidates'],plan,cfg,asset_directory)
    timing.update(fine_time)
    # 步骤6：只对新帧补齐纯视觉特征，来源不参与最终竞争。
    blocks = [coarse_features]
    timing.update(fine_feature_read_seconds=0.,fine_visual_preprocess_seconds=0.,fine_visual_forward_seconds=0.)
    for offset in range(0,len(assets),16):
        started = time.perf_counter(); images=[]
        for asset in assets[offset:offset+16]:
            with Image.open(asset['path']) as im:
                images.append(im.convert('RGB'))
        timing['fine_feature_read_seconds'] += time.perf_counter()-started
        vectors, pre, forward = scorer.visual_batch(images)
        blocks.append(vectors); timing['fine_visual_preprocess_seconds'] += pre; timing['fine_visual_forward_seconds'] += forward
        for im in images:
            im.close()
    started = time.perf_counter(); features = np.concatenate(blocks,axis=0)
    rows = pool['candidates'] + added
    selected, trace = select_joint(rows,features,plan,pool['video'],cfg)
    timing['selection_compete_seconds'] = time.perf_counter()-started
    result = dict(video=pool['video'],candidates=rows,initial_count=len(pool['candidates']),initial_scores=pool['scores'],
                  blip_question_tokens=pool['blip_question_tokens'],plan=plan,refinement=reports,new_assets=assets,
                  new_candidate_count=len(added),selected_indices=selected,selected_frames=[rows[i] for i in selected],
                  selection_trace=trace,application_cache_hits=0,new_frames_have_itm_scores=False)
    return result, features, timing
