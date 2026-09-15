import time
import numpy as np
from videoqa_runtime.baseline_selection import iter_candidate_frames, select_topk
from videoqa_runtime.video import decode_selected
from .algorithm import segment_scores, compete, hotspot_candidates, reselect, membership
from .refinement import decode_hotspots


def scan(path, question, scorer, progress=None):
    """核心接口：独立扫描原视频，按16帧批次产生相关性和纯视觉特征；返回候选池、特征矩阵、分阶段计时和问题编码，不读取其他变体缓存。"""
    # 步骤1：独立建立本次扫描状态，只保留元数据、分数、特征与一个RGB批次。
    rows, scores, blocks, batch, video = [], [], [], [], {}
    times = dict(candidate_decode_seconds=0.0, rgb_seconds=0.0, blip_preprocess_seconds=0.0,
                 blip_forward_features_seconds=0.0)
    start = time.perf_counter()
    encoded = scorer.tokenize(question)
    times['blip_preprocess_seconds'] += time.perf_counter() - start

    def flush():
        """处理当前图像批次，累计CPU特征与计时后释放RGB批次，控制长视频内存上界。"""
        values, features, preprocessing, forward = scorer.score_batch_features(encoded, batch)
        scores.extend(values)
        blocks.append(features)
        times['blip_preprocess_seconds'] += preprocessing
        times['blip_forward_features_seconds'] += forward
        batch.clear()

    # 步骤2：解码等待、RGB转换和BLIP批次分别计时，避免GPU异步提交造成漏计。
    iterator = iter_candidate_frames(path, video)
    while True:
        start = time.perf_counter()
        try:
            row, frame = next(iterator)
        except StopIteration:
            times['candidate_decode_seconds'] += time.perf_counter() - start
            break
        times['candidate_decode_seconds'] += time.perf_counter() - start
        rows.append(row)
        start = time.perf_counter()
        batch.append(frame.to_image())
        times['rgb_seconds'] += time.perf_counter() - start
        if len(batch) == 16:
            flush()
        if progress and len(rows) % 256 == 0:
            progress(len(rows))
    if batch:
        flush()
    # 步骤3：按候选编号拼接特征，与分数保持严格一一对应。
    return dict(video=video, candidates=rows, scores=scores, blip_question_tokens=int(encoded['input_ids'].shape[1])), np.concatenate(blocks), times, encoded


def select_method(path, question, variant, scorer, classifier, cfg, scope_state, progress=None):
    """核心接口：执行A/B/C的完整选择流程，返回16张RGB帧、可序列化机制记录、特征和计时；标准答案及选项不进入该接口。"""
    # 步骤1：B/C先分类并计入方法总耗时，A使用固定λ；分类结果只影响选帧。
    started = time.perf_counter()
    if variant in ('B', 'C'):
        scope = classifier.classify(question, scope_state)
        lam = cfg['lambda'][scope['label']]
    else:
        scope = dict(label='FIXED', fallback=False, seconds=0.0)
        lam = cfg['fixed_lambda']
    # 步骤2：独立全片扫描、分段、逐帧预算竞争，冻结粗选配额。
    pool, features, times, encoded = scan(path, question, scorer, progress)
    times['scope_seconds'] = scope['seconds']
    start = time.perf_counter()
    segments, wavelet = segment_scores(pool['scores'], pool['candidates'], pool['video'], cfg)
    times['segmentation_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    coarse, quotas, competition = compete(pool['candidates'], pool['scores'], features, segments, lam, cfg, cfg['frames'])
    times['competition_seconds'] = time.perf_counter() - start
    n_initial = len(pool['candidates'])
    selected, hotspots, decoding, reselection = coarse, [], [], []
    times.update(hotspot_seconds=0.0, fine_decode_seconds=0.0, fine_rgb_seconds=0.0,
                 fine_preprocess_seconds=0.0, fine_forward_features_seconds=0.0, reselection_seconds=0.0)
    # 步骤3：仅C判断热点并做一次有限补查；A/B不进入此分支。
    if variant == 'C':
        start = time.perf_counter()
        hotspots, windows = hotspot_candidates(pool['candidates'], pool['scores'], features, coarse,
                                               segments, quotas, pool['video'], cfg)
        times['hotspot_seconds'] = time.perf_counter() - start
        added, fine_images, decoding, fine_times = decode_hotspots(pool['video'], pool['candidates'], windows, cfg)
        times.update(fine_times)
        new_features = []
        for i in range(0, len(fine_images), 16):
            values, matrix, pre, forward = scorer.score_batch_features(encoded, fine_images[i:i + 16])
            pool['scores'].extend(values)
            new_features.append(matrix)
            times['fine_preprocess_seconds'] += pre
            times['fine_forward_features_seconds'] += forward
        pool['candidates'].extend(added)
        if new_features:
            features = np.concatenate([features, *new_features])
        # 步骤4：在重选与回答之前截取C的真实前序耗时，用于D的诊断成本归因。
        pool_acquisition_seconds = time.perf_counter() - started
        start = time.perf_counter()
        selected, reselection = reselect(pool['candidates'], pool['scores'], features, segments, quotas,
                                         coarse, n_initial, lam, cfg)
        times['reselection_seconds'] = time.perf_counter() - start
    else:
        pool_acquisition_seconds = None
    start = time.perf_counter()
    # 步骤5：按真实PTS解码最终16帧，并保留分段、增益、配额与热点的完整证据。
    selected_rows = [pool['candidates'][i] for i in selected]
    frames = decode_selected(pool['video'], selected_rows)
    times['selected_decode_seconds'] = time.perf_counter() - start
    pool.update(scope=scope, lambda_value=lam, initial_count=n_initial, segments=segments, wavelet=wavelet,
                segment_ids=membership(pool['candidates'], segments).tolist(), coarse_selected=coarse,
                quotas=quotas, competition_trace=competition, hotspot_trace=hotspots, fine_decode_trace=decoding,
                reselection_trace=reselection, selected_indices=selected, selected_frames=selected_rows,
                pool_acquisition_seconds=pool_acquisition_seconds)
    times['selection_core_seconds'] = sum(v for k, v in times.items() if k not in (
        'candidate_decode_seconds', 'fine_decode_seconds', 'selected_decode_seconds'))
    return frames, pool, features, times


def select_diagnostic(donor):
    """接口：D只在C已冻结的候选池上做全局Top-K并解码16帧，返回实际新增的选择成本；前序成本由调用方单独归因。"""
    started = time.perf_counter()
    rows = select_topk(donor['pool']['candidates'], donor['pool']['scores'])
    ranking_seconds = time.perf_counter() - started
    started = time.perf_counter()
    frames = decode_selected(donor['pool']['video'], rows)
    decode_seconds = time.perf_counter() - started
    pool = dict(donor['pool'])
    pool['selected_indices'] = [r['candidate_index'] for r in rows]
    pool['selected_frames'] = rows
    pool['diagnostic_donor_variant'] = 'C'
    return frames, pool, dict(ranking_seconds=ranking_seconds, selected_decode_seconds=decode_seconds,
                             selection_core_seconds=ranking_seconds)
