"""Independent raw-video frontends with serial, disjoint timing intervals."""
import math
import time
from fractions import Fraction

from .video import decode_selected, uniform_selection


class ProtocolError(ValueError):
    pass


def select_topk(candidates, scores, count=16):
    """Top-K选帧：count 为最大预算（2026-09-17 修订），候选不足时全部入选；
    候选数≥预算时与原固定预算行为完全一致。"""
    if len(candidates) != len(scores) or not candidates or count < 1:
        raise ProtocolError('Invalid candidate/score lengths or empty candidates')
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ProtocolError('Non-finite or out-of-range BLIP score')
    effective = min(count, len(candidates))
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], candidates[i]['source_pts'], candidates[i]['candidate_index']))[:effective]
    return sorted((candidates[i] for i in order), key=lambda r: (r['source_pts'], r['candidate_index']))


def iter_candidate_frames(path, metadata):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        tb, start = Fraction(stream.time_base), stream.start_time or 0
        if stream.duration is None or stream.duration <= 0:
            raise ProtocolError('Missing positive video duration')
        metadata.update(path=str(path), time_base=str(tb), start_pts=start,
                        duration_seconds=float(stream.duration * tb), candidate_fps=1, phase_seconds=0.25)
        previous, target, number = None, Fraction(1, 4), 0
        for source_index, frame in enumerate(container.decode(stream)):
            pts = frame.pts
            if pts is None or (previous is not None and pts <= previous):
                raise ProtocolError('Missing or non-increasing source PTS')
            previous = pts
            timestamp = (pts - start) * tb
            if timestamp < target:
                continue
            row = dict(candidate_index=number, source_frame_index=source_index, source_pts=pts,
                       source_seconds=float(pts * tb), timestamp_seconds=float(timestamp), requested_seconds=float(target))
            number += 1
            target += int(timestamp - target) + 1
            yield row, frame


class BlipItmScorer:
    def __init__(self, model_path):
        import torch
        from transformers import BlipForImageTextRetrieval, BlipProcessor
        self.torch = torch
        self.processor = BlipProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = BlipForImageTextRetrieval.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.float32).eval().requires_grad_(False).cuda()
        self.limit = self.model.config.text_config.max_position_embeddings

    def tokenize(self, question):
        encoded = self.processor.tokenizer(question, return_tensors='pt', truncation=False)
        if encoded['input_ids'].shape[1] > self.limit:
            raise ProtocolError('BLIP question would exceed its position limit')
        return encoded

    def score_batch(self, encoded, images):
        torch = self.torch
        if not 1 <= len(images) <= 16:
            raise ProtocolError('BLIP batch exceeds the frozen batch size')
        torch.cuda.synchronize()
        start = time.perf_counter()
        pixels = self.processor(images=images, return_tensors='pt')['pixel_values'].to('cuda', torch.float32)
        inputs = {k: v.repeat(len(images), 1).cuda() for k, v in encoded.items() if k in ('input_ids', 'attention_mask')}
        torch.cuda.synchronize()
        preprocess = time.perf_counter() - start
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model(pixel_values=pixels, **inputs, use_itm_head=True)
            scores = output.itm_score.softmax(-1)[:, 1].cpu().tolist()
        torch.cuda.synchronize()
        forward = time.perf_counter() - start
        if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise ProtocolError('Invalid ITM positive-class probabilities')
        return scores, preprocess, forward

    def warmup(self):
        from PIL import Image
        encoded = self.tokenize('Which synthetic color is visible in this image?')
        for _ in range(2):
            self.score_batch(encoded, [Image.new('RGB', (1280, 720), 'gray')] * 16)


def prepare_selection(path, question, method, scorer, progress=None):
    if method not in ('uniform', 'topk'):
        raise ProtocolError('Unknown baseline')
    timing = dict(candidate_decode_seconds=0.0, rgb_conversion_seconds=0.0,
                  blip_preprocess_seconds=0.0, blip_forward_seconds=0.0,
                  ranking_seconds=0.0, selected_decode_seconds=0.0)
    candidates, scores, batch, metadata = [], [], [], {}
    token_count = None
    if method == 'topk':
        start = time.perf_counter()
        encoded = scorer.tokenize(question)
        token_count = encoded['input_ids'].shape[1]
        timing['blip_preprocess_seconds'] += time.perf_counter() - start

    def flush():
        values, preprocessing, forward = scorer.score_batch(encoded, batch)
        scores.extend(values)
        timing['blip_preprocess_seconds'] += preprocessing
        timing['blip_forward_seconds'] += forward
        batch.clear()

    iterator = iter_candidate_frames(path, metadata)
    while True:
        start = time.perf_counter()
        try:
            row, frame = next(iterator)
        except StopIteration:
            timing['candidate_decode_seconds'] += time.perf_counter() - start
            break
        timing['candidate_decode_seconds'] += time.perf_counter() - start
        candidates.append(row)
        if progress is not None and len(candidates) % 256 == 0:
            progress(len(candidates))
        if method == 'topk':
            start = time.perf_counter()
            batch.append(frame.to_image())
            timing['rgb_conversion_seconds'] += time.perf_counter() - start
            if len(batch) == 16:
                flush()
    if batch:
        flush()
    if not candidates:
        # 2026-09-17 修订：16帧预算改为最大上限，候选不足时选帧函数取全部候选；空候选仍为协议错误
        raise ProtocolError('No distinct candidate frames')
    start = time.perf_counter()
    selected = uniform_selection(candidates) if method == 'uniform' else select_topk(candidates, scores)
    timing['ranking_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    frames = decode_selected(metadata, selected)
    timing['selected_decode_seconds'] = time.perf_counter() - start
    timing['selection_core_seconds'] = sum(timing[k] for k in (
        'rgb_conversion_seconds', 'blip_preprocess_seconds', 'blip_forward_seconds', 'ranking_seconds'))
    return frames, dict(video=metadata, candidates=candidates, scores=scores, selected_frames=selected,
                        blip_question_tokens=token_count, timings=timing, application_cache_hits=0)
