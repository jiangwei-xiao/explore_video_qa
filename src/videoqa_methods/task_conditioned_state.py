"""问题条件状态探针：固定最后层prefill状态，不生成文字或选帧意见。"""
import hashlib
import time
import numpy as np


def image_residual(state, blank_state):
    """接口：相同问题/位置/媒体网格的真实减空白状态→单位向量及有效性。

    减法只定义视觉输入相对空白的影响，不保证语义充分性。完全无差异时
    没有新信号，返回无效标志，由重复度接口沿用原视觉规则，不奖励空帧。
    """
    a = np.asarray(state, dtype=np.float32); b = np.asarray(blank_state, dtype=np.float32)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('State/baseline identity or values differ')
    # 步骤1：先在FP32计算残差，避免BF16下继续丢失微小状态差异。
    residual = a - b; norm = float(np.linalg.norm(residual))
    # 步骤2：零向量不生成任意方向，也不通过加epsilon制造新奇奖励。
    if norm <= 1e-12:
        return np.zeros_like(residual), False, norm
    return residual / norm, True, norm


def conditional_redundancy(old_target, old_context, target, context, target_valid, context_valid):
    """接口：同一问题下的任务状态余弦→最大重复度；无信号帧对复用原视觉D。"""
    old = np.clip(np.asarray(old_context) @ np.asarray(old_target), 0, 1)
    new = np.clip(np.asarray(context) @ np.asarray(target), 0, 1)
    valid = np.asarray(context_valid, dtype=bool)
    if old.ndim != 1 or old.shape != new.shape or valid.shape != old.shape:
        raise ValueError('Selected-context populations differ')
    # 步骤1：只在双方都有视觉残差时启用新表示，不混合系数或调整原题支持。
    values = np.where(valid & bool(target_valid), new, old)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite task-conditioned redundancy')
    # 步骤2：同样取已选帧的最大相似，维持父选择器的集合更新语义。
    return float(values.max()) if len(values) else 0.0


class QuestionStateEncoder:
    """接口：只读Qwen全模型prefill；原问题与独立单帧，不输入选项/答案。"""
    def __init__(self):
        """接口：复用已验收BF16/FA2加载，不升级依赖，不改变官方模型代码。"""
        from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
        # 步骤1：与exp24的纯视觉均值不同，这里明确计入语言网络prefill成本。
        self.backend = QwenFA2VideoBackend()
        self.torch = self.backend.torch
        self.model = self.backend.model
        self.processor = self.backend.processor
        self.load_report = self.backend.load_report

    def encode(self, image, question, before_forward=None):
        """接口：原RGB/原问题→最后输入位置的FP32状态及实际输入、Token、计时。

        仅调用官方forward，禁止generate；输出logits只检查有限性，不用于评分。
        before_forward在模型前向前持久化开始状态，未知返回不能自动重复调用。
        """
        from videoqa_full.qwen_backend import MIN_PIXELS, MAX_PIXELS
        torch = self.torch
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Original question is required')
        torch.cuda.synchronize(); began = time.perf_counter()
        rgb = np.ascontiguousarray(np.asarray(image.convert('RGB')))
        # 步骤1：同一帧自身组成时间patch，原尺寸/帧内容不裁剪、不加入其他源帧。
        video = np.stack([rgb, rgb])
        messages = [dict(role='system', content='You are a helpful assistant.'),
            dict(role='user', content=[dict(type='video'), dict(type='text', text=question)])]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], videos=[video], fps=2.0,
            images_kwargs={'size': {'shortest_edge': MIN_PIXELS, 'longest_edge': MAX_PIXELS}},
            padding=False, truncation=False, return_tensors='pt')
        grid = inputs['video_grid_thw'].tolist()
        if len(grid) != 1 or grid[0][0] != 1:
            raise ValueError('One independent frame must give one temporal grid')
        height, width = grid[0][1] * 14, grid[0][2] * 14
        if height % 28 or width % 28 or not MIN_PIXELS <= height * width <= MAX_PIXELS:
            raise ValueError('Actual processor size violates frozen limits')
        visual = int(np.prod(grid[0]) // 4)
        if int((inputs['input_ids'] == self.model.config.video_token_id).sum()) != visual:
            raise ValueError('Video placeholders and features differ')
        if int((inputs['input_ids'] == self.model.config.image_token_id).sum()) != 0 or list(inputs['second_per_grid_ts']) != [1.0]:
            raise ValueError('Media type or encoding-FPS convention differs')
        if inputs['input_ids'].shape[1] >= self.model.config.max_position_embeddings:
            raise ValueError('Question input would exceed context; truncation forbidden')
        metadata = dict(prompt=prompt, question=question, rgb_shape=list(rgb.shape),
            rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), video_grid_thw=grid,
            processed_size=[height, width], visual_tokens=visual,
            prefill_tokens=int(inputs['input_ids'].shape[1]),
            token_sha256=hashlib.sha256(inputs['input_ids'].numpy().tobytes()).hexdigest(),
            pixel_sha256=hashlib.sha256(inputs['pixel_values_videos'].numpy().tobytes()).hexdigest(),
            unique_source_frames=1, internal_duplicate_frames=1,
            encoding_fps=2.0, second_per_grid_ts=[1.0],
            question_only=True, options_or_answers=False, generated_tokens=0)
        inputs = inputs.to('cuda:0'); torch.cuda.synchronize()
        metadata['preprocess_seconds'] = time.perf_counter() - began
        captured = {}; counters = dict(vision=0, language=0)
        # 步骤2：只观察官方最后层输出，不要求返回全部层，也不修改隐藏状态。
        def capture(module, args, output):
            """接口：一次语言prefill→最后有效输入位置；无padding，因此取末位置。"""
            counters['language'] += 1
            hidden = output.last_hidden_state
            if hidden.shape[:2] != inputs['input_ids'].shape:
                raise ValueError('Decoder state and input sequence differ')
            captured['state'] = hidden[0, -1].detach().float().clone()
        def vision_count(module, args, output):
            """接口：计数实际视觉前向，防止将语言表示的额外成本漏报。"""
            counters['vision'] += 1
        hooks = [self.model.model.register_forward_hook(capture),
            self.model.visual.register_forward_hook(vision_count)]
        try:
            if before_forward is not None:
                before_forward(metadata)
            torch.cuda.reset_peak_memory_stats(); began = time.perf_counter()
            with torch.inference_mode():
                result = self.model(**inputs, use_cache=False, output_hidden_states=False,
                    output_attentions=False, return_dict=True)
            torch.cuda.synchronize()
            if counters != dict(vision=1, language=1) or 'state' not in captured:
                raise ValueError('Expected exactly one vision and language prefill')
            if not torch.isfinite(result.logits[:, -1]).all():
                raise ValueError('Non-finite prefill logits')
            state = captured['state'].cpu().numpy()
            if not np.isfinite(state).all():
                raise ValueError('Non-finite final state')
            metadata.update(forward_seconds=time.perf_counter() - began,
                actual_forward_counts=counters, dimension=len(state),
                attention=self.model.config._attn_implementation, dtype=str(self.model.dtype),
                allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                reserved_peak_bytes=torch.cuda.max_memory_reserved(), qa_calls=0)
            del result
            return state, metadata
        finally:
            # 步骤3：成功或异常都移除本次观测钩子，下一帧不继承旧捕获对象。
            for hook in hooks:
                hook.remove()
