"""非生成式选项支持探针：空白反事实分布，不把模型偏好当正确答案。"""
import hashlib
import time
import numpy as np


def normalized_js(left, right):
    """接口：五类概率的Jensen-Shannon散度÷ln2，严格保留Other质量，不拟合温度。"""
    p = np.asarray(left, dtype=np.float64); q = np.asarray(right, dtype=np.float64)
    if p.shape != (5,) or q.shape != (5,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Expected five finite next-token outcomes')
    if (p < 0).any() or (q < 0).any() or not np.isclose(p.sum(), 1, atol=1e-7) or not np.isclose(q.sum(), 1, atol=1e-7):
        raise ValueError('Invalid probability simplex')
    # 步骤1：使用真实五类质量，不把微小字母概率重新放大为四选项的强支持。
    middle = (p + q) / 2; value = 0.0
    for values in (p, q):
        mask = values > 0
        value += 0.5 * float(np.sum(values[mask] * np.log(values[mask] / middle[mask])))
    # 步骤2：0ln0=0；截断只处理浮点尾差，不增加语义判断阈值。
    return float(np.clip(value / np.log(2), 0, 1))


def evidence_support(frame_distribution, blank_distribution):
    """接口：合法字母质量×反事实变化，返回E及可解释分量，不读取标准答案。

    E是下一Token选择支持变化的代理，非正确性、置信校准或证据充分性。
    乘合法字母质量，避免主要转向非字母回复的画面因格式变化得到强奖励。
    """
    divergence = normalized_js(frame_distribution, blank_distribution)
    mass = float(np.sum(np.asarray(frame_distribution, dtype=np.float64)[:4]))
    return dict(value=mass * divergence, letter_mass=mass, normalized_js=divergence)


def reference_question_text(question, options):
    """接口：最小原题/原选项对象→固定参考任务文本，禁止携带正确答案字段。"""
    from videoqa_lmms_bridge.common import load_task, POST_PROMPT
    if not isinstance(question, str) or not question.strip() or len(options) != 4:
        raise ValueError('Question and four original options required')
    if not all(isinstance(option, str) and option.startswith(letter + '.') for letter, option in zip('ABCD', options)):
        raise ValueError('Original option order must remain A/B/C/D')
    # 步骤1：直接调用固定任务函数；不调用含answer的task_doc，不额外加入时间说明。
    return load_task().videomme_doc_to_text(dict(question=question, options=list(options)), {'post_prompt': POST_PROMPT})


class OptionEvidenceScorer:
    """接口：同后端Qwen官方视频prefill，只读取下一Token分布，不生成或解析答案。"""
    def __init__(self):
        """接口：冻结BF16/FA2模型和四个字母Token；不训练、不增模型、不换模板。"""
        from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
        self.backend = QwenFA2VideoBackend(); self.torch = self.backend.torch
        self.model = self.backend.model; self.processor = self.backend.processor
        token_ids = [self.processor.tokenizer.encode(letter, add_special_tokens=False) for letter in 'ABCD']
        if any(len(ids) != 1 for ids in token_ids) or len({ids[0] for ids in token_ids}) != 4:
            raise ValueError('Four single distinct uppercase letter tokens required')
        self.label_ids = [ids[0] for ids in token_ids]; self.load_report = self.backend.load_report

    def score(self, frames, question, options, before_forward=None):
        """接口：两个有序输入位置/题/四选项→五类分布及成本，无gold或时间文字接口。

        基线是同一锚点自身重复，候选条件是锚点与候选按真实PTS排序。
        物理源身份在外层账本记录，此处不把不同源帧的相同RGB误算成同一源。
        """
        from videoqa_full.qwen_backend import MIN_PIXELS, MAX_PIXELS
        torch = self.torch; torch.cuda.synchronize(); began = time.perf_counter()
        if len(frames) != 2:
            raise ValueError('Exactly two context input positions required')
        rgb = [np.ascontiguousarray(np.asarray(image.convert('RGB'))) for image in frames]
        if rgb[0].shape != rgb[1].shape:
            raise ValueError('Anchor and candidate source geometry differs')
        task_text = reference_question_text(question, options)
        messages = [dict(role='system', content='You are a helpful assistant.'),
            dict(role='user', content=[dict(type='video'), dict(type='text', text=task_text)])]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 步骤1：仅使用两个调用方冻结位置，真实PTS只用于顺序，不写入提示词。
        inputs = self.processor(text=[prompt], videos=[np.stack(rgb)], fps=2.0,
            images_kwargs={'size': {'shortest_edge': MIN_PIXELS, 'longest_edge': MAX_PIXELS}},
            padding=False, truncation=False, return_tensors='pt')
        grid = inputs['video_grid_thw'].tolist(); height, width = grid[0][1] * 14, grid[0][2] * 14
        if len(grid) != 1 or grid[0][0] != 1 or height % 28 or width % 28 or not MIN_PIXELS <= height * width <= MAX_PIXELS:
            raise ValueError('Actual single-frame video geometry violates protocol')
        visual = int(np.prod(grid[0]) // 4)
        if int((inputs['input_ids'] == self.model.config.video_token_id).sum()) != visual:
            raise ValueError('Video placeholders differ')
        if int((inputs['input_ids'] == self.model.config.image_token_id).sum()) or list(inputs['second_per_grid_ts']) != [1.0]:
            raise ValueError('Media type or FPS convention differs')
        if inputs['input_ids'].shape[1] >= self.model.config.max_position_embeddings:
            raise ValueError('Input truncation forbidden')
        meta = dict(question=question, options=list(options), prompt=prompt,
            rgb_sha256=[hashlib.sha256(array.tobytes()).hexdigest() for array in rgb], rgb_shape=list(rgb[0].shape),
            video_grid_thw=grid, processed_size=[height, width], visual_tokens=visual,
            prefill_tokens=int(inputs['input_ids'].shape[1]),
            token_sha256=hashlib.sha256(inputs['input_ids'].numpy().tobytes()).hexdigest(),
            pixel_sha256=hashlib.sha256(inputs['pixel_values_videos'].numpy().tobytes()).hexdigest(),
            letter_token_ids=self.label_ids, encoding_fps=2.0, second_per_grid_ts=[1.0],
            input_positions=2, physical_source_identity='Persisted by caller, not inferred from RGB equality',
            input_options=True, input_gold=False, generated_tokens=0, final_QA_calls=0)
        inputs = inputs.to('cuda:0'); torch.cuda.synchronize(); meta['preprocess_seconds'] = time.perf_counter() - began
        counts = dict(vision=0, language=0)
        def vision(module, args, output):
            """接口：计数真实视觉前向，不把辅助回答分布成本隐藏在旧评分中。"""
            counts['vision'] += 1
        def language(module, args, output):
            """接口：计数语言prefill，监督的是公开选项，不是标准答案。"""
            counts['language'] += 1
        hooks = [self.model.visual.register_forward_hook(vision), self.model.model.register_forward_hook(language)]
        try:
            # 步骤2：开始检查点先于前向；只读原logits，既不采样也不构造生成配置。
            if before_forward is not None:
                before_forward(meta)
            torch.cuda.reset_peak_memory_stats(); began = time.perf_counter()
            with torch.inference_mode():
                result = self.model(**inputs, use_cache=False, output_hidden_states=False,
                    output_attentions=False, return_dict=True)
                logits = result.logits[0, -1].float()
                if not torch.isfinite(logits).all() or counts != dict(vision=1, language=1):
                    raise ValueError('Invalid next-token forward')
                probabilities = torch.softmax(logits, dim=-1)
                letters = probabilities[self.label_ids].double().cpu().numpy()
                raw_logits = logits[self.label_ids].cpu().tolist()
            torch.cuda.synchronize()
            # 步骤3：Other包含全部剩余词表；全五类重归一仅消除浮点质量尾差。
            distribution = np.concatenate([letters, [max(0.0, 1 - float(letters.sum()))]])
            distribution /= distribution.sum()
            meta.update(forward_seconds=time.perf_counter() - began, actual_forward_counts=counts,
                letter_logits=raw_logits, allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                attention=self.model.config._attn_implementation, dtype=str(self.model.dtype))
            del result
            return distribution, meta
        finally:
            for hook in hooks:
                hook.remove()
