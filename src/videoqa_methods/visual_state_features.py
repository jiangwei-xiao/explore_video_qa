"""exp24独立视觉表征：只编码完整画面，不生成文本或判断答案。"""
import hashlib
import time
from pathlib import Path
import numpy as np


def pool_visual_tokens(tokens, grid, merge_size=2):
    """接口：按官方grid分开每个单帧视频，FP32均值及L2归一化，返回[B,D]。

    时间patch的两个位置只能来自同一源帧；不得把另一图像混入这一表示。
    """
    import torch
    grid = torch.as_tensor(grid).cpu()
    # 步骤1：单帧视频内部重复末帧后只有一个时间grid，空间尺寸必须可合并。
    if grid.ndim != 2 or grid.shape[1] != 3 or not torch.all(grid[:, 0] == 1):
        raise ValueError("Expected independently encoded single-frame videos")
    if not torch.all(grid[:, 1:] > 0) or not torch.all(grid[:, 1:] % merge_size == 0):
        raise ValueError("Invalid spatial merge grid")
    lengths = (grid.prod(-1) // merge_size**2).tolist()
    if tokens.ndim != 2 or sum(lengths) != len(tokens) or not torch.isfinite(tokens).all():
        raise ValueError("Visual token/grid population mismatch")
    # 步骤2：先转FP32再平均；不选择层、patch或词，不混合不同源图像。
    means = torch.stack([chunk.float().mean(0) for chunk in tokens.split(lengths)])
    if not torch.isfinite(means).all() or torch.any(means.norm(dim=-1) <= 1e-12):
        raise ValueError("Invalid mean visual state")
    return torch.nn.functional.normalize(means, dim=-1), lengths


class VisualStateEncoder:
    """接口：只读加载Qwen视觉权重，BF16/FA2，最多8个独立单帧视频。"""
    def __init__(self, model_path):
        """接口：严格加载visual前缀全部权重；不实例化语言网络，尺寸以实际grid验收。"""
        import torch
        from safetensors import safe_open
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
        import json
        root = Path(model_path)
        config = AutoConfig.from_pretrained(root, local_files_only=True)
        config.vision_config._attn_implementation = "flash_attention_2"
        # 步骤1：只构建视觉网络，逐分片读取visual.*；语言模型既不加载也不前向。
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(
            config.vision_config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        index = json.loads((root / "model.safetensors.index.json").read_text())['weight_map']
        names = {key: filename for key, filename in index.items() if key.startswith("visual.")}
        if not names:
            raise ValueError("No independently loadable visual checkpoint")
        state = {}
        for filename in sorted(set(names.values())):
            with safe_open(root / filename, framework="pt", device="cpu") as checkpoint:
                for key in sorted(k for k, f in names.items() if f == filename):
                    state[key.removeprefix("visual.")] = checkpoint.get_tensor(key)
        self.visual.load_state_dict(state, strict=True)
        del state
        self.visual = self.visual.to("cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
        if self.visual.config._attn_implementation != "flash_attention_2":
            raise ValueError("FA2 required; fallback forbidden")
        # 步骤2：复用官方处理器，只调用媒体部分，不分词、不构建chat、不接受问题。
        self.processor = AutoProcessor.from_pretrained(root, local_files_only=True, use_fast=False,
            min_pixels=100352, max_pixels=602112,
            size={'shortest_edge': 100352, 'longest_edge': 602112}).image_processor
        self.merge_size = config.vision_config.spatial_merge_size
        self.dimension = config.vision_config.out_hidden_size

    def encode(self, images):
        """接口：有序RGB图像→特征和实际grid/Token/像素身份/同步耗时，无文本输入。"""
        import torch
        if not 1 <= len(images) <= 8:
            raise ValueError("Fixed visual batch bound exceeded")
        torch.cuda.synchronize()
        started = time.perf_counter()
        rgb = [np.asarray(image.convert("RGB")) for image in images]
        # 步骤1：每张图是独立视频，官方处理器复制自身；不把相邻图当作时间patch。
        processed = preprocess_independent_frames(self.processor, rgb)
        grid = processed['video_grid_thw']
        pixels = processed['pixel_values_videos']
        if len(grid) != len(images) or not torch.all(grid[:, 0] == 1):
            raise ValueError("Processor changed source frame count")
        areas = grid[:, 1] * grid[:, 2] * 14**2
        if not torch.all((areas >= 100352) & (areas <= 602112)):
            raise ValueError("Actual processor size violates frozen pixel bounds before vision forward")
        observation = dict(rgb_sha256=[hashlib.sha256(array.tobytes()).hexdigest() for array in rgb],
            pixel_sha256=hashlib.sha256(pixels.numpy().tobytes()).hexdigest(),
            video_grid_thw=grid.tolist(), processed_sizes=(grid[:, 1:] * 14).tolist(),
            source_frames=len(images), internal_duplicate_frames=len(images),
            input_media="independent single-frame videos; no question, options, or text",
            feature_dtype="float32", vision_dtype="bfloat16", attention="flash_attention_2")
        pixels = pixels.to("cuda", dtype=torch.bfloat16)
        grid_gpu = grid.to("cuda")
        torch.cuda.synchronize()
        observation['preprocess_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        # 步骤2：只调用视觉网络；patch合并后按grid分组，均值向量不称证据充分性。
        with torch.inference_mode():
            tokens = self.visual(pixels, grid_thw=grid_gpu)
            features, lengths = pool_visual_tokens(tokens, grid, self.merge_size)
            features = features.cpu().numpy()
        torch.cuda.synchronize()
        observation.update(visual_token_counts=lengths, forward_seconds=time.perf_counter()-started,
            dimension=features.shape[1], allocated_peak_bytes=torch.cuda.max_memory_allocated(),
            qa_calls=0, text_forward_calls=0)
        return features, observation


def encode_visual_states(images, encoder):
    """接口：显式注入已加载编码器，复用长期进程；不隐式预热或新增模型。"""
    return encoder.encode(images)


def preprocess_independent_frames(processor, rgb):
    """接口：显式调用官方媒体preprocess，images=None避免BaseImageProcessor必填位置参数。

    每个外层列表项都是独立视频，单帧补齐由官方实现完成，不借用其他画面。
    """
    return processor.preprocess(images=None, videos=[array[None] for array in rgb],
        size={'shortest_edge': 100352, 'longest_edge': 602112}, return_tensors="pt")


def select_with_state_features(pool, features, cfg, parent_trace):
    """接口：同一池/ITM/计划换视觉数组，重算原前缀与exp12 DP；另报固定父配额。

    旧曲线不能与新特征混用；先重建等权曲线，再原样调用加权选择器。
    """
    from videoqa_methods.prefix_coverage_budget import select_prefix_coverage
    from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
    from videoqa_methods.density_selection import frame_key
    # 步骤1：定位和旧配额仅用于控制，不接受答案或人工见证。
    _, prefix = select_prefix_coverage(pool['candidates'], features, pool['plan'], pool['video'], cfg,
        parent_trace['region_budgets'], pool['initial_scores'])
    # 步骤2：用新特征对应的新曲线计算原exp12目标，不改变覆盖权重和稳定同分规则。
    selected, weighted = select_relevance_coverage(pool, features, prefix)
    fixed = sorted([i for r in prefix['regions'] for i in r['prefix'][:parent_trace['region_budgets'][str(r['region_id'])]]],
        key=lambda i: frame_key(pool['candidates'][i]))
    return selected, dict(weighted, prefix_trace=prefix, fixed_parent_quota_indices=fixed,
        visual_state_rule="Qwen vision post-merger FP32 mean/L2; same exp12 selection rules")


def competition_snapshot(pool, region, before, witness, winner, feature_lookup):
    """接口：冻结父版本实际竞争上下文，只换表征重算所需状态和当步胜者。

    feature_lookup按候选号提供归一化向量；人工见证不传入正式选择器。
    """
    import math
    seeds = next(r['seed_indices'] for r in pool['plan']['regions'] if r['region_id'] == region['region_id'])
    scores = pool['initial_scores']
    peak = max(scores[j] for j in seeds)
    # 步骤1：同一批已选画面和种子权重；不以新特征另挑有利的比较对象。
    def value(i):
        """在同一原始已选上下文里计算支持和最大重复度，不读取人工答案。"""
        support = max((scores[j]/peak if peak else 1.) * float(np.clip(feature_lookup[i] @ feature_lookup[j], 0, 1)) for j in seeds)
        redundant = max((float(np.clip(feature_lookup[i] @ feature_lookup[j], 0, 1)) for j in before), default=0.)
        return dict(support=support, redundancy=redundant, local_score=support*math.sqrt(1-redundant))
    a, b = value(witness), value(winner)
    # 步骤2：只记录同表示中的排序；不同表示的分数尺度不直接比较。
    return dict(witness=a, actual_winner=b, witness_ahead=a['local_score'] > b['local_score'] + 1e-6,
        witness_index=witness, actual_winner_index=winner, selected_before=list(before))
