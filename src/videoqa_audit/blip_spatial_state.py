"""原问题条件的BLIP空间诊断；不生成答案、不参与正式选帧。"""
from contextlib import contextmanager
import numpy as np


@contextmanager
def capture_cross_attention(module):
    """接口：临时读取一层cross-attention概率，退出时恢复接口并移除hook。

    仅改变是否返回注意力，不改变上下文计算；额外概率在交给父层前去掉。
    """
    captured = {}
    # 步骤1：兼容HF按位置传入的output_attentions；原接口必须为不输出注意力。
    def before(_module, args, kwargs):
        if len(args) > 6:
            if args[6]:
                raise ValueError("Original attention output unexpectedly enabled")
            args = (*args[:6], True, *args[7:])
        else:
            if kwargs.get("output_attentions", False):
                raise ValueError("Original attention output unexpectedly enabled")
            kwargs = dict(kwargs, output_attentions=True)
        return args, kwargs

    def after(_module, _args, output):
        if len(output) != 3 or "attention" in captured:
            raise ValueError("Unexpected attention interface or repeated layer")
        captured["attention"] = output[1]
        return (output[0], output[2])

    pre = module.register_forward_pre_hook(before, with_kwargs=True)
    post = module.register_forward_hook(after)
    try:
        yield captured
    finally:
        # 步骤2：即使前向/梯度失败也移除临时hook，不能污染后续基准控制。
        post.remove()
        pre.remove()


def spatial_batch(scorer, encoded, images, spatial=False):
    """接口：最多16个同问题RGB→ITM、纯视觉CLS/patch、空间图及同步耗时。

    spatial=True使用固定第6索引层的正ITM logit梯度；权重保持冻结，只让输入可求导。
    """
    import hashlib
    import time
    import torch
    if not 1 <= len(images) <= 16:
        raise ValueError("BLIP batch bound exceeded")
    if any(p.requires_grad for p in scorer.model.parameters()):
        raise ValueError("Diagnostic weights must remain frozen")
    torch.cuda.synchronize()
    started = time.perf_counter()
    # 步骤1：复用官方预处理和原无截断分词，不导入LAVIS默认35 Token限制。
    pixels = scorer.processor(images=images, return_tensors="pt")["pixel_values"].to("cuda", torch.float32)
    inputs = {k: v.repeat(len(images), 1).cuda() for k, v in encoded.items() if k in ("input_ids", "attention_mask")}
    pixel_hash = hashlib.sha256(pixels.detach().cpu().numpy().tobytes()).hexdigest()
    torch.cuda.synchronize()
    pre_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if spatial:
        pixels.requires_grad_(True)
        target = scorer.model.text_encoder.encoder.layer[6].crossattention.self
        with torch.enable_grad(), capture_cross_attention(target) as captured:
            output = scorer.model(pixel_values=pixels, **inputs, use_itm_head=True)
            attention = captured["attention"]
            gradient, = torch.autograd.grad(output.itm_score[:, 1].sum(), attention)
            # 步骤2：按官方正梯度×注意力、head均值、非特殊文本Token均值取得576 patch图。
            mask = inputs["attention_mask"].bool().clone()
            mask[:, 0] = False
            last = inputs["attention_mask"].sum(-1) - 1
            mask[torch.arange(len(images), device="cuda"), last] = False
            if not mask.any(-1).all():
                raise ValueError("No original question content tokens")
            per_token = (attention[:, :, :, 1:] * gradient[:, :, :, 1:].clamp(min=0)).mean(1)
            cams = (per_token * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            if cams.shape != (len(images), 576) or not torch.isfinite(cams).all():
                raise ValueError("Invalid fixed 24x24 Grad-CAM")
    else:
        with torch.inference_mode():
            output = scorer.model(pixel_values=pixels, **inputs, use_itm_head=True)
        cams = torch.zeros((len(images), 576), device="cuda", dtype=torch.float32)
    # 步骤3：同次前向返回纯视觉表示；零空间质量标记无效，不悄悄用均匀图代替。
    features = output.last_hidden_state.detach().float()
    if features.shape[1] != 577 or not torch.isfinite(features).all():
        raise ValueError("Unexpected pure-visual patch representation")
    normalized = torch.nn.functional.normalize(features, dim=-1)
    scores = output.itm_score.detach().softmax(-1)[:, 1]
    arrays = dict(scores=scores.cpu().numpy(), cls=normalized[:, 0].cpu().numpy(),
                  patches=normalized[:, 1:].cpu().numpy(), gradcam=cams.detach().cpu().numpy())
    torch.cuda.synchronize()
    return arrays, dict(pixel_sha256=pixel_hash, token_ids=encoded["input_ids"][0].tolist(),
                        preprocess_seconds=pre_seconds, forward_backward_seconds=time.perf_counter() - started,
                        backward=spatial, spatial_layer_index=6 if spatial else None)


def compare_pair(cls_a, cls_b, patch_a, patch_b, cam_a, cam_b):
    """接口：固定源对的全图/等权patch/问题加权patch差异，只作输入层诊断。

    对应位置比较适用于固定画布；视角/移动不自动配准，低质量空间图保留无效状态。
    """
    # 步骤1：检查归一化输入和空间非负有限值，防止数值差异冒充语义增量。
    for vectors in (np.asarray(cls_a)[None], np.asarray(cls_b)[None], patch_a, patch_b):
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=-1), 1, atol=1e-5):
            raise ValueError("Invalid normalized visual vectors")
    if patch_a.shape != patch_b.shape or patch_a.shape[0] != 576:
        raise ValueError("Patch grid changed")
    cams = [np.asarray(c, dtype=np.float64) for c in (cam_a, cam_b)]
    if any(c.shape != (576,) or not np.isfinite(c).all() or np.any(c < 0) for c in cams):
        raise ValueError("Invalid spatial map")
    different = 1 - np.clip(np.sum(patch_a * patch_b, axis=-1), 0, 1)
    result = dict(cls_difference=float(1 - np.clip(np.dot(cls_a, cls_b), 0, 1)),
                  uniform_patch_difference=float(np.mean(different)), spatial_valid=all(c.sum() > 1e-12 for c in cams))
    # 步骤2：归一化关注图对称平均，不根据人工目标圈区域或挑Token。
    result["query_patch_difference"] = (float(np.sum(((cams[0]/cams[0].sum() + cams[1]/cams[1].sum())/2) * different))
                                           if result["spatial_valid"] else None)
    return result
