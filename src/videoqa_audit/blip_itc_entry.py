"""官方BLIP图文对比检索探针；不是帧间投影支持或答案自评分。"""
import numpy as np


def rank_itc_candidates(candidates, scores, count):
    """接口：有限[-1,1]余弦按分数/PTS/编号排序，不假装为ITM概率。"""
    from videoqa_methods.density_selection import frame_key
    values = np.asarray(scores)
    if values.shape != (len(candidates),) or not len(values) or count < 1:
        raise ValueError('Invalid ITC ranking population')
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1+1e-6):
        raise ValueError('Invalid image-text cosine')
    # 步骤1：只比较同问题的相对排序；同分按真实源时间稳定决定。
    return sorted(range(len(values)), key=lambda i: (-float(values[i]), *frame_key(candidates[i])))[:min(count,len(values))]


def original_question_itc_embedding(scorer, question):
    """接口：原问题→官方ITC归一化文本投影，显式无图像交互/截断/生成。"""
    import torch
    encoded = scorer.tokenize(question)
    inputs = {k:v.cuda() for k,v in encoded.items() if k in ('input_ids','attention_mask')}
    # 步骤1：与HF use_itm_head=False分支相同，复用原Tokenizer而不加入选项。
    with torch.inference_mode():
        hidden = scorer.model.text_encoder(**inputs, return_dict=True).last_hidden_state[:,0]
        feature = torch.nn.functional.normalize(scorer.model.text_proj(hidden), dim=-1)
    # 步骤2：不使用帧间余弦代替图文余弦；返回的256维文本向量用于已验证图像投影。
    if not torch.isfinite(feature).all() or feature.shape != (1,256):
        raise ValueError('Invalid official ITC text feature')
    return feature[0].cpu().numpy(), encoded['input_ids'][0].tolist()


def score_itc_entry(features, text_feature, projection_parameters):
    """接口：冻结归一化CLS→已验证官方带偏置投影→原题余弦，返回评分与恢复尺度。

    不把旧归一化CLS直接送入带偏置Linear；复用通过310帧1e-6验证的代数恢复。
    """
    from videoqa_audit.blip_projection import recover_projected_features
    # 步骤1：恢复原post-LayerNorm向量尺度，输出官方图文对比图像向量。
    projected, scales = recover_projected_features(features, projection_parameters)
    text_feature = np.asarray(text_feature,dtype=np.float32)
    if text_feature.shape != (256,) or not np.isfinite(text_feature).all() or not np.isclose(np.linalg.norm(text_feature),1,atol=1e-5):
        raise ValueError('Invalid normalized text projection')
    # 步骤2：以训练过的图像-文本两侧投影计算原始余弦，不做阈值或概率变换。
    return projected @ text_feature, scales
