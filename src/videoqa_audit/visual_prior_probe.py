"""原生四选项分布对空白输入的变化度量，不使用正确答案索引。"""
import numpy as np


def distribution_shift(real_logits, blank_logits):
    """接口：四个原始logit→KL(real||blank)，单位nat；变化量不是正确性置信度。"""
    # 步骤1：明确只比较同一题同一选项顺序，拒绝非有限或错误维度。
    a, b = (np.asarray(x, dtype=np.float64) for x in (real_logits, blank_logits))
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected finite four-option logits')
    # 步骤2：稳定log-softmax，不靠平滑系数改变近零概率的贡献。
    def log_probs(x):
        x = x - x.max()
        return x - np.log(np.exp(x).sum())
    pa, pb = log_probs(a), log_probs(b)
    # 步骤3：只输出分布变化，不选取正确选项，不把变化等同视觉证据充分。
    return float(np.exp(pa) @ (pa - pb))
