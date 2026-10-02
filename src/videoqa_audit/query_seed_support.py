"""多查询种子支持的独立诊断；不修改冻结选择器或把匹配值解释为答案概率。"""
import numpy as np


def query_supported_seed_weights(plan, query_scores):
    """接口：输入同粗池各查询分数，输出供原max支持算子使用的区内种子权重。

    每查询在本区种子内用峰值归一化，再跨查询取最大：
    max_q max_j w_qj*cos_+(i,j) = max_j(max_q w_qj)*cos_+(i,j)。
    这是候选假设的析取支持，不是各查询概率可比较或证据充分性的假设。
    """
    scores=np.asarray(query_scores,dtype=np.float64)
    if scores.ndim!=2 or not scores.shape[0] or not scores.shape[1] or not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():
        raise ValueError('Expected finite query by coarse-frame probabilities')
    weights=np.zeros(scores.shape[1],dtype=np.float64)
    seen=set()
    trace={}
    # 步骤1：种子必须属于互不重叠的区域；不为非种子创建保留资格。
    for region in plan['regions']:
        ids=region['seed_indices']
        if not ids or len(set(ids))!=len(ids) or min(ids)<0 or max(ids)>=scores.shape[1] or seen.intersection(ids):
            raise ValueError('Invalid or overlapping regional seeds')
        seen.update(ids)
        local=scores[:,ids]
        peaks=local.max(axis=1)
        # 步骤2：全零查询不提供相对支持；全部查询全零才沿用均匀支持回退。
        normalized=np.divide(local,peaks[:,None],out=np.zeros_like(local),where=peaks[:,None]>0)
        combined=normalized.max(axis=0) if np.any(peaks>0) else np.ones(len(ids))
        weights[ids]=combined
        trace[str(region['region_id'])]=dict(seed_indices=ids,query_peaks=peaks.tolist(),
            per_query_relative=normalized.tolist(),combined_weights=combined.tolist(),all_zero_fallback=not bool(np.any(peaks>0)))
    # 步骤3：非种子零值仅为兼容原函数输入长度，不改变候选、锚点或覆盖参考集合。
    return weights.tolist(),trace
