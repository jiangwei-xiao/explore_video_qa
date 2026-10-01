"""BLIP同次前向的两种表征诊断；不修改任何正式评分器或选帧入口。"""
import time
import numpy as np


def score_dual_embeddings(scorer,encoded,images):
    """接口：原问题和至多16张RGB→原ITM、纯视觉CLS、图文交互CLS及同步耗时。"""
    import torch
    if not 1<=len(images)<=16:raise ValueError('Frozen BLIP batch bound exceeded')
    # 步骤1：预处理、文本和精度与原FeatureScorer完全一致；新表征只作为诊断输出。
    torch.cuda.synchronize();started=time.perf_counter()
    pixels=scorer.processor(images=images,return_tensors='pt')['pixel_values'].to('cuda',torch.float32)
    inputs={k:v.repeat(len(images),1).cuda() for k,v in encoded.items() if k in ('input_ids','attention_mask')}
    torch.cuda.synchronize();pre=time.perf_counter()-started;started=time.perf_counter()
    with torch.inference_mode():
        output=scorer.model(pixel_values=pixels,**inputs,use_itm_head=True)
        scores=output.itm_score.softmax(-1)[:,1].cpu().tolist()
        arrays=[]
        # 步骤2：两个向量来自同次前向；交互CLS不是纯视觉特征，必须独立命名，不能覆盖旧资产。
        for cls in (output.last_hidden_state[:,0,:].float(),output.question_embeds[:,0,:].float()):
            if not torch.isfinite(cls).all() or torch.any(torch.linalg.vector_norm(cls,dim=-1)<=1e-12):raise ValueError('Invalid embedding')
            arrays.append(torch.nn.functional.normalize(cls,p=2,dim=-1).cpu().numpy())
    torch.cuda.synchronize();forward=time.perf_counter()-started
    if not np.isfinite(scores).all() or any(not 0<=v<=1 for v in scores):raise ValueError('Invalid ITM scores')
    return scores,arrays[0],arrays[1],pre,forward


def geometry_summary(vectors):
    """接口：只统计固定帧库的相似度分布，不把高/低余弦解释为证据充分性。"""
    x=np.asarray(vectors,dtype=np.float32)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-5,rtol=0):raise ValueError('Invalid normalized bank')
    # 步骤1：去除自身比较；步骤2：保留原余弦而不中心化或事后白化以追求更好分离。
    matrix=np.clip(x@x.T,-1,1);values=matrix[np.triu_indices(len(x),1)]
    return dict(pairs=len(values),cosine_quantiles=np.quantile(values,[0,.1,.5,.9,1]).tolist(),
        mean_cosine=float(values.mean()),interpretation='Within this fixed diagnostic bank only; no semantic or QA score')
