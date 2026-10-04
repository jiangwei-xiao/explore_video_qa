"""Qwen冻结时间patch权重与exp12实际帧配对审计，不运行模型前向。"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import sys
import hashlib
import inspect
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE,FULL


def main():
    """接口：读取一层权重和200题冻结配对，区域不同不等于语义无关。"""
    import torch
    from safetensors import safe_open
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionPatchEmbed
    torch.set_num_threads(2)
    model=Path('/home/models/Qwen/Qwen2.5-VL-7B-Instruct')
    manifest=ROOT/'data/manifests/video_mme_development_200_20260920.json'
    parent=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
    index=model/'model.safetensors.index.json';key='visual.patch_embed.proj.weight'
    source=Path(inspect.getfile(Qwen2_5_VisionPatchEmbed))
    sources={str(p):sha256(p) for p in [Path(__file__),manifest,index,model/'config.json',source]}
    # 步骤1：只读取冻结首层张量，不加载模型、不改权重或视觉协议。
    shard=model/read_json(index)['weight_map'][key]
    with safe_open(str(shard),framework='pt',device='cpu') as handle:
        raw=handle.get_tensor(key)
    w=raw.float()
    if tuple(w.shape)!=(1280,3,2,14,14):raise ValueError('Unexpected temporal patch geometry')
    a,b=w[:,:,0],w[:,:,1]
    weights=dict(key=key,shape=list(w.shape),dtype=str(raw.dtype),
        tensor_sha256=hashlib.sha256(raw.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        implementation_sha256=sha256(source),temporal_cosine=float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0)),
        difference_to_sum_frobenius_ratio=float((a-b).norm()/(a+b).norm()),exact_equal=torch.equal(a,b),
        algebra='e=(W0+W1)*mean(x0,x1)+(W0-W1)*half_difference(x0,x1)',
        limitation='Frobenius比值不是实际激活贡献或信息损失比例，后续Transformer仍可利用差分')
    records=[]
    # 步骤2：按真实最终输入顺序两两配对，不按候选编号或原区域编号重排输入。
    for row in read_json(manifest)['rows']:
        q=row['question_id'];paths=dict(pool=BASE/'pools'/(q+'.json'),features=BASE/'features'/(q+'.npy'),
            trace=FULL/'traces'/(q+'.json'),selection=parent/'selections/RD-1.2-exp12'/(q+'.json'))
        sources.update({str(p):sha256(p) for p in paths.values()})
        pool=read_json(paths['pool']);item=read_json(paths['selection']);trace=read_json(paths['trace'])
        frames=item['selection']['selected_frames'];indices=item['selected_indices']
        if len(frames)!=32 or len(indices)!=32:raise ValueError('Fixed budget changed')
        owners={i:r['region_id'] for r in trace['regions'] for i in r['members']}
        f=np.load(paths['features'],allow_pickle=False)
        pairs=[]
        for j in range(0,32,2):
            left,right=indices[j:j+2];lf,rf=frames[j:j+2]
            if (pool['candidates'][left]['source_pts'],pool['candidates'][right]['source_pts'])!=(lf['source_pts'],rf['source_pts']):
                raise ValueError('Feature and actual source frame mismatch')
            pairs.append(dict(position=j//2,left=lf,right=rf,
                same_selection_region=owners[left]==owners[right],
                seconds_apart=rf['timestamp_seconds']-lf['timestamp_seconds'],
                cls_cosine=float(np.clip(f[left]@f[right],-1,1))))
        records.append(dict(question_id=q,pairs=pairs))
    # 步骤3：报告全部配对及描述性分布，不能根据旧对错断言配对是原因。
    pairs=[p for r in records for p in r['pairs']]
    cross=[p for p in pairs if not p['same_selection_region']]
    if len(records)!=200 or len(pairs)!=3200:raise ValueError('Pair population changed')
    summary=dict(questions=200,pairs=3200,cross_selection_region_pairs=len(cross),
        quantile_levels=[0,.25,.5,.75,.95,1],
        time_gap_quantiles=np.quantile([p['seconds_apart'] for p in pairs],[0,.25,.5,.75,.95,1]).tolist(),
        cls_cosine_quantiles=np.quantile([p['cls_cosine'] for p in pairs],[0,.25,.5,.75,.95,1]).tolist(),
        new_model_forwards=0,new_answers=0)
    result=dict(sources=sources,weights=weights,records=records,summary=summary,
        scope='编码结构诊断，不把跨选择区域当作跨语义事件，不据此改变帧顺序、Token或预算')
    path=ROOT/'outputs/analysis/temporal_patch_pairs200_20261004_r1/audit.json'
    if path.exists():
        if read_json(path)!=result:raise ValueError('Audit source or result changed')
    else:path.parent.mkdir(parents=True,exist_ok=True);durable(path,result)
    print(weights,flush=True);print(summary,flush=True)


if __name__=='__main__':main()
