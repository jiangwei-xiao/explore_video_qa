"""仅CPU重建292-3历史Qwen输入，核验两个时间端点的真实预处理可见性。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,QWEN_PATH
from videoqa_budget32.frames import FrameProvider
from videoqa_budget32.observe import tensor_hash


def main(run):
    """接口：核对完整历史RGB和像素Tensor后逆变换显示；不加载问答模型权重。"""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    torch.set_num_threads(2)
    source=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1/imported/results/RD-1.2-exp12/292-3.json'
    result=read_json(source);observed=result['answer']['observed']
    # 步骤1：源帧与完整预处理Tensor逐字节复现，不能凭原图可读替代模型输入核验。
    images,hashes=FrameProvider(result['selection']).decode()
    if hashes!=result['rgb_sha256']:raise ValueError('原输入RGB身份不一致')
    processor=AutoProcessor.from_pretrained(QWEN_PATH,local_files_only=True)
    ip=processor.image_processor
    if (ip.patch_size,ip.temporal_patch_size,ip.merge_size)!=(14,2,2):raise ValueError('视觉patch配置变化')
    data=ip(images=None,videos=[np.stack([np.asarray(im.convert('RGB')) for im in images])],
            size=dict(shortest_edge=100352,longest_edge=602112),return_tensors='pt')
    for im in images:im.close()
    if tensor_hash(data['pixel_values_videos'])!=observed['processor_pixel_sha256']:
        raise ValueError('历史像素Tensor不一致')
    if data['video_grid_thw'].tolist()!=observed['video_grid_thw']:raise ValueError('历史视觉网格不一致')
    # 步骤2：逆排官方patch布局并反归一化，只显示冻结的两个目标时刻。
    t,h,w=data['video_grid_thw'][0].tolist()
    patches=data['pixel_values_videos'].numpy().reshape(t,h//2,w//2,2,2,3,2,14,14)
    frames=patches.transpose(0,6,5,1,3,7,2,4,8).reshape(t*2,3,h*14,w*14)
    records=[];run.mkdir(parents=True,exist_ok=True)
    for target in [55.267,57.267]:
        metadata=result['selection']['selected_frames']
        i=min(range(len(metadata)),key=lambda j:abs(metadata[j]['timestamp_seconds']-target))
        if abs(metadata[i]['timestamp_seconds']-target)>.003:raise ValueError('目标时刻不在历史输入')
        rgb=np.clip((frames[i].transpose(1,2,0)*np.asarray(ip.image_std)+np.asarray(ip.image_mean))*255,0,255).round().astype(np.uint8)
        path=run/f'292-3_{int(target)}_model_view.png';Image.fromarray(rgb).save(path)
        records.append(dict(source_frame=metadata[i],input_position=i,size=[w*14,h*14],sha256=sha256(path)))
    # 步骤3：只记录可见性依据，不推断模型内部是否正确读取或完成运算。
    output=dict(source_sha256=sha256(source),code_sha256=sha256(__file__),records=records,
        pixel_sha256=observed['processor_pixel_sha256'],new_model_calls=0)
    path=run/'visibility.json'
    if path.exists() and read_json(path)!=output:raise ValueError('可见性恢复结果改变')
    if not path.exists():durable(path,output)
    print('Exact historical tensor verified',records)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    main(parser.parse_args().run)
