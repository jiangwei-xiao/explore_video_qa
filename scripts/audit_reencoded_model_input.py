"""核对重编码控制中源帧改变但RGB相同的实际Qwen视觉输入，不运行模型。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,QWEN_PATH
from videoqa_audit.query_input_probe import BASE
from videoqa_budget32.frames import FrameProvider
from videoqa_budget32.observe import tensor_hash

PARENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'


def main(run):
    """接口：CPU官方视频预处理与历史Tensor比较，保留离散源帧差异，不降低容差。"""
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(2)
    control=read_json(run/'reencoded_control.json')
    if control['changed_rgb_ids']:raise ValueError('存在实际RGB变化，不能作为相同输入放行')
    processor=AutoProcessor.from_pretrained(QWEN_PATH,local_files_only=True);records=[]
    # 步骤1：仅处理离散选择不一致项，重新精确读取其全部有序RGB。
    for row in control['records']:
        if row['selection_equal']:continue
        q=row['question_id'];pool=read_json(BASE/'pools'/(q+'.json'))
        parent=read_json(PARENT/'imported/results/RD-1.2-exp12'/(q+'.json'))
        frames=[pool['candidates'][i] for i in row['new_selected']]
        images,hashes=FrameProvider(dict(video=pool['video'],selected_frames=frames)).decode()
        if hashes!=parent['rgb_sha256']:raise ValueError('有序RGB未复现')
        # 步骤2：Qwen协议不输入真实PTS文字且编码FPS恒为2，视觉Tensor及网格须完全一致。
        pixels=processor.image_processor(images=None,videos=[np.stack([np.asarray(im.convert('RGB')) for im in images])],
            size=dict(shortest_edge=100352,longest_edge=602112),return_tensors='pt')
        for image in images:image.close()
        observed=parent['answer']['observed']
        digest=tensor_hash(pixels['pixel_values_videos'])
        if digest!=observed['processor_pixel_sha256'] or pixels['video_grid_thw'].tolist()!=observed['video_grid_thw']:
            raise ValueError('实际视觉输入不一致')
        records.append(dict(question_id=q,source_indices_equal=False,ordered_rgb_equal=True,
            processor_pixel_sha256=digest,video_grid_thw=observed['video_grid_thw']))
    # 步骤3：只证明模型可见视觉输入等价，不声称200题源帧编号全部相同。
    result=dict(status='passed',code_sha256=sha256(__file__),control_sha256=sha256(run/'reencoded_control.json'),
        records=records,new_model_calls=0,scope='源帧差异保留在控制记录；不改算法、不重算特征、不附用新方法预测')
    path=run/'reencoded_input_equivalence.json'
    if path.exists() and read_json(path)!=result:raise ValueError('输入核验恢复变化')
    if not path.exists():durable(path,result)
    print('Verified equivalent visual inputs',len(records),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    main(parser.parse_args().run)
