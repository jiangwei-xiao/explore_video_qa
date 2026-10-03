"""证据偏好诊断的零模型输入核验：逐图预处理拼接与实际prefill逐字节比较。"""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_full.state import durable, QWEN_PATH
from videoqa_full.qwen_backend import MIN_PIXELS, MAX_PIXELS
from videoqa_budget32.frames import FrameProvider
from videoqa_budget32.observe import tensor_hash
from videoqa_runtime.common import read_json, sha256


def main():
    """接口：不加载权重、不新增生成；排除图像顺序、提示词、分辨率和模型输入传递错误。"""
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(2)
    run=ROOT/'outputs/diagnostics/evidence_binding_probe4_20261003_r2'
    spec=read_json(run/'protocol.json'); checks=[]
    processor=AutoProcessor.from_pretrained(QWEN_PATH,local_files_only=True,use_fast=False,
        min_pixels=MIN_PIXELS,max_pixels=MAX_PIXELS,size={'shortest_edge':MIN_PIXELS,'longest_edge':MAX_PIXELS})
    # 步骤1：从真PTS重新取帧，逐图独立预处理后按两种呈现顺序拼接。
    for q in spec['question_ids']:
        r=spec['records'][q]
        canonical=sorted(r['groups']['evidence']+r['groups']['contrast'],key=lambda f:f['source_pts'])
        images, hashes=FrameProvider(dict(video=r['video'],selected_frames=canonical)).decode()
        lookup={f['source_pts']:im for f,im in zip(canonical,images)}
        single={}
        for f,h,im in zip(canonical,hashes,images):
            assert h==r['source_rgb_sha256'][str(f['candidate_index'])]
            processed=processor.image_processor(images=[im],return_tensors='pt')
            single[f['source_pts']]=processed['pixel_values']
        for orientation in ['forward','reversed']:
            groups=['evidence','contrast'] if orientation=='forward' else ['contrast','evidence']
            frames=[f for g in groups for f in r['groups'][g]]
            raw=read_json(run/'raw'/orientation/(q+'.json'))
            pictures=[lookup[f['source_pts']] for f in frames]
            # 步骤2：使用已记录完整提示词重建Token，确保不是仅比较无文字的图像Tensor。
            inputs=processor(text=[raw['prompt']],images=pictures,
                images_kwargs={'size':{'shortest_edge':MIN_PIXELS,'longest_edge':MAX_PIXELS}},
                padding=False,truncation=False,return_tensors='pt')
            concatenated=torch.cat([single[f['source_pts']] for f in frames],dim=0)
            assert tensor_hash(concatenated)==tensor_hash(inputs['pixel_values'])==raw['processor_pixel_sha256']
            assert inputs['input_ids'][0].tolist()==raw['input_token_ids']
            assert inputs['image_grid_thw'].tolist()==raw['image_grid_thw']
            assert raw['measurements']['prefill_calls']==1 and raw['measurements']['checked_logit_steps']>0
            for t,h,w in raw['image_grid_thw']:
                assert t==1 and h%2==w%2==0 and MIN_PIXELS<=h*w*196<=MAX_PIXELS
            checks.append(dict(question_id=q,orientation=orientation,source_pts=[f['source_pts'] for f in frames],
                               actual_input_sha256=raw['processor_pixel_sha256'],tokens=len(raw['input_token_ids'])))
        for image in images:image.close()
    # 步骤3：工程通过不等于语义判断通过；位置偏好与提示词表述的局限保留。
    durable(run/'input_audit.json',dict(status='passed',checks=checks,new_model_calls=0,
        protocol_sha256=sha256(run/'protocol.json'),scope='Pixel order, single-image preprocessing, prompt tokens and model hook parity only'))
    print('Eight diagnostic inputs verified; new model calls 0',flush=True)


if __name__=='__main__':main()
