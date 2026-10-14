"""有界复核的真实处理器视图：反变换已有输入像素，不执行模型前向。"""
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import argparse
from pathlib import Path
import sys
import re
import numpy as np
import torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable
from videoqa_budget32.frames import FrameProvider
from videoqa_budget32.observe import tensor_hash
from videoqa_scorer_comparison.runtime import make_tools


def unpack_video(pixels, grid, processor):
    """接口：逆转固定官方Qwen patch排列，保留时间补齐与空间尺寸，不另行缩放。"""
    t, h, w = map(int, grid); p = processor.patch_size; m = processor.merge_size
    temporal = processor.temporal_patch_size
    # 步骤1：官方transpose(0,3,6,4,7,2,1,5,8)的逆排列。
    return pixels.reshape(t, h // m, w // m, m, m, 3, temporal, p, p).transpose(
        0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(t * temporal, 3, h * p, w * p)


def export(root, profile):
    """接口：只导出三道已复核关键题的S2 exp07实际模型像素，摘要必须与QA观测一致。"""
    processor, _ = make_tools(profile); records = []
    targets = {'103-1': 5.756, '383-2': 944.277}
    if profile == 'qwen32': targets['395-2'] = 277.250
    for q, target in targets.items():
        path = root / 'qa' / profile / 'results/RD-1.2-exp07' / (q + '.json')
        item = read_json(path); images, rgb = FrameProvider(item['selection']).decode()
        try:
            # 步骤1：精确原源帧重新走官方CPU预处理；禁止任何模型调用或新采样。
            if rgb != item['rgb_sha256']: raise ValueError('Source RGB changed')
            if profile == 'qwen32':
                output = processor(text=['<|vision_start|><|video_pad|><|vision_end|>'],
                    videos=[np.stack([np.asarray(x) for x in images])], fps=2.,
                    images_kwargs={'size': dict(shortest_edge=100352, longest_edge=602112)},
                    truncation=False, return_tensors='pt')
                pixels = output['pixel_values_videos']
                expected = item['answer']['observed']['processor_pixel_sha256']
                frames = unpack_video(pixels.numpy(), output['video_grid_thw'][0].tolist(), processor.image_processor)
                mean, std = processor.image_processor.image_mean, processor.image_processor.image_std
            else:
                pixels = processor.preprocess(images, return_tensors='pt')['pixel_values'].to(torch.bfloat16)
                expected = item['answer']['observed']['pixel_sha256']
                frames = pixels.float().numpy(); mean, std = processor.image_mean, processor.image_std
            if tensor_hash(pixels) != expected: raise ValueError('Actual QA pixels changed')
            # 步骤2：仅反归一化为可见RGB；不把导出的8bit预览称为模型特征。
            index = min(range(len(images)), key=lambda i: abs(item['selection']['selected_frames'][i]['timestamp_seconds'] - target))
            a = frames[index].transpose(1, 2, 0)
            a = np.rint(np.clip(a * np.array(std) + np.array(mean), 0, 1) * 255).astype('uint8')
            dest = root / 'review/processed_views' / profile / (q + '.png'); dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(a).save(dest)
            records.append(dict(question_id=q, frame=item['selection']['selected_frames'][index],
                image=str(dest.relative_to(root)), sha256=sha256(dest), pixel_sha256=expected,
                result_sha256=sha256(path), source_rgb_sha256=rgb[index], width=a.shape[1], height=a.shape[0]))
        finally:
            for image in images: image.close()
    durable(root / 'review/processed_views' / (profile + '.json'), dict(records=records, new_model_calls=0))


def main():
    """接口：按问答环境运行指定配置，图像处理与语言/视觉推理严格隔离。"""
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True)
    parser.add_argument('--profile', choices=['qwen32', 'llava8'], required=True); args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id): raise ValueError('Invalid run ID')
    torch.set_num_threads(2)
    export(ROOT / 'outputs/full_videoqa' / args.run_id, args.profile)


if __name__ == '__main__': main()
