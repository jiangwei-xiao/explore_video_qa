"""两题同预算人工修复诊断准备：只有CPU官方预处理，不加载模型、不生成答案。"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_audit.query_input_probe import BASE, FULL
from videoqa_budget32.frames import FrameProvider, validate_frames
from videoqa_budget32.observe import tensor_hash
from videoqa_full.state import durable, exclusive, QWEN_PATH
from videoqa_runtime.common import read_json, sha256

REPAIRS = {
    '409-3': dict(add=[77, 80], remove_pts=[11438427, 14123109]),
    '556-1': dict(add=[42], remove_pts=[3517514]),
}


def prepare(run):
    """接口：冻结两题原输入与修复输入、来源哈希和4次真实调用上限，恢复不覆盖已有准备。"""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    torch.set_num_threads(2)
    run = Path(run)
    with exclusive(run / 'prepare.lock'):
        manifest_path = run / 'input_manifest.json'
        if manifest_path.exists():
            spec = read_json(manifest_path)
            for path, digest in spec['sources'].items():
                assert sha256(path) == digest, path
            for rel, digest in spec['files'].items():
                assert sha256(run / rel) == digest, rel
            print('Prepared inputs verified; new model calls 0', flush=True)
            return spec
        processor = AutoProcessor.from_pretrained(QWEN_PATH, local_files_only=True)
        ip = processor.image_processor
        assert (ip.patch_size, ip.temporal_patch_size, ip.merge_size) == (14, 2, 2)
        sources = {str(Path(__file__).resolve()): sha256(__file__)}
        files, observations = {}, {}
        # 步骤1：只操作固定两题原有效池。每次加入与移除一一对应，其余帧不变。
        for q, change in REPAIRS.items():
            result_path = FULL / 'imported/results/RD-1.2-exp07' / (q + '.json')
            pool_path = BASE / 'pools' / (q + '.json')
            sources.update({str(p): sha256(p) for p in [result_path, pool_path]})
            old, pool = read_json(result_path), read_json(pool_path)
            assert old['gpu'] == '1'
            original = old['selection']['selected_frames']
            assert len(original) == 32 and len(change['add']) == len(change['remove_pts']) <= 4
            assert set(change['remove_pts']) <= {f['source_pts'] for f in original}
            additions = [pool['candidates'][i] for i in change['add']]
            assert all(i < pool['initial_count'] for i in change['add'])
            assert not ({f['source_pts'] for f in additions} & {f['source_pts'] for f in original})
            repaired = sorted([f for f in original if f['source_pts'] not in change['remove_pts']] + additions,
                              key=lambda f: (f['source_pts'], f['candidate_index']))
            observations[q] = dict(historical_result=str(result_path), historical_gpu=old['gpu'],
                                   replacements=change, conditions={})
            for condition, frames in [('original', original), ('repair', repaired)]:
                selection = dict(old['selection'], selected_frames=frames)
                validate_frames(selection)
                assert len(frames) == 32
                images, hashes = FrameProvider(selection).decode()
                if condition == 'original':
                    assert hashes == old['rgb_sha256']
                # 步骤2：通过官方图像处理器重建完整视频Tensor，原条件须逐字节复现历史。
                try:
                    data = ip(images=None, videos=[np.stack([np.asarray(im.convert('RGB')) for im in images])],
                              size=dict(shortest_edge=100352, longest_edge=602112), return_tensors='pt')
                finally:
                    for im in images:
                        im.close()
                digest = tensor_hash(data['pixel_values_videos'])
                grid = data['video_grid_thw'].tolist()
                old_observation = old['answer']['observed']
                assert grid == old_observation['video_grid_thw']
                if condition == 'original':
                    assert digest == old_observation['processor_pixel_sha256']
                observations[q]['conditions'][condition] = dict(processor_pixel_sha256=digest, video_grid_thw=grid)
                rel = f'selections/{condition}/{q}.json'
                durable(run / rel, dict(question_id=q, method=condition, selection=selection,
                                       rgb_sha256=hashes, scope='human-directed diagnostic, not algorithm output'))
                files[rel] = sha256(run / rel)
                # 步骤3：新增帧从实际处理后的时间patch逆排，仅生成可见性审核图，不修改模型输入。
                if condition == 'repair':
                    t, h, w = grid[0]
                    patches = data['pixel_values_videos'].numpy().reshape(t, h // 2, w // 2, 2, 2, 3, 2, 14, 14)
                    pixels = patches.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(t * 2, 3, h * 14, w * 14)
                    for i, frame in enumerate(frames):
                        if frame['source_pts'] not in {f['source_pts'] for f in additions}:
                            continue
                        rgb = np.clip((pixels[i].transpose(1, 2, 0) * np.asarray(ip.image_std) +
                                       np.asarray(ip.image_mean)) * 255, 0, 255).round().astype(np.uint8)
                        name = f'visibility/{q}_{frame["source_pts"]}.png'
                        dest = run / name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(rgb).save(dest)
                        files[name] = sha256(dest)
        # 步骤4：先完成CPU输入冻结；问答入口必须另验可见性和调用账本，不能自动生成。
        spec = dict(status='prepared_pending_visibility_review', question_ids=list(REPAIRS),
                    conditions=['original', 'repair'], method_reference='RD-1.2-exp07',
                    max_real_generations=4, frame_budget=32, physical_gpu='1',
                    new_model_calls=0, sources=sources, files=files, observations=observations,
                    scope='Two selected failures; no algorithm accuracy claim; original replay mismatch suspends attribution')
        durable(manifest_path, spec)
        print('Prepared four inputs; visibility review pending; new model calls 0', flush=True)
        return spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    prepare(parser.parse_args().run)
