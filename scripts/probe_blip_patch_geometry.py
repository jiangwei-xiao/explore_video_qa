"""固定帧库的CLS与局部patch几何探针；不改变正式选帧或运行问答。"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256, offline_environment
from videoqa_full.state import durable, exclusive, utc
from videoqa_audit.query_input_probe import BASE
from videoqa_budget32.frames import FrameProvider

CASES = {'409-3': [67, 69, 77, 80, 94, 99], '556-1': [16, 17, 41, 42, 43, 47, 48, 49],
         '383-2': [941, 942, 943, 944, 945, 947, 948, 949],
         '395-2': [270, 271, 272, 273, 274, 275, 276], '538-1': [10, 11, 12, 13]}
RUN = ROOT / 'outputs/diagnostics/blip_patch_geometry_20261004_r2'
MODEL = Path('/home/models/Salesforce/blip-itm-base-coco')


def invoke(scorer, images, name):
    """接口：视觉前向开始即记账；未知调用不重复，完成项只核对特征文件。"""
    import torch
    path = RUN / 'calls' / (name + '.json')
    asset = RUN / 'features' / (name + '.npy')
    fingerprint = sha256(RUN / 'protocol.json')
    if path.exists():
        record = read_json(path)
        if record['status'] != 'completed' or record['fingerprint'] != fingerprint or sha256(asset) != record['sha256']:
            raise ValueError('Unknown call or changed feature asset')
        return np.load(asset, allow_pickle=False)
    if not 1 <= len(images) <= 16:
        raise ValueError('Batch limit exceeded')
    pixels = scorer.processor(images=images, return_tensors='pt')['pixel_values'].cuda().float()
    durable(path, dict(status='started', fingerprint=fingerprint, frames=len(images), utc=utc()))
    with torch.inference_mode():
        h = scorer.model.vision_model(pixel_values=pixels, return_dict=True).last_hidden_state.float()
        if not torch.isfinite(h).all() or torch.any(torch.linalg.vector_norm(h, dim=-1) <= 1e-12):
            raise ValueError('Invalid spatial features')
        features = torch.nn.functional.normalize(h, dim=-1).cpu().numpy()
    torch.cuda.synchronize()
    asset.parent.mkdir(parents=True, exist_ok=True)
    temp = asset.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.save(stream, features, allow_pickle=False)
    os.replace(temp, asset)
    durable(path, dict(status='completed', fingerprint=fingerprint, frames=len(images),
                       shape=list(features.shape), sha256=sha256(asset), utc=utc()))
    return features


def main():
    """接口：33张真实源帧、2张合成图；只比较视觉几何，不运行ITM/文本/问答。"""
    offline_environment()
    from videoqa_runtime.baseline_run import available_gpus
    from videoqa_runtime.baseline_selection import BlipItmScorer
    from videoqa_budget32.selection import configure_visual_precision
    import torch
    from PIL import Image
    with exclusive(RUN / 'run.lock'):
        # 步骤1：冻结图像身份、模型与精度；不根据输出增加裁剪、帧或距离公式。
        sources = {str(Path(__file__)): sha256(__file__)}
        for p in MODEL.iterdir():
            if p.is_file():
                sources[str(p)] = sha256(p)
        for q in CASES:
            for folder, suffix in [('pools', '.json'), ('features', '.npy')]:
                p = BASE / folder / (q + suffix);sources[str(p)] = sha256(p)
        spec = dict(sources=sources, cases=CASES, real_visual_frames=33, synthetic_visual_frames=2,
                    max_qa_calls=0, max_itm_calls=0, precision='FP32; both TF32 disabled',
                    patch_distance='1 - mean of two directional mean max clipped patch cosine matches',
                    limits='局部patch非对象标注；无空间匹配约束的视觉距离不等于语义或关系距离')
        pp = RUN / 'protocol.json'
        if pp.exists():
            if read_json(pp) != spec:
                raise ValueError('Protocol changed')
        else:
            durable(pp, spec)
        # 步骤2：完整恢复不加载模型，也不重算距离。
        if (RUN / 'summary.json').exists():
            summary = read_json(RUN / 'summary.json')
            for p, digest in summary['assets'].items():
                if sha256(p) != digest:
                    raise ValueError('Completed asset changed')
            print('Completed resume: no new forward', flush=True)
            return
        available, gpu_states = available_gpus(['1'])
        if available != ['1']:
            raise RuntimeError('GPU1 is not available')
        durable(RUN/'gpu_preflight.json', dict(available=available, states=gpu_states,
                                              pid=os.getpid(), sid=os.getsid(0), utc=utc()))
        configure_visual_precision(torch)
        torch.set_num_threads(2)
        scorer = BlipItmScorer(MODEL)
        synth = [Image.new('RGB', (384, 384), (60, 80, 120)) for _ in range(2)]
        control = invoke(scorer, synth, 'synthetic')
        if np.max(np.abs(control[0]-control[1])) > 1e-6:
            raise ValueError('Identical image control mismatch')
        for im in synth: im.close()
        records = []
        # 步骤3：一次视觉编码同时取CLS与全部patch，核对旧CLS，再计算唯一固定距离。
        for q, indices in CASES.items():
            pool = read_json(BASE / 'pools' / (q + '.json'))
            frames = [pool['candidates'][i] for i in indices]
            images, rgb = FrameProvider(dict(video=pool['video'], selected_frames=frames)).decode()
            h = invoke(scorer, images, q)
            for im in images: im.close()
            old = np.load(BASE / 'features' / (q + '.npy'), allow_pickle=False)[indices]
            error = float(np.max(np.abs(h[:, 0]-old)))
            if error > 1e-6:
                raise ValueError(f'Historical CLS mismatch: {q}: {error}')
            cls = 1-np.clip(h[:, 0] @ h[:, 0].T, 0, 1)
            patch = torch.from_numpy(h[:, 1:]).cuda()
            distances = np.zeros((len(indices), len(indices)), dtype=np.float64)
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    sims = torch.clamp(patch[i] @ patch[j].T, 0, 1)
                    d = 1-float((sims.max(0).values.mean()+sims.max(1).values.mean()).item()/2)
                    distances[i, j] = distances[j, i] = d
            records.append(dict(question_id=q, indices=indices, frames=frames, rgb_sha256=rgb,
                                cls_error=error, cls_distance=cls.tolist(), patch_distance=distances.tolist()))
            print(q, 'CLS error', error, flush=True)
        # 步骤4：只输出能力诊断矩阵，不生成候选选择或附用历史答案。
        durable(RUN / 'analysis.json', dict(records=records, new_qa_calls=0, new_itm_calls=0))
        assets = {str(p):sha256(p) for folder in ['calls', 'features'] for p in (RUN/folder).iterdir()}
        assets[str(RUN/'analysis.json')] = sha256(RUN/'analysis.json')
        durable(RUN/'summary.json', dict(status='completed', real_frames=33, synthetic_frames=2,
                                        qa_calls=0, itm_calls=0, assets=assets, utc=utc()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        durable(RUN/'failure.json', dict(error=repr(exc), utc=utc()))
        raise
