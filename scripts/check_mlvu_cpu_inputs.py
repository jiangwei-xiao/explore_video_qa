"""MLVU真实源帧的CPU接口验收，不加载模型权重或生成答案。"""
from pathlib import Path
import argparse
import sys
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from videoqa_runtime.common import read_json
from videoqa_full.state import durable, digest
from videoqa_budget32.frames import native_frames, FrameProvider
from videoqa_mlvu.data import rows, PROFILES
from videoqa_mlvu.backends import cpu_identity
from videoqa_mlvu.runtime import initialize_cpu, source_order


def main():
    """接口：按冻结顺序选择首个已索引真实视频，仅验证像素/文字/JSON身份。"""
    p = argparse.ArgumentParser(); p.add_argument('--run-id', required=True); p.add_argument('--profile', choices=list(PROFILES), required=True)
    a = p.parse_args(); root = ROOT / 'outputs/full_videoqa' / a.run_id
    row = next(r for r in rows(root) if (root / 'indices' / (r['video_id']+'.json')).exists())
    record = read_json(root / 'indices' / (row['video_id']+'.json'))
    selected = native_frames(np.load(record['pts_path'], allow_pickle=False), record['video'], PROFILES[a.profile])
    images, rgb = FrameProvider(dict(video=record['video'], selected_frames=selected)).decode()
    try:
        # 步骤1：官方CPU预处理，不假设两模型Tensor相同，不调用视觉或语言前向。
        initialize_cpu(a.profile)
        from videoqa_mlvu import runtime
        identity = cpu_identity(a.profile, runtime._TOOLS, images, row)
        identity.update(source=source_order(dict(selected_frames=selected)), rgb_sha256=rgb)
        import json
        if identity != json.loads(json.dumps(identity)): raise ValueError('JSON identity container differs')
        # 步骤2：记录真实预算、完整提示和像素，明确只是输入验收，不是回答成绩。
        value = dict(status='passed', profile=a.profile, question_id=row['question_id'], source_frames=len(selected),
            input_identity=identity, fingerprint=digest(identity), new_model_calls=0)
        durable(root / 'cpu_acceptance' / (a.profile+'.json'), value)
        print(dict(status='passed', profile=a.profile, source_frames=len(selected), text_tokens=len(identity['input_token_ids']), new_model_calls=0), flush=True)
    finally:
        for image in images: image.close()


if __name__ == '__main__': main()
