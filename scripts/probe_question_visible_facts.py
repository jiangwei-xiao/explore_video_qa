"""原问题条件下的可见事实能力检查；固定六帧与空白对照，不接入选帧。"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256, offline_environment
from videoqa_full.state import durable, exclusive, utc
from videoqa_audit.query_input_probe import BASE
from videoqa_budget32.frames import FrameProvider
import probe_visible_facts as original

RUN = ROOT / 'outputs/diagnostics/question_visible_facts6_20261004_r1'
CASES = original.CASES
PROMPT = ('Inspect only this single image. The video question below is context, not a fact about the image. '
          'Describe the directly visible objects, actions, or spatial relationships that bear on the question. '
          'If the requested relationship is not visible, explicitly say it cannot be established from this image. '
          'Do not answer the video question, infer unseen events, or invent details from its wording. '
          'Use at most two concise sentences.\n\nVideo question:\n{question}')


def main():
    """接口：六张固定原帧、每题一张空白和一次合成；辅助描述上限10次，无正式QA。"""
    from videoqa_budget32.run import verify
    with exclusive(RUN / 'run.lock'):
        verify(original.SOURCE)
        manifest = ROOT / 'data/manifests/video_mme_full_2700.json'
        questions = {r['question_id']: r['question'] for r in read_json(manifest)['rows']}
        pools = {q: read_json(BASE / 'pools' / (q + '.json')) for q, _ in CASES}
        # 步骤1：在生成前冻结全部真图和空白条件，提示中不含选项、答案或目标物体提示。
        files = [Path(__file__), Path(original.__file__), manifest, original.SOURCE / 'protocol.json']
        files += [BASE / 'pools' / (q + '.json') for q in pools]
        spec = dict(sources={str(p): sha256(p) for p in files}, prompt=PROMPT,
                    cases=[dict(question_id=q, candidate_index=i, source_frame=pools[q]['candidates'][i]) for q, i in CASES],
                    questions={q: questions[q] for q in pools}, max_real_image_descriptions=6,
                    max_blank_descriptions=3, max_synthetic_descriptions=1, max_final_qa=0,
                    max_new_tokens=96, scope='事后固定能力面板；不生成候选排序或模型正式答案')
        protocol = RUN / 'protocol.json'
        if protocol.exists():
            if read_json(protocol) != spec: raise ValueError('Probe protocol changed')
        else: durable(protocol, spec)
        fingerprint = sha256(protocol)
        jobs = [(q, i, 'real') for q, i in CASES] + [(q, next(i for qq, i in CASES if qq == q), 'blank') for q in pools]
        def output(q, i, kind):
            return RUN / 'calls' / (f'{kind}_{q}_{i}.json')
        for q, i, kind in jobs:
            path = output(q, i, kind)
            if path.exists():
                prior = read_json(path)
                if prior['status'] != 'completed' or prior['identity']['protocol_sha256'] != fingerprint:
                    raise ValueError('Uncertain call must not be repeated')
        pending = [(q, i, k) for q, i, k in jobs if not output(q, i, k).exists()]
        # 步骤2：完整恢复不加载模型；正式环境和单图观察路径复用旧描述探针。
        if pending:
            from videoqa_runtime.baseline_run import available_gpus
            free, states = available_gpus(['1'])
            if '1' not in free: raise RuntimeError('GPU1 not idle')
            offline_environment()
            import torch
            from PIL import Image
            from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            durable(RUN/'status.json', dict(status='loading', pid=os.getpid(), sid=os.getsid(0), gpu_states=states, utc=utc()))
            backend = QwenFA2VideoBackend()
            original.PROMPT = PROMPT.format(question='Which colors are visible?')
            synth = Image.new('RGB', (640, 360), 'red')
            original.describe(backend, synth, RUN/'synthetic.json', dict(kind='synthetic', protocol_sha256=fingerprint))
            synth.close()
            # 步骤3：先完成全部真图，再完成每题空白；每次独立会话，不反馈前一次描述。
            for q, i, kind in pending:
                pool = pools[q]
                images, hashes = FrameProvider(dict(video=pool['video'], selected_frames=[pool['candidates'][i]])).decode()
                image = images[0]
                if kind == 'blank':
                    size = image.size;image.close();image = Image.new('RGB', size, 'white')
                original.PROMPT = PROMPT.format(question=questions[q])
                durable(RUN/'status.json', dict(status='running', question_id=q, kind=kind,
                                              pid=os.getpid(), sid=os.getsid(0), utc=utc()))
                original.describe(backend, image, output(q, i, kind), dict(question_id=q, candidate_index=i,
                    kind=kind, protocol_sha256=fingerprint, source_rgb_sha256=hashes[0],
                    input_content='source_frame' if kind=='real' else 'uniform_white_control'))
                image.close()
        # 步骤4：只记录原文与调用数；空白能否拒绝臆测和真实事实是否正确由画面核对。
        results = [read_json(output(q,i,k)) for q,i,k in jobs]
        durable(RUN/'summary.json', dict(real_image_calls=6, blank_calls=3, synthetic_calls=1,
            new_final_qa=0, records=[dict(identity=r['identity'], text=r['raw_text'], normal_eos=r['normal_eos']) for r in results]))
        durable(RUN/'status.json', dict(status='completed', pid=os.getpid(), sid=os.getsid(0), utc=utc()))


if __name__ == '__main__':
    try: main()
    except BaseException as exc:
        durable(RUN/'status.json', dict(status='paused', error=repr(exc), utc=utc()))
        raise
