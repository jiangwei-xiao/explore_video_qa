"""两题四次人工取证诊断；复用正式Qwen生成与账本，不能当作算法成绩。"""
import argparse
import os
from pathlib import Path
import sys
import time

os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_audit.query_input_probe import BASE
from videoqa_budget32.run import verify, process, validate, recover_call
from videoqa_full.state import durable, exclusive, rows, utc
from videoqa_runtime.common import read_json, sha256, offline_environment


def execute(run):
    """接口：只允许已审核的四份输入，同题同卡同进程；状态不确定时停止而非再次生成。"""
    run = Path(run)
    with exclusive(run / 'coordinator.lock'):
        durable(run / 'launcher_identity.json', dict(pid=os.getpid(), sid=os.getsid(0), utc=utc()))
        manifest_path = run / 'input_manifest.json'
        inputs = read_json(manifest_path)
        review = read_json(run / 'visibility_review.json')
        assert review['status'] == 'passed' and review['input_manifest_sha256'] == sha256(manifest_path)
        assert inputs['question_ids'] == ['409-3', '556-1'] and inputs['max_real_generations'] == 4
        for path, digest in inputs['sources'].items():
            assert sha256(path) == digest, path
        for rel, digest in inputs['files'].items():
            assert sha256(run / rel) == digest, rel
        # 步骤1：继承已验收协议，不改变采样、精度、任务提示词、解析或生成参数。
        if not (run / 'protocol.json').exists():
            parent = verify(BASE)
            spec = dict(parent, run_id=run.name, methods=['original', 'repair'],
                        question_ids=inputs['question_ids'], max_calls=4, max_calls_per_method=2,
                        allowed_gpus=['1'], diagnostic=True,
                        selection_exports={k: v for k, v in inputs['files'].items() if k.startswith('selections/')},
                        diagnostic_input_sha256=sha256(manifest_path),
                        visibility_review_sha256=sha256(run / 'visibility_review.json'),
                        runner_sha256=sha256(__file__),
                        scope='Human-directed same-budget intervention; not an algorithm evaluation')
            durable(run / 'protocol.json', spec)
        spec = verify(run)
        assert spec['runner_sha256'] == sha256(__file__)
        assert spec['diagnostic_input_sha256'] == sha256(manifest_path)
        assert spec['visibility_review_sha256'] == sha256(run / 'visibility_review.json')
        data = {r['question_id']: r for r in rows()}
        dev = read_json(ROOT / 'data/manifests/video_mme_development_200_20260920.json')['rows']
        ordinals = {r['question_id']: i for i, r in enumerate(dev)}
        schedule = [(q, c) for q in inputs['question_ids'] for c in
                    (['original', 'repair'] if ordinals[q] % 2 == 0 else ['repair', 'original'])]
        allowed = {(q, c) for q, c in schedule}
        calls = list((run / 'calls').glob('*/*.json'))
        assert len(calls) <= 4 and all((p.stem, p.parent.name) in allowed for p in calls)
        for p in calls:
            recover_call(run, data[p.stem], p.parent.name, spec)
        pending = [(q, c) for q, c in schedule if not (run / 'results' / c / (q + '.json')).exists()]
        for q, c in schedule:
            if (q, c) not in pending:
                validate(run, data[q], c, spec)
        # 步骤2：仅需要新回答时检查原物理GPU1；忙则停止等待，不抢占或换卡。
        if pending:
            from videoqa_runtime.baseline_run import available_gpus
            free, states = available_gpus(['1'])
            assert '1' in free, '原物理GPU1忙，等待后恢复，不换卡'
            durable(run / 'gpu_preflight.json', dict(gpu='1', states=states, utc=utc()))
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
            offline_environment()
            import torch
            from PIL import Image
            from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
            from videoqa_budget32.qwen_adapter import FrozenInputQwen
            from videoqa_budget32.observe import observe
            from videoqa_lmms_bridge.bridge import synthetic_row
            torch.set_num_threads(2); torch.set_num_interop_threads(1)
            durable(run / 'status.json', dict(status='loading', pid=os.getpid(), sid=os.getsid(0), utc=utc()))
            backend = QwenFA2VideoBackend(); adapter = FrozenInputQwen(backend)
            synthetic = run / 'synthetic' / (str(time.time_ns()) + '.json')
            durable(synthetic, dict(status='started', real_calls=0))
            pictures = [Image.new('RGB', (896, 672), (40, i * 7, 150)) for i in range(32)]
            warm = observe(backend, adapter, pictures, synthetic_row(32), {})
            for picture in pictures:
                picture.close()
            durable(synthetic, dict(status='completed', result=warm, real_calls=0))
            # 步骤3：每个条件只调用一次原正式路径；异常退出，完整原始输出可恢复后处理。
            for q, condition in pending:
                durable(run / 'status.json', dict(status='running', question_id=q, condition=condition,
                                                pid=os.getpid(), sid=os.getsid(0), utc=utc()))
                process(run, data[q], condition, backend, adapter, '1', spec)
                current = validate(run, data[q], condition, spec)
                expected = inputs['observations'][q]['conditions'][condition]['processor_pixel_sha256']
                assert current['answer']['observed']['processor_pixel_sha256'] == expected
        # 步骤4：原条件与历史回答不一致时暂停该题归因，不追加调用或选用有利输出。
        pairs = []
        for q in inputs['question_ids']:
            original = validate(run, data[q], 'original', spec)
            repair = validate(run, data[q], 'repair', spec)
            old = read_json(inputs['observations'][q]['historical_result'])
            same = original['answer']['parsed_answer'] == old['answer']['parsed_answer']
            pairs.append(dict(question_id=q, original=original['answer']['parsed_answer'],
                              repair=repair['answer']['parsed_answer'], original_correct=original['correct'],
                              repair_correct=repair['correct'], historical_replay_matches=same,
                              attribution='eligible' if same else 'suspended_replay_mismatch'))
        assert len(list((run / 'calls').glob('*/*.json'))) == 4
        durable(run / 'summary.json', dict(status='completed', actual_real_calls=4, pairs=pairs,
                interpretation='Added and removed frames are one combined intervention; not algorithm accuracy'))
        durable(run / 'status.json', dict(status='completed', pid=os.getpid(), sid=os.getsid(0), utc=utc()))
        print(pairs, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    try:
        execute(args.run)
    except BaseException as exc:
        durable(Path(args.run) / 'failure.json', dict(status='paused', error=repr(exc), utc=utc()))
        raise
