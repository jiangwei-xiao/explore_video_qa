"""五题自动定位入口诊断：同粗池双条件，最多10次问答，不形成方法精度成绩。"""
import argparse
import os
from pathlib import Path
import sys
import time

os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256, offline_environment
from videoqa_full.state import durable, exclusive, rows, utc, invoke_once, QWEN_PATH
from videoqa_full.prepare import environment
from videoqa_budget32.frames import FrameProvider, validate_frames
from videoqa_budget32.run import process, validate, recover_call, critical_code, PYTHON
from videoqa_audit.framework_contraction import RUN32
from videoqa_audit.region_scope_runtime import parent_result, parent_call_sources, cpu_identity, source_order
from videoqa_lmms_bridge.common import framework_identity

SOURCE = ROOT / 'outputs/analysis/event_clause_full_coarse__videomme5__20261009_r1'
QIDS = ('555-2', '560-2', '599-1', '252-3', '553-3')
CONDITIONS = ('original_locator_coarse_only', 'event_locator_coarse_only')


def schedule(ordinals):
    """接口：仅按冻结题序交替两个条件的先后，不依据历史答案或标准答案排队。"""
    return [(q, c) for q in QIDS for c in (CONDITIONS if ordinals[q] % 2 == 0 else CONDITIONS[::-1])]


def check_identity(observed, expected):
    """接口：核验真实模型的官方像素、完整提示、Token和网格，无二次采样。"""
    for actual, frozen in (
        ('processor_pixel_sha256', 'pixel_sha256'), ('prompt', 'prompt'),
        ('input_token_ids', 'input_token_ids'), ('video_grid_thw', 'video_grid_thw'),
    ):
        if observed[actual] != expected[frozen]:
            raise ValueError('Actual input differs: ' + actual)


def checked_calls(root):
    """接口：正式额度只含五题双条件；合成放在独立目录，未知调用不得另发。"""
    paths = list((Path(root) / 'calls').glob('*/*.json'))
    allowed = {(q, c) for q in QIDS for c in CONDITIONS}
    if len(paths) > 10 or any((p.stem, p.parent.name) not in allowed for p in paths):
        raise ValueError('Real call budget or population differs')
    return paths


def prepare(root):
    """接口：CPU冻结十套自动输入和官方预处理身份；不人工修帧、不新增评分。"""
    root = Path(root); root.mkdir(parents=True, exist_ok=True); offline_environment()
    with exclusive(root / 'prepare.lock'):
        if (root / 'protocol.json').exists():
            return verify(root)
        import torch
        from transformers import AutoProcessor
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        parent = read_json(RUN32 / 'protocol.json')
        if framework_identity() != parent['framework'] or environment(PYTHON) != parent['environment']:
            raise ValueError('Parent evaluation environment changed')
        if read_json(SOURCE / 'status.json')['status'] != 'completed':
            raise ValueError('Global localization probe is not complete')
        replay = read_json(SOURCE / 'coarse_selection_replay/summary.json')
        if {r['question_id'] for r in replay['records']} != set(QIDS):
            raise ValueError('Frozen five-question population differs')
        for p, h in replay['files'].items():
            if sha256(p) != h:
                raise ValueError('Frozen coarse selection changed')
        for p, meta in parent['model_files'].items():
            if sha256(p) != meta['sha256']:
                raise ValueError('Frozen model bytes changed')
        processor = AutoProcessor.from_pretrained(
            QWEN_PATH, local_files_only=True, use_fast=False, min_pixels=100352,
            max_pixels=602112, size=dict(shortest_edge=100352, longest_edge=602112))
        bank = {r['question_id']: r for r in rows()}
        sources = {str(p): sha256(p) for p in (
            RUN32 / 'protocol.json', SOURCE / 'protocol.json',
            SOURCE / 'coarse_selection_replay/summary.json')}
        exports = {}; observations = {}
        # 步骤1：只读取已经冻结的自动选择；人工见证、事件查询不进入最终QA提示。
        for q in QIDS:
            historical_path = parent_result('qwen32', q); historical = read_json(historical_path)
            sources.update(parent_call_sources('qwen32', historical_path, historical))
            row = bank[q]; video = Path(parent['video_stats'][row['video_id']]['path'])
            if sha256(video) != parent['video_hashes'][row['video_id']]:
                raise ValueError('Video bytes changed')
            observations[q] = dict(historical_prediction=historical['answer']['parsed_answer'], conditions={})
            for condition in CONDITIONS:
                src = SOURCE / 'coarse_selection_replay' / condition / (q + '.json')
                item = read_json(src); selection = item['selection']
                if item['question_id'] != q or item['condition'] != condition or not item['coarse_only']:
                    raise ValueError('Wrong automatic selection source')
                validate_frames(selection, 32)
                if len(selection['selected_frames']) != 32 or any(
                    f['candidate_index'] >= item['source_initial_count'] for f in selection['selected_frames']):
                    raise ValueError('Fine/manual frames entered coarse-only diagnostic')
                sources[str(src)] = sha256(src); sources.update(item['sources'])
                images, rgb = FrameProvider(selection).decode()
                try:
                    identity = cpu_identity('qwen32', (processor, None), images, row)
                    same = source_order(selection) == source_order(historical['selection'])
                    if same:
                        if rgb != historical['rgb_sha256']:
                            raise ValueError('Historical same-source RGB differs')
                        check_identity(historical['answer']['observed'], identity)
                    rel = f'selections/{condition}/{q}.json'
                    durable(root / rel, dict(question_id=q, video_id=row['video_id'], method=condition,
                        selection=selection, rgb_sha256=rgb, input_identity=identity,
                        historical_input_identical=same))
                    exports[rel] = sha256(root / rel)
                    observations[q]['conditions'][condition] = dict(
                        input_identity=identity, historical_input_identical=same)
                finally:
                    for image in images: image.close()
            # 两条件保留同一原题与选项，媒体位置数相同；事件文字只曾用于上游评分。
            a, b = [observations[q]['conditions'][c]['input_identity'] for c in CONDITIONS]
            if a['prompt'] != b['prompt'] or a['input_token_ids'] != b['input_token_ids']:
                raise ValueError('Pair changed QA text or visual token count')
        # 步骤2：事前锁定10次真实/1次合成额度，不导入历史不同帧答案。
        code = dict(critical_code())
        for name in ('scripts/run_event_locator_diagnostic.py', 'src/videoqa_audit/region_scope_runtime.py',
                     'src/videoqa_full/state.py', 'src/videoqa_lmms_bridge/common.py'):
            code[name] = sha256(ROOT / name)
        spec = dict(model=parent['model'], question_ids=list(QIDS), methods=list(CONDITIONS),
            max_calls=10, max_calls_per_method=5, max_synthetic_calls=1,
            allowed_gpus=list(map(str, range(8))), model_files=parent['model_files'],
            framework=parent['framework'], environment=parent['environment'], code_sha256=code,
            sources=sources, selection_exports=exports, observations=observations,
            video_stats={bank[q]['video_id']: parent['video_stats'][bank[q]['video_id']] for q in QIDS},
            scope='Fixed five-case automatic locator comparison; both coarse-only; not formal algorithm accuracy',
            utc=utc())
        durable(root / 'protocol.json', spec)
        return verify(root)


def verify(root):
    """接口：来源、正式配置或代码身份变化即停止，绝不重算输入追求继续运行。"""
    root = Path(root); spec = read_json(root / 'protocol.json')
    if spec['question_ids'] != list(QIDS) or spec['methods'] != list(CONDITIONS) or spec['max_calls'] != 10:
        raise ValueError('Diagnostic population changed')
    if spec['max_calls_per_method'] != 5 or spec['max_synthetic_calls'] != 1:
        raise ValueError('Diagnostic budget changed')
    if framework_identity() != spec['framework'] or environment(PYTHON) != spec['environment']:
        raise ValueError('Evaluation environment changed')
    for p, h in spec['sources'].items():
        if sha256(p) != h: raise ValueError('Source changed: ' + p)
    for p, h in spec['selection_exports'].items():
        if sha256(root / p) != h: raise ValueError('Frozen input changed')
    for p, h in spec['code_sha256'].items():
        if sha256(ROOT / p) != h: raise ValueError('Frozen code changed')
    for p, meta in spec['model_files'].items():
        st = Path(p).stat()
        if st.st_size != meta['bytes'] or st.st_mtime_ns != meta['mtime_ns']:
            raise ValueError('Model identity changed')
    checked_calls(root)
    return spec


def audit(root):
    """接口：复算十个答案及配对身份；历史不同帧不冒充重复性或因果对照。"""
    root = Path(root); spec = verify(root); bank = {r['question_id']: r for r in rows()}; pairs = []
    for q in QIDS:
        results = [validate(root, bank[q], c, spec) for c in CONDITIONS]
        calls = [read_json(root / 'calls' / c / (q + '.json')) for c in CONDITIONS]
        if calls[0]['pid'] != calls[1]['pid'] or calls[0]['gpu'] != calls[1]['gpu']:
            raise ValueError('Pair not same process/GPU')
        history = {}
        for c, result in zip(CONDITIONS, results):
            frozen = spec['observations'][q]['conditions'][c]
            check_identity(result['answer']['observed'], frozen['input_identity'])
            same = frozen['historical_input_identical']
            history[c] = dict(input_identical=same, prediction_matches=(
                result['answer']['parsed_answer'] == spec['observations'][q]['historical_prediction'] if same else None))
        a, b = results
        pairs.append(dict(question_id=q, original_prediction=a['answer']['parsed_answer'],
            event_prediction=b['answer']['parsed_answer'], original_correct=a['correct'], event_correct=b['correct'],
            historical_checks=history, attribution_paused=any(
                x['input_identical'] and not x['prediction_matches'] for x in history.values())))
    paths = checked_calls(root) + list((root / 'results').glob('*/*.json')) + list((root / 'raw_generations').glob('*/*.json'))
    if len(paths) != 30: raise ValueError('Ten physical calls/results/raw returns not closed')
    synthetic = read_json(root / 'synthetic/calls/synthetic/warmup.json')
    raw = read_json(root / 'synthetic_raw.json')
    if synthetic['status'] != 'completed' or raw['generation_state'] != {'started': True, 'returned': True}:
        raise ValueError('Synthetic call not completed')
    hashes = {str(p): sha256(p) for p in paths}
    if (root / 'summary.json').exists() and read_json(root / 'summary.json')['artifacts'] != hashes:
        raise ValueError('Completed artifacts changed')
    value = dict(status='completed', real_calls=10, synthetic_calls=1, pairs=pairs,
        artifacts=hashes, scope=spec['scope'], utc=utc())
    durable(root / 'summary.json', value); durable(root / 'status.json', value)
    return value


def execute(root):
    """接口：单卡长期进程后台完成十次固定诊断，未知生成仅恢复原文，不另发。"""
    root = Path(root); spec = verify(root)
    with exclusive(root / 'coordinator.lock'):
        bank = {r['question_id']: r for r in rows()}
        order = schedule({r['question_id']: i for i, r in enumerate(rows())})
        calls = checked_calls(root)
        for p in calls: recover_call(root, bank[p.stem], p.parent.name, spec)
        if all((root / 'results' / c / (q + '.json')).exists() for q, c in order):
            return audit(root)
        # 存在部分生成时，仅重建已返回结果；不在新进程自动重复或续发该诊断。
        if calls:
            for p in calls:
                call = read_json(p)
                process(root, bank[p.stem], p.parent.name, None, None, call['gpu'], spec)
            if len(calls) == 10: return audit(root)
            raise RuntimeError('Partial diagnostic; inspect remaining budget before continuing')
        from videoqa_runtime.baseline_run import available_gpus
        while True:
            try: free, _ = available_gpus(spec['allowed_gpus'])
            except RuntimeError: free = []
            if free: break
            durable(root / 'status.json', dict(status='waiting_resources', pid=os.getpid(), sid=os.getsid(0), utc=utc()))
            time.sleep(15)
        gpu = free[0]; os.environ['CUDA_VISIBLE_DEVICES'] = gpu; offline_environment()
        import torch
        from PIL import Image
        from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
        from videoqa_budget32.qwen_adapter import FrozenInputQwen
        from videoqa_budget32.observe import observe
        from videoqa_lmms_bridge.bridge import synthetic_row
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        backend = QwenFA2VideoBackend(); adapter = FrozenInputQwen(backend)
        durable(root / 'load_report.json', dict(gpu=gpu, pid=os.getpid(), report=backend.load_report))
        images = [Image.new('RGB', (896, 672), (40, i * 7, 150)) for i in range(32)]
        fp = sha256(root / 'protocol.json')
        try:
            invoke_once(root / 'synthetic', 'warmup', 'synthetic', fp, gpu,
                lambda state: observe(backend, adapter, images, synthetic_row(32), state,
                    root / 'synthetic_raw.json', dict(question_id='warmup', method='synthetic', protocol_sha256=fp, gpu=gpu)))
        finally:
            for image in images: image.close()
        # 步骤1：每次生成前对实际处理器重新校验；仅原问题与选项进入问答后端。
        for q, condition in order:
            durable(root / 'status.json', dict(status='running', question_id=q, condition=condition,
                gpu=gpu, pid=os.getpid(), sid=os.getsid(0), utc=utc()))
            item = read_json(root / 'selections' / condition / (q + '.json'))
            images, rgb = FrameProvider(item['selection']).decode()
            try:
                if rgb != item['rgb_sha256'] or cpu_identity('qwen32', (backend.processor, None), images, bank[q]) != item['input_identity']:
                    raise ValueError('Actual frozen input changed before generation')
            finally:
                for image in images: image.close()
            process(root, bank[q], condition, backend, adapter, gpu, spec)
        # 步骤2：所有输入固定后计分，不能根据答错换输入再调用。
        return audit(root)


def main():
    """接口：准备/问答/审计分离；只允许这五题十次诊断，不启动200题或全量。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--stage', choices=('prepare', 'qa', 'audit'), required=True)
    args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum(): parser.error('Invalid run ID')
    root = ROOT / 'outputs/diagnostics' / args.run_id
    try:
        (prepare if args.stage == 'prepare' else execute if args.stage == 'qa' else audit)(root)
    except BaseException as exc:
        durable(root / 'failure.json', dict(status='paused', error=repr(exc), utc=utc()))
        raise


if __name__ == '__main__':
    main()

