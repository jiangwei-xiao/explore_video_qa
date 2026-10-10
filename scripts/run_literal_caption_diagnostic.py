"""两题四次Qwen32诊断：原输入与字面见证输入，不是正式方法或开发集成绩。"""
import argparse
import os
from pathlib import Path
import sys
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from videoqa_full.state import rows, durable, exclusive, utc
from videoqa_runtime.common import read_json, sha256
from videoqa_budget32.frames import FrameProvider
from videoqa_budget32.run import process, recover_call, validate
from videoqa_lmms_bridge.common import framework_identity
from videoqa_audit.framework_contraction import RUN32

SOURCE = ROOT / 'outputs/analysis/literal_options__pilot8__20261010_r2'
IDS = ('007-2', '584-3')
CONDITIONS = ('original', 'literal')


def prepare(root):
    """接口：先冻结四组真实源/RGB及调用上限；标准答案仅留在生成后的计分对象。"""
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'protocol.json').exists():
        return verify(root)
    source = read_json(SOURCE / 'protocol.json'); byid = {r['question_id']: r for r in rows()}
    exports = {}; sources = dict(source['sources']); stats = {}
    legacy = read_json(RUN32 / 'protocol.json')
    for path, identity in legacy['model_files'].items():
        if sha256(path) != identity['sha256']:
            raise ValueError('Frozen QA model changed')
        sources[path] = identity['sha256']
    # 步骤1：新输入只来自已冻结字面选择，原输入只来自对应父选帧，不择优改图。
    for q in IDS:
        item = source['inputs']['qwen32:' + q]; old_path = Path(item['parent_selection'])
        new_path = SOURCE / 'selections/qwen32' / (q + '.json')
        for condition, path in [('original', old_path), ('literal', new_path)]:
            value = read_json(path); selection = value['selection']
            images, hashes = FrameProvider(selection).decode()
            for image in images:
                image.close()
            dest = root / 'selections' / condition / (q + '.json')
            durable(dest, dict(question_id=q, selection=selection, rgb_sha256=hashes))
            exports[str(dest.relative_to(root))] = sha256(dest); sources[str(path)] = sha256(path)
            video = Path(selection['video']['path']); st = video.stat()
            stats[byid[q]['video_id']] = dict(path=str(video), bytes=st.st_size, mtime_ns=st.st_mtime_ns)
    # 步骤2：模型/源/代码身份来自父指纹，新增诊断命名与正式方法版本分开。
    for path in [Path(__file__), ROOT / 'src/videoqa_budget32/run.py', ROOT / 'src/videoqa_budget32/observe.py', ROOT / 'src/videoqa_budget32/qwen_adapter.py',
        ROOT / 'src/videoqa_full/qwen_fa2_backend.py', ROOT / 'src/videoqa_full/qwen_backend.py']:
        sources[str(path)] = sha256(path)
    spec = dict(model='Qwen2.5-VL-7B-Instruct', reference_method='RD-1.2-exp07',
        condition_namespace=list(CONDITIONS), question_ids=list(IDS), maximum_new_real_generations=4,
        maximum_synthetic_generations=1, sources=sources, selection_exports=exports,
        video_stats=stats, framework=framework_identity(),
        scope='Two familiar cases, paired input diagnostic; not a method accuracy',
        generation_limit=16, seed=2027, GPU='7')
    durable(root / 'protocol.json', spec)
    return spec


def verify(root):
    """接口：所有输入和代码字节相同才恢复；完整返回只后处理，不确定生成不重复。"""
    spec = read_json(root / 'protocol.json')
    for path, expected in spec['sources'].items():
        if sha256(path) != expected:
            raise ValueError('Diagnostic source changed: ' + path)
    for relative, expected in spec['selection_exports'].items():
        if sha256(root / relative) != expected:
            raise ValueError('Frozen diagnostic input changed')
    return spec


def execute(root):
    """接口：同进程同GPU、交替条件顺序，总共四次真实生成；完成恢复零生成。"""
    with exclusive(root / 'coordinator.lock'):
        spec = prepare(root); verify(root); byid = {r['question_id']: r for r in rows()}
        finished = 0
        if len(list((root / 'calls').rglob('*.json'))) > 4:
            raise ValueError('Real diagnostic call limit exceeded')
        # 步骤1：先检查已有开始/原始返回，不通过换条件ID绕过不确定的物理调用。
        for q in IDS:
            for condition in CONDITIONS:
                recover_call(root, byid[q], condition, spec)
                path = root / 'results' / condition / (q + '.json')
                if path.exists():
                    validate(root, byid[q], condition, spec); finished += 1
        if finished < 4:
            from PIL import Image
            from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
            from videoqa_budget32.qwen_adapter import FrozenInputQwen
            from videoqa_budget32.observe import observe
            from videoqa_lmms_bridge.bridge import synthetic_row
            backend = QwenFA2VideoBackend(); backend.torch.set_num_threads(2); adapter = FrozenInputQwen(backend)
            # 步骤2：正式路径合成只预热一次，单列，不据开发题结果选配置。
            if not (root / 'synthetic.json').exists():
                if (root / 'synthetic_started.json').exists():
                    raise ValueError('Uncertain synthetic generation; repetition forbidden')
                images = [Image.new('RGB', (896, 672), (20 + i * 3, 100, 150)) for i in range(32)]
                durable(root / 'synthetic_started.json', dict(started=True, utc=utc()))
                value = observe(backend, adapter, images, synthetic_row(32), {})
                for image in images:
                    image.close()
                durable(root / 'synthetic.json', value)
            for ordinal, q in enumerate(IDS):
                order = CONDITIONS if ordinal % 2 == 0 else CONDITIONS[::-1]
                for condition in order:
                    process(root, byid[q], condition, backend, adapter, '7', spec)
                    finished = sum((root / 'results' / c / (question + '.json')).exists() for question in IDS for c in CONDITIONS)
                    durable(root / 'status.json', dict(status='running', completed=finished,
                        pid=os.getpid(), sid=os.getsid(0), utc=utc()))
        # 步骤3：四条完整回算，保留新旧原输出，不因答案高低重跑。
        reports = []
        for q in IDS:
            pair = {c: validate(root, byid[q], c, spec) for c in CONDITIONS}
            reports.append(dict(question_id=q, outputs={c: pair[c]['answer']['raw_output'] for c in CONDITIONS},
                correct={c: pair[c]['correct'] for c in CONDITIONS}))
        files = {str(p): sha256(p) for folder in ['calls', 'raw_generations', 'results'] for p in (root / folder).rglob('*.json')}
        if (root / 'summary.json').exists() and read_json(root / 'summary.json')['files'] != files:
            raise ValueError('Completed diagnostic artifacts changed')
        durable(root / 'summary.json', dict(reports=reports, files=files, real_generations=4,
            synthetic_generations=1, scope=spec['scope']))
        durable(root / 'status.json', dict(status='completed', utc=utc()))


def main():
    """接口：独立后台诊断，开始失败占额度；当前空闲GPU7之外不抢占资源。"""
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True); args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum():
        parser.error('Invalid run ID')
    root = ROOT / 'outputs/diagnostics' / args.run_id
    try:
        execute(root)
    except BaseException as exc:
        durable(root / 'failure.json', dict(status='paused', error=repr(exc), utc=utc())); raise


if __name__ == '__main__':
    main()
