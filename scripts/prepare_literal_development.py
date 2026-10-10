"""字面文字见证的200题准备：独立OCR资产、冻结原候选、只选帧不生成答案。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import shutil
import sys
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from videoqa_full.state import durable, exclusive, utc
from videoqa_runtime.common import read_json, sha256
from videoqa_audit.region_scope_runtime import development_rows
from videoqa_audit.framework_contraction import BASE, locations
from videoqa_audit.text_state_assets import extract_question, ocr_identity
from videoqa_methods.literal_option_coverage import option_word_groups, literal_matches, select_literal_coverage


def verify(root):
    """接口：冻结200题/旧池/特征/代码/OCR环境；不接受按旧答案挑题或隐式换规则。"""
    spec = read_json(root / 'protocol.json')
    if [row['question_id'] for row in development_rows()] != spec['question_ids'] or ocr_identity() != spec['ocr']:
        raise ValueError('Frozen development list or OCR environment changed')
    for path, expected in spec['sources'].items():
        if sha256(path) != expected:
            raise ValueError('Frozen development source changed: ' + path)
    return spec


def prepare(root):
    """接口：仅冻结200题选择准备；正式答题另立协议，不把OCR字面识别当准确率。"""
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'protocol.json').exists():
        return verify(root)
    data = development_rows(); inputs = {}; sources = {}
    # 步骤1：原M32粗细资产复用；公共题选项与正确答案对象分开保存。
    for row in data:
        q = row['question_id']; pp, fp, tp, sp = locations(q, 32)
        for path in (pp, fp, tp, sp):
            sources[str(path)] = sha256(path)
        inputs[q] = dict(pool=str(pp), features=str(fp), parent_trace=str(tp), parent_selection=str(sp),
            question=row['question'], options=row['options'])
    for path in [Path(__file__), ROOT / 'src/videoqa_methods/literal_option_coverage.py',
        ROOT / 'src/videoqa_audit/text_state_assets.py', ROOT / 'scripts/text_state_ocr_worker.py',
        ROOT / 'src/videoqa_methods/text_state_redundancy.py']:
        sources[str(path)] = sha256(path)
        dest = root / 'code_snapshot' / path.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    # 步骤2：旧安全OCR接口只需要qid→video绑定；本运行独立产物，不扩写旧exp29目录。
    spec = dict(reference_method='RD-1.2-exp07', question_ids=[r['question_id'] for r in data],
        question_video_ids={r['question_id']: r['video_id'] for r in data}, inputs=inputs,
        sources=sources, ocr=ocr_identity(), maximum_new_ocr_frame_positions=sum(len(read_json(i['pool'])['candidates']) for i in inputs.values()),
        final_QA_limit=0, source_BLIP_or_visual_calls=0, scope='Frozen Qwen32 development200 OCR and selection preparation, not QA',
        rule='Complete public option word-group coverage; same literal pilot r2 formula and original exp07 quota')
    durable(root / 'protocol.json', spec)
    return spec


def select_question(root, q, item):
    """接口：完整OCR与原配额→32帧上限输入；空组控制必须逐源复现父结果。"""
    bank = read_json(root / 'ocr_banks' / (q + '.json')); pool = read_json(item['pool'])
    if bank['pool_sha256'] != sha256(item['pool']) or bank['protocol_sha256'] != sha256(root / 'protocol.json'):
        raise ValueError('New OCR bank input identity differs')
    f = np.load(item['features'], allow_pickle=False); parent = read_json(item['parent_trace'])
    groups = option_word_groups(item['question'], item['options']); matches = literal_matches(bank['states'], groups)
    # 步骤1：原视觉规则控制，不能因文字缓存或池大小差异混入另一个方法。
    control, _ = select_literal_coverage(pool, f, parent, [set() for _ in bank['states']], 32)
    expected = read_json(item['parent_selection'])['selection']['selected_frames']
    if [pool['candidates'][i] for i in control] != expected:
        raise ValueError('Original control differs in development: ' + q)
    # 步骤2：仅改变首次字面组保留机会，不改模型、窗口、补查、预算或答案提示词。
    selected, trace = select_literal_coverage(pool, f, parent, matches, 32)
    value = dict(question_id=q, selection=dict(video=pool['video'], selected_frames=[pool['candidates'][i] for i in selected]),
        selected_indices=selected, groups=[list(group) for group in groups], trace=trace,
        old_control_exact=True, changed_frames=len(set(control) ^ set(selected)) // 2,
        matching_frames=sum(bool(m) for m in matches), final_QA_calls=0)
    path = root / 'selections' / (q + '.json')
    if path.exists() and read_json(path) != value:
        raise ValueError('Completed development selection changed')
    durable(path, value)
    return value


def execute(root, workers):
    """接口：后台CPU OCR→全200选择；成功缓存安全恢复，不依赖会话继续发送命令。"""
    with exclusive(root / 'coordinator.lock'):
        spec = prepare(root); verify(root); completed = []
        durable(root / 'launch.json', dict(pid=os.getpid(), sid=os.getsid(0), workers=workers, utc=utc()))
        # 步骤1：单视频问题最多一个OCR进程，CPU每次只保存当前RGB，原资产只读。
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(extract_question, (str(root), q)): q for q in spec['question_ids']}
            for future in as_completed(futures):
                q = futures[future]; future.result(); completed.append(q)
                durable(root / 'status.json', dict(status='ocr', completed=len(completed), total=200,
                    active_finished=q, pid=os.getpid(), sid=os.getsid(0), utc=utc()))
        # 步骤2：全量结果都完成才冻结选择，OCR失败不生成缺失题的答案。
        reports = [select_question(root, q, item) for q, item in spec['inputs'].items()]
        files = {str(root / 'selections' / (r['question_id'] + '.json')): sha256(root / 'selections' / (r['question_id'] + '.json')) for r in reports}
        summary = dict(questions=200, changed_questions=sum(bool(r['changed_frames']) for r in reports),
            literal_hit_questions=sum(bool(r['matching_frames']) for r in reports), selection_exports=files,
            final_QA_calls=0, new_ocr_banks=200, ready_for_separate_QA_input_review=True)
        if (root / 'summary.json').exists() and read_json(root / 'summary.json') != summary:
            raise ValueError('Completed preparation summary changed')
        durable(root / 'summary.json', summary); durable(root / 'status.json', dict(status='completed', utc=utc()))


def main():
    """接口：8个有界CPU提取进程，不抢GPU；准备阶段始终不启动正式生成。"""
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True); parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum() or not 1 <= args.workers <= 8:
        parser.error('Invalid run ID or CPU concurrency')
    root = ROOT / 'outputs/methods' / args.run_id
    try:
        execute(root, args.workers)
    except BaseException as exc:
        durable(root / 'failure.json', dict(status='paused', error=repr(exc), utc=utc())); raise


if __name__ == '__main__':
    main()
