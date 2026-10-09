"""事件定位适用范围扩展：完整九题、三题新评分、一题语义回退，新增QA为0。"""
import argparse
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from videoqa_runtime.common import read_json, sha256, offline_environment
from videoqa_full.state import rows, durable, exclusive, utc
from videoqa_full.prepare import environment
from videoqa_audit.framework_contraction import BASE
from videoqa_audit.event_clause import event_clause
from videoqa_audit.event_query_contract import checked_event_clause
from videoqa_audit.query_input_probe import MODEL, PYTHON, score_once
from videoqa_budget32.selection import SOURCE, configure_visual_precision, config
from videoqa_budget32.frames import FrameProvider
from videoqa_methods.budget_contract import select_exp07
from videoqa_methods.density_selection import build_regions
from replay_event_seed_coarse_selection import locator_only

PARENT = ROOT / 'outputs/analysis/event_clause_full_coarse__videomme5__20261009_r1'
FILES = ('scripts/probe_event_clause_extension.py', 'src/videoqa_audit/event_query_contract.py',
         'src/videoqa_audit/event_clause.py', 'src/videoqa_audit/query_input_probe.py',
         'src/videoqa_methods/model_ops.py', 'src/videoqa_runtime/baseline_selection.py',
         'src/videoqa_methods/density_selection.py', 'src/videoqa_budget32/selection.py',
         'src/videoqa_budget32/frames.py', 'scripts/replay_event_seed_coarse_selection.py')


def prepare(root):
    """接口：只由原题句式确定九题总体，冻结回退/父引用/新评分，绝不按答案筛选。"""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with exclusive(root / 'prepare.lock'):
        if (root / 'protocol.json').exists(): return verify(root)
        old = read_json(SOURCE / 'protocol.json'); parent = read_json(PARENT / 'summary.json')
        if parent['status'] != 'completed': raise ValueError('Parent score population not complete')
        model_files = {p: x['sha256'] for p, x in old['model_files'].items() if p.startswith(str(MODEL) + '/')}
        for p, h in model_files.items():
            if sha256(p) != h: raise ValueError('BLIP model changed')
        sources = {str(p): sha256(p) for p in (PARENT / 'protocol.json', PARENT / 'summary.json')}
        for p, h in parent['artifacts'].items():
            if sha256(p) != h: raise ValueError('Parent completed return changed')
            sources[p] = h
        data = [r for r in rows() if event_clause(r['question'])['supported']]
        if len(data) != 9: raise ValueError('Original-question grammar population changed')
        exports = {}; planned_batches = 0; planned_positions = 0
        # 步骤1：输入对象排除选项、标准答案与人工事实；同一原文表达才引用父评分。
        for row in data:
            q = row['question_id']; pp = BASE / 'pools' / (q + '.json'); fp = BASE / 'features' / (q + '.npy')
            cp = BASE / 'selection_checkpoints' / (q + '.json'); checkpoint = read_json(cp)
            for p in (pp, fp):
                if sha256(p) != checkpoint['files'][str(p.relative_to(BASE))]: raise ValueError('Parent coarse asset changed')
                sources[str(p)] = sha256(p)
            pool = read_json(pp); n = pool['initial_count']; query = checked_event_clause(row['question'])
            video = pool['video']['path']
            if sha256(video) != old['video_hashes'][row['video_id']]: raise ValueError('Video bytes changed')
            sources[video] = old['video_hashes'][row['video_id']]
            imported = None; starts = []; controls = []
            if query['supported']:
                path = PARENT / 'results' / (q + '.json')
                if path.exists():
                    saved = read_json(path)
                    if saved['query']['event_query'] != query['event_query'] or saved['coarse_count'] != n:
                        raise ValueError('Parent query or frame population differs')
                    imported = str(path); sources[str(path)] = sha256(path)
                else:
                    starts = list(range(0, n, 16)); controls = sorted({0, (n - 1) // 16 * 16})
                    planned_batches += len(starts) + len(controls)
                    planned_positions += n + sum(min(16, n - i) for i in controls)
            item = dict(question_id=q, question=row['question'], query=query, pool_path=str(pp),
                feature_path=str(fp), initial_count=n, imported_result=imported,
                new_starts=starts, original_control_starts=controls)
            path = root / 'inputs' / (q + '.json'); durable(path, item); exports[q] = sha256(path)
        # 步骤2：三类成本分开；回退没有新评分，已有五题只引用完整事件分数。
        spec = dict(question_ids=[r['question_id'] for r in data], input_exports=exports, sources=sources,
            model_files=model_files, files={p: sha256(ROOT / p) for p in FILES},
            environment=environment(PYTHON), allowed_gpus=list(map(str, range(8))),
            new_batch_calls=planned_batches, new_frame_text_scores=planned_positions,
            max_synthetic_calls=1, new_qa_limit=0, replay_tolerance=1e-6,
            configuration=config(32, 32), utc=utc(),
            scope='Whole literal event-query grammar cohort; guarded conditions; coarse-only diagnostic, not new method accuracy')
        for p in FILES:
            dest = root / 'code_snapshot' / p; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / p, dest)
        durable(root / 'protocol.json', spec)
        return spec


def verify(root):
    """接口：冻结身份验收；前向不确定或代码变化即暂停，不重新评分修复现状。"""
    root = Path(root); spec = read_json(root / 'protocol.json')
    if len(spec['question_ids']) != 9 or spec['new_qa_limit'] != 0: raise ValueError('Scope changed')
    if environment(PYTHON) != spec['environment']: raise ValueError('Scoring environment changed')
    for p, h in {**spec['sources'], **spec['model_files']}.items():
        if sha256(p) != h: raise ValueError('Input source changed: ' + p)
    for p, h in spec['files'].items():
        if sha256(ROOT / p) != h: raise ValueError('Frozen execution code changed')
    for q, h in spec['input_exports'].items():
        if sha256(root / 'inputs' / (q + '.json')) != h: raise ValueError('Question/query contract changed')
    return spec


def real_score_paths(root):
    """接口：真实帧评分账本与synthetic分开计数，额外未知题仍进入验收检查。"""
    return [p for p in (Path(root) / 'calls').glob('*/*/*.json') if p.parent.parent.name != 'synthetic']


def report(root):
    """接口：复算九题种子与无细帧输入；没有已列见证的画面不自动判为无关。"""
    root = Path(root); spec = verify(root); records = []; artifacts = {}
    for q in spec['question_ids']:
        item = read_json(root / 'inputs' / (q + '.json')); pool = read_json(item['pool_path'])
        n = item['initial_count']; coarse = dict(pool, candidates=pool['candidates'][:n])
        vectors = np.load(item['feature_path'], allow_pickle=False)[:n]
        if not item['query']['supported']:
            scores = pool['initial_scores']
        elif item['imported_result']:
            scores = read_json(item['imported_result'])['comparisons']['event_clause']['scores']
        else:
            scores = []
            for condition, starts in [('event_clause', item['new_starts']), ('original_question', item['original_control_starts'])]:
                for start in starts:
                    p = root / 'calls' / q / condition / (str(start) + '.json'); call = read_json(p)
                    if call['status'] != 'completed' or call['identity']['protocol_sha256'] != sha256(root / 'protocol.json'):
                        raise ValueError('Uncertain BLIP return')
                    if sha256(p.with_suffix('.npy')) != call['feature_sha256']: raise ValueError('Feature return changed')
                    artifacts[str(p)] = sha256(p); artifacts[str(p.with_suffix('.npy'))] = sha256(p.with_suffix('.npy'))
                    if condition == 'event_clause': scores += call['scores']
                    elif max(abs(s - pool['initial_scores'][start + i]) for i, s in enumerate(call['scores'])) > spec['replay_tolerance']:
                        raise ValueError('Original native-batch replay differs')
        if len(scores) != n: raise ValueError('Incomplete event scoring population')
        # 步骤1：只改定位分数；原问题支持/视觉差异/等权覆盖与K32仍使用冻结代码。
        selected_a, trace_a = select_exp07(coarse, vectors, 32)
        selected_b, trace_b, plan_b = locator_only(coarse, vectors, scores)
        plan_a = build_regions(coarse['candidates'], pool['initial_scores'], pool['video'], spec['configuration'])
        detail = {}
        for condition, indices, trace, plan in [
            ('original_locator_coarse_only', selected_a, trace_a, plan_a),
            ('event_locator_coarse_only', selected_b, trace_b, plan_b)]:
            result = dict(question_id=q, condition=condition, selection=dict(video=pool['video'],
                selected_frames=[pool['candidates'][i] for i in indices]), selected_indices=indices,
                trace=trace, plan=plan, query=item['query'], coarse_only=True, new_qa_calls=0)
            path = root / 'selections' / condition / (q + '.json')
            if path.exists() and read_json(path) != result: raise ValueError('Coarse selection changed on recovery')
            if not path.exists(): durable(path, result)
            detail[condition] = dict(seed_indices=plan['seed_indices'], region_budgets=trace['region_budgets'])
        records.append(dict(question_id=q, query=item['query'], scoring_kind=(
            'fallback' if not item['query']['supported'] else 'imported' if item['imported_result'] else 'new'),
            changed_frames=len(set(selected_a) ^ set(selected_b)) // 2, details=detail))
    real_calls = real_score_paths(root)
    if len(real_calls) != spec['new_batch_calls'] or len(artifacts) != 2 * spec['new_batch_calls']:
        raise ValueError('New scoring ledger budget not closed')
    if sum(read_json(p)['identity']['frames'] for p in real_calls) != spec['new_frame_text_scores']:
        raise ValueError('New frame-text positions not closed')
    # 步骤2：报告区分父引用与新题，不以正向小案例自动允许问答或全量。
    value = dict(status='completed', records=records, artifacts=artifacts, new_batches=len(real_calls),
        new_frame_text_scores=spec['new_frame_text_scores'], new_qa_calls=0, scope=spec['scope'])
    if (root / 'summary.json').exists() and read_json(root / 'summary.json')['artifacts'] != artifacts:
        raise ValueError('Completed scoring artifacts changed')
    durable(root / 'summary.json', value); durable(root / 'status.json', dict(status='completed', utc=utc()))
    return value


def execute(root):
    """接口：后台单卡执行新三题的完整粗池，原题首末历史分块控制，完成不再前向。"""
    root = Path(root); spec = verify(root)
    if (root / 'summary.json').exists(): return report(root)
    with exclusive(root / 'coordinator.lock'):
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
        from videoqa_methods.model_ops import FeatureScorer
        torch.set_num_threads(2); torch.set_num_interop_threads(1); configure_visual_precision(torch)
        scorer = FeatureScorer(str(MODEL)); images = [Image.new('RGB', (1280, 720), 'gray') for _ in range(16)]
        try:
            score_once(root, 'synthetic', 'warmup', 0, scorer, scorer.tokenize('Which synthetic color is visible?'),
                images, gpu, [__import__('hashlib').sha256(np.asarray(im).tobytes()).hexdigest() for im in images])
        finally:
            for image in images: image.close()
        # 步骤1：每一批最多16张真实源帧，纯视觉返回对齐旧粗特征，事件不改变表征。
        for q in spec['question_ids']:
            item = read_json(root / 'inputs' / (q + '.json'))
            if not item['new_starts']: continue
            pool = read_json(item['pool_path']); vectors = np.load(item['feature_path'], allow_pickle=False)
            encoded = {c: scorer.tokenize(t) for c, t in (
                ('original_question', item['question']), ('event_clause', item['query']['event_query']))}
            for start in item['new_starts']:
                batch = pool['candidates'][start:min(start + 16, item['initial_count'])]
                images, rgb = FrameProvider(dict(video=pool['video'], selected_frames=batch)).decode()
                try:
                    conditions = ['event_clause'] + (['original_question'] if start in item['original_control_starts'] else [])
                    for condition in conditions:
                        scores, features = score_once(root, q, condition, start, scorer, encoded[condition], images, gpu, rgb)
                        if float(np.max(np.abs(features - vectors[start:start + len(batch)]))) > spec['replay_tolerance']:
                            raise ValueError('Pure visual features differ from original native batch')
                        if condition == 'original_question' and max(
                            abs(s - pool['initial_scores'][start + i]) for i, s in enumerate(scores)) > spec['replay_tolerance']:
                            raise ValueError('Original native score control differs')
                finally:
                    for image in images: image.close()
                durable(root / 'status.json', dict(status='scoring', question_id=q, coarse_start=start,
                    coarse_total=item['initial_count'], gpu=gpu, pid=os.getpid(), sid=os.getsid(0), utc=utc()))
        # 步骤2：完成输入路径后停止，保留新题画面审核，不自动启动QA。
        return report(root)


def main():
    """接口：准备/评分/报告分阶段；独立适用范围诊断不能冒充exp或200题成绩。"""
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True)
    parser.add_argument('--stage', choices=('prepare', 'score', 'report'), required=True)
    args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum(): parser.error('Invalid run ID')
    root = ROOT / 'outputs/analysis' / args.run_id
    try:
        (prepare if args.stage == 'prepare' else execute if args.stage == 'score' else report)(root)
    except BaseException as exc:
        durable(root / 'status.json', dict(status='paused', error=repr(exc), pid=os.getpid(), sid=os.getsid(0), utc=utc()))
        raise


if __name__ == '__main__':
    main()
