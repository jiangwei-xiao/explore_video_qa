"""已有双查询分数的单一联合入口检验：九题、零模型调用、不改正式方法。"""
import argparse
import os
from pathlib import Path
import sys
import numpy as np

os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive, utc
from videoqa_audit.event_rank_entry import joint_rank_scores
from replay_event_seed_coarse_selection import locator_only

SOURCE = ROOT / 'outputs/analysis/event_clause_extension__videomme9__20261009_r2'
FILES = ('scripts/replay_joint_event_entry.py', 'src/videoqa_audit/event_rank_entry.py',
    'scripts/replay_event_seed_coarse_selection.py', 'src/videoqa_methods/prefix_coverage_budget.py',
    'src/videoqa_methods/coarse_eligibility.py', 'src/videoqa_methods/density_selection.py',
    'src/videoqa_methods/budget_contract.py')


def read_event_scores(item, root):
    """接口：只消费完整冻结的事件分数；回退使用原分数，不重新评分或生成查询。"""
    pool = read_json(item['pool_path']); sources = {}
    # 步骤1：三种来源分别记录，已有父引用不伪造新调用。
    if not item['query']['supported']:
        return pool['initial_scores'], sources
    if item['imported_result']:
        path = Path(item['imported_result']); sources[str(path)] = sha256(path)
        return read_json(path)['comparisons']['event_clause']['scores'], sources
    values = []
    for start in item['new_starts']:
        path = Path(root) / 'calls' / item['question_id'] / 'event_clause' / (str(start) + '.json')
        call = read_json(path)
        if call['status'] != 'completed': raise ValueError('Uncertain source scores')
        sources[str(path)] = sha256(path); values += call['scores']
    # 步骤2：同池身份和完整分数数目必须闭合。
    if len(values) != pool['initial_count']: raise ValueError('Incomplete source event scores')
    return values, sources


def execute(root):
    """接口：事前冻结九题双排序规则，再复算粗池选择；评价字段不参与入口函数。"""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with exclusive(root / 'coordinator.lock'):
        parent = read_json(SOURCE / 'protocol.json'); completed = read_json(SOURCE / 'summary.json')
        if completed['status'] != 'completed' or completed['new_qa_calls'] != 0:
            raise ValueError('Source population not complete')
        sources = {str(p): sha256(p) for p in (SOURCE / 'protocol.json', SOURCE / 'summary.json')}
        code = {p: sha256(ROOT / p) for p in FILES}
        prepared = {}
        # 步骤1：选择函数仅接收候选、双查询分数和特征；不接收题号/选项/答案。
        for q in parent['question_ids']:
            p = SOURCE / 'inputs' / (q + '.json')
            if sha256(p) != parent['input_exports'][q]: raise ValueError('Frozen source query changed')
            item = read_json(p); event, referenced = read_event_scores(item, SOURCE)
            for path in (p, Path(item['pool_path']), Path(item['feature_path'])):
                sources[str(path)] = sha256(path)
            sources.update(referenced)
            prepared[q] = (item, event)
        spec = dict(question_ids=parent['question_ids'], source_hashes=sources, code_sha256=code,
            rule='minimize original_question_rank * event_query_rank; stable PTS/id ties',
            budget=32, new_model_calls=0, new_QA_calls=0,
            scope='One heuristic joint-entry replay, coarse-only; not probability or formal algorithm accuracy')
        if (root / 'protocol.json').exists():
            if read_json(root / 'protocol.json') != spec: raise ValueError('Frozen joint replay changed')
        else:
            for p in FILES:
                import shutil
                dest = root / 'code_snapshot' / p; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / p, dest)
            durable(root / 'protocol.json', spec)
        files = {}; records = []
        # 步骤2：联合分数只改变种子/区域；支持仍按原题，视觉差异及等权DP均不改。
        for q, (item, event) in prepared.items():
            pool = read_json(item['pool_path']); n = pool['initial_count']
            coarse = dict(pool, candidates=pool['candidates'][:n])
            vectors = np.load(item['feature_path'], allow_pickle=False)[:n]
            scores, rank_trace = joint_rank_scores(coarse['candidates'], pool['initial_scores'], event)
            indices, trace, plan = locator_only(coarse, vectors, scores)
            original = read_json(SOURCE / 'selections/original_locator_coarse_only' / (q + '.json'))
            pure_event = read_json(SOURCE / 'selections/event_locator_coarse_only' / (q + '.json'))
            selection = dict(video=pool['video'], selected_frames=[pool['candidates'][i] for i in indices])
            if not item['query']['supported'] and selection != original['selection']:
                raise ValueError('Identical-query fallback did not preserve original input')
            path = root / 'selections/joint_rank_coarse_only' / (q + '.json')
            value = dict(question_id=q, condition='joint_rank_coarse_only', query=item['query'],
                selection=selection, selected_indices=indices, plan=plan, trace=trace,
                rank_trace=rank_trace, new_model_calls=0, coarse_only=True)
            if path.exists() and read_json(path) != value: raise ValueError('Selection changed on recovery')
            if not path.exists(): durable(path, value)
            files[str(path)] = sha256(path)
            records.append(dict(question_id=q, changed_from_original=len(set(indices) ^ set(original['selected_indices'])) // 2,
                changed_from_event=len(set(indices) ^ set(pure_event['selected_indices'])) // 2,
                fallback=not item['query']['supported']))
        # 步骤3：完整输出后才保存完成标志；恢复只重算与比对，不追加模型调用。
        summary = dict(status='completed', records=records, artifacts=files, new_model_calls=0,
            new_QA_calls=0, utc=utc(), scope=spec['scope'])
        if (root / 'summary.json').exists() and read_json(root / 'summary.json')['artifacts'] != files:
            raise ValueError('Completed replay changed')
        durable(root / 'summary.json', summary)
        return summary


def main():
    """接口：一次固定离线回放，不扫描融合系数，也不触发评分或真实问答。"""
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True); args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum(): parser.error('Invalid run ID')
    root = ROOT / 'outputs/analysis' / args.run_id
    try: execute(root)
    except BaseException as exc:
        durable(root / 'failure.json', dict(status='paused', error=repr(exc), utc=utc())); raise


if __name__ == '__main__':
    main()
