#!/usr/bin/env python3
"""完整粗池检索对照的零调用评价：已列事实与全部入口得失分开记录。"""
import argparse
from fractions import Fraction
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import ranking, WITNESSES
from videoqa_methods.density_selection import source_time
from run_option_route_probe import seed_windows
from videoqa_audit.joint_query_views import select_joint_views


def evaluate(run):
    """接口：模型结果已完成才评价；事实不进入检索函数，细帧不冒充粗候选。"""
    summary = read_json(run/'summary.json')
    assert summary['status'] == 'completed' and summary['qa_calls'] == 0
    for p, h in summary['artifacts'].items():
        assert sha256(p) == h
    facts, selections = [], {}
    # 步骤1：分别记录原题、问题加选项、仅选项；使用同样32入口和原几何窗口。
    for q, comparison in summary['comparisons'].items():
        item = read_json(run/'inputs'/(q+'.json'))
        pool = read_json(item['pool_path'])
        coarse = pool['candidates'][:item['initial_count']]
        original_order = ranking(coarse, pool['initial_scores'])
        groups = {'question': sorted(original_order[:32]),
                  'question_options': comparison['old_indices'], 'options_only': comparison['new_indices']}
        orders = {'question': [original_order],
                  'question_options': comparison['old_trace']['per_route_orders'],
                  'options_only': comparison['new_trace']['per_route_orders']}
        old_scores = read_json(item['previous_scores'])['scores']
        new_scores = read_json(run/'results'/(q+'.json'))['scores']
        for name, vectors in [('joint_question_options', old_scores), ('joint_options_only', new_scores)]:
            ids, trace = select_joint_views(coarse, pool['initial_scores'], vectors)
            groups[name] = ids
            orders[name] = trace['per_view_orders']
        windows = {k:seed_windows(coarse, ids, pool['video']) for k,ids in groups.items()}
        selections[q] = {k:dict(indices=ids, windows=windows[k], frames=[coarse[i] for i in ids]) for k,ids in groups.items()}
        # 步骤2：只计既有清单事实；没有列出见证不等于画面无关。
        for label, ids, meaning in WITNESSES.get(q, []):
            row = dict(question_id=q, fact=label, candidate_indices=ids, meaning=meaning, conditions={})
            for condition, selected in groups.items():
                hit = bool(set(ids) & set(selected))
                reachable = any(Fraction(w['left_fraction']) <= source_time(coarse[i], pool['video']) <= Fraction(w['right_fraction'])
                                for w in windows[condition] for i in ids)
                row['conditions'][condition] = dict(seed_hit=hit, window_hit=reachable,
                    per_route_best_rank=[min(order.index(i)+1 for i in ids) for order in orders[condition]])
            facts.append(row)
    totals = {k:{metric:sum(f['conditions'][k][metric] for f in facts) for metric in ['seed_hit','window_hit']}
              for k in ['question','question_options','options_only','joint_question_options','joint_options_only']}
    # 步骤3：保存全量入口与分母，不计算准确率，不把扩窗可达当作已保留证据。
    value = dict(source_summary_sha256=sha256(run/'summary.json'), facts=facts, totals=totals,
        fact_count=len(facts), question_count=len({f['question_id'] for f in facts}), selections=selections,
        qa_calls=0, limitations='Frozen partial witness list; excludes fine-only brochure; no answer accuracy')
    path = run/'evaluation_joint.json'
    if path.exists():
        assert read_json(path) == value
    else:
        durable(path, value)
    print(dict(facts=len(facts), totals=totals), flush=True)
    for f in facts:
        print(f['question_id'], f['fact'], f['conditions'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    import re
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id)
    evaluate(ROOT/'outputs/analysis'/args.run_id)
