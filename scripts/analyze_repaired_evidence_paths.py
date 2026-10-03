"""已完成修复诊断的失效路径核验；只分析旧分数和轨迹，不选择新模型输入。"""
from fractions import Fraction
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_audit.query_input_probe import BASE, FULL
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable


def main():
    """接口：明确区分全局粗帧资格、原补查窗口、描述前缀资格以及最终配额。"""
    sources, results = {}, []
    def read(path):
        sources[str(path)] = sha256(path)
        return read_json(path)
    # 步骤1：目标只供事后评价；保持原评分、源候选、窗口和局部轨迹不变。
    for q, targets in [('409-3', [77, 80]), ('556-1', [42])]:
        pool = read(BASE / 'pools' / (q + '.json'))
        old = read(FULL / 'traces' / (q + '.json'))
        new = read(ROOT / 'outputs/analysis/local_query_coverage200_20261003_r1/questions' / (q + '.json'))
        option = read(ROOT / 'outputs/analysis/option_only_dev200_20261003_r1/results' / (q + '.json'))
        candidates, n = pool['candidates'], pool['initial_count']
        scores = pool['initial_scores']
        rank = lambda values, i: 1 + sum(v > values[i] for v in values)
        trace = new['trace']
        for i in targets:
            f = candidates[i]
            t = (f['source_pts'] - pool['video']['start_pts']) * Fraction(pool['video']['time_base'])
            distances = [max(Fraction(r['left_fraction']) - t, t - Fraction(r['right_fraction']), 0)
                         for r in pool['plan']['regions']]
            r = next(r for r in old['regions'] if i in r['members'])
            e = next(r for r in trace['regions'] if i in r['members'])
            # 步骤2：四选项分数全部报告，不能选正确选项制造定位器；名次只在同一查询内解释。
            results.append(dict(question_id=q, candidate_index=i, source_frame=f,
                original_question_rank=rank(scores, i), original_question_score=scores[i],
                option_ranks=[rank(v, i) for v in option['scores']],
                option_scores=[v[i] for v in option['scores']], coarse_pool_count=n,
                original_windows=[dict(region_id=w['region_id'], left_fraction=w['left_fraction'],
                                       right_fraction=w['right_fraction']) for w in pool['plan']['regions']],
                nearest_original_window_distance_seconds=float(min(distances)),
                inside_original_window=min(distances) == 0,
                global_coarse_eligible=True,
                exp07=dict(region_id=r['region_id'], prefix_position=r['prefix'].index(i)+1 if i in r['prefix'] else None,
                           quota=old['region_budgets'][str(r['region_id'])]),
                exp17=dict(region_id=e['region_id'], description_eligible=i in e['original_window_indices'],
                           prefix_position=e['prefix'].index(i)+1 if i in e['prefix'] else None,
                           quota=trace['region_budgets'][str(e['region_id'])])))
    # 步骤3：追加独立分析记录，来源变化暂停，不修改已完成诊断输入或问答结果。
    output = ROOT / 'outputs/analysis/research_problem_map_20261003_r1/repaired_evidence_paths.json'
    record = dict(sources=sources, records=results, new_model_calls=0,
                  scope='Three human-localized coarse witnesses; no automated selection or algorithm accuracy')
    if output.exists():
        assert read_json(output) == record
    else:
        durable(output, record)
    for r in results:
        print({k: v for k, v in r.items() if k not in ['original_windows', 'source_frame']}, flush=True)


if __name__ == '__main__':
    main()
