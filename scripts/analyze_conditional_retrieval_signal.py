"""缓存检索信号核验：条件描述相对原题的log-odds增量，不生成新选帧问答成绩。"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive
from videoqa_audit.query_input_probe import BASE
from videoqa_methods.local_query_coverage import strict_rank_counts


def log_odds(values):
    """接口：仅用机器精度避免0/1对数无定义；不是可调置信度阈值，裁剪数量另记。"""
    a = np.asarray(values, dtype=np.float64)
    assert np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all()
    eps = np.finfo(np.float64).eps
    b = np.clip(a, eps, 1 - eps)
    return np.log(b) - np.log1p(-b), int(np.count_nonzero(a != b))


def main():
    """接口：全200题同一公式，42题75事实仅事后评价；保留已有分母，不调参或追加问答。"""
    # r2仅修复JSON配对记录的tuple/list恢复比较，评分公式及200题排名保持不变。
    run = ROOT / 'outputs/analysis/conditional_retrieval_signal200_20261003_r2'
    with exclusive(run / 'analysis.lock'):
        sources = {}
        def read(path):
            sources[str(path)] = sha256(path)
            return read_json(path)
        dataset = ROOT / 'data/manifests/video_mme_development_200_20260920.json'
        ids = [r['question_id'] for r in read(dataset)['rows']]
        facts = read(ROOT / 'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json')['entries']
        assert len(ids) == len(set(ids)) == 200
        assert len(facts) == 75 and len({f['question_id'] for f in facts}) == 42
        summaries, fact_results, file_hashes = [], [], {}
        # 步骤1：评分来自同一冻结粗池，描述四路均对称使用，不读取标准答案选择查询。
        for q in ids:
            pool = read(BASE / 'pools' / (q + '.json'))
            joined = read(ROOT / 'outputs/analysis/option_rank_dev200_20261001_r1/results' / (q + '.json'))
            original = np.asarray(pool['initial_scores'], dtype=np.float64)
            alternatives = np.asarray(joined['scores'], dtype=np.float64)
            n = pool['initial_count']
            assert original.shape == (n,) and alternatives.shape == (4, n)
            a, clipped_q = log_odds(original)
            b, clipped_options = log_odds(alternatives)
            # 步骤2：同图相减只是一种去除共同匹配分量的假设，不推导为条件概率或贝叶斯因子。
            gain = (b - a[None, :]).max(axis=0)
            joined_rank = strict_rank_counts(alternatives).max(axis=0)
            signals = dict(original_question=original, joined_best_rank=joined_rank, conditional_gain=gain)
            orders = {name: sorted(range(n), key=lambda i: (-float(values[i]), pool['candidates'][i]['source_pts'], i))
                      for name, values in signals.items()}
            ranks = {name: {i: r+1 for r, i in enumerate(order)} for name, order in orders.items()}
            lookup = {f['source_pts']: i for i, f in enumerate(pool['candidates'][:n])}
            selected = {name: set(order[:min(32, n)]) for name, order in orders.items()}
            per_facts = []
            for fact in [f for f in facts if f['question_id'] == q]:
                witness_ids = [lookup[p] for p in fact['witness_pts'] if p in lookup]
                assert witness_ids
                rec = dict(question_id=q, fact_id=fact['fact_id'], role=fact['role'], witness_indices=witness_ids,
                           outcomes={name: dict(best_rank=min(ranks[name][i] for i in witness_ids),
                                               normalized_best_rank=(min(ranks[name][i] for i in witness_ids)-1)/max(1, n-1),
                                               reachable_top32=bool(selected[name].intersection(witness_ids))) for name in signals})
                per_facts.append(rec); fact_results.append(rec)
            rec = dict(question_id=q, candidates=n, ranking_orders=orders,
                       clipped_original_values=clipped_q, clipped_option_values=clipped_options,
                       positive_gain_count=int((gain > 0).sum()), facts=per_facts,
                       new_model_calls=0, scope='Retrieval proposals only, not final 32-frame input or QA')
            path = run / 'questions' / (q + '.json')
            if path.exists():
                assert read_json(path) == rec
            else:
                durable(path, rec)
            file_hashes[str(path.relative_to(run))] = sha256(path)
            summaries.append(dict(question_id=q, clipped_original=clipped_q, clipped_options=clipped_options))
        # 步骤3：报告全部得失、按题宏平均和分母；既有标注缺失不能当作负标签。
        metrics = {}
        for name in ['original_question', 'joined_best_rank', 'conditional_gain']:
            groups = [[f for f in fact_results if f['question_id'] == q] for q in sorted({f['question_id'] for f in fact_results})]
            metrics[name] = dict(facts_reachable=sum(f['outcomes'][name]['reachable_top32'] for f in fact_results),
                fact_denominator=75, question_denominator=42,
                macro_reachability=float(np.mean([np.mean([f['outcomes'][name]['reachable_top32'] for f in fs]) for fs in groups])))
        comparisons = {}
        for reference in ['original_question', 'joined_best_rank']:
            comparisons[reference] = dict(improved=[[f['question_id'], f['fact_id']] for f in fact_results if f['outcomes']['conditional_gain']['reachable_top32'] and not f['outcomes'][reference]['reachable_top32']],
                lost=[[f['question_id'], f['fact_id']] for f in fact_results if not f['outcomes']['conditional_gain']['reachable_top32'] and f['outcomes'][reference]['reachable_top32']])
        sources[str(Path(__file__).resolve())] = sha256(__file__)
        spec = dict(sources=sources, result_hashes=file_hashes, new_model_calls=0,
                    hypothesis='max_o[logit ITM(question+option_o, frame) - logit ITM(question, frame)]',
                    limitations='Not calibrated conditional probability; no automatic selector or final QA; reused developmental witnesses')
        summary = dict(metrics=metrics, comparisons=comparisons,
                       clipped_original=sum(r['clipped_original'] for r in summaries),
                       clipped_options=sum(r['clipped_options'] for r in summaries), new_model_calls=0)
        for name, value in [('protocol.json', spec), ('summary.json', summary), ('facts.json', fact_results)]:
            path = run / name
            if path.exists():
                assert read_json(path) == value
            else:
                durable(path, value)
        print(summary, flush=True)


if __name__ == '__main__':
    main()
