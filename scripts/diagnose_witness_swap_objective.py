"""人工已定位见证的单帧替换代价诊断；不构造可部署方法或调用问答。"""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE


def main():
    """接口：对三张冻结粗见证穷举同区单帧替换，区分前缀约束与代理目标偏好。

    目标帧来自既有人工诊断，仅作事后分析；最高代理值替换也不执行问答，
    不能称为算法收益、证据充分性或精度上限。
    """
    # 步骤1：固定目标、原exp12池和配额；不按输出新增目标或调整公式。
    parent = ROOT / 'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
    sources, records = {}, []
    for q, targets in [('409-3', [77, 80]), ('556-1', [42])]:
        paths = dict(pool=BASE / 'pools' / (q + '.json'), features=BASE / 'features' / (q + '.npy'),
                     trace=parent / 'traces' / (q + '.json'),
                     selection=parent / 'selections/RD-1.2-exp12' / (q + '.json'))
        sources.update({str(p): sha256(p) for p in paths.values()})
        pool, trace, item = (read_json(paths[k]) for k in ['pool', 'trace', 'selection'])
        features = np.load(paths['features'], allow_pickle=False)
        selected = set(item['selected_indices'])
        for target in targets:
            if not 0 <= target < pool['initial_count']:
                raise ValueError('Target must belong to frozen coarse pool')
            regions = [r for r in trace['regions'] if target in r['reference_coarse_indices']]
            if len(regions) != 1:
                raise ValueError('Ambiguous region membership')
            r = regions[0]
            quota = trace['region_budgets'][str(r['region_id'])]
            chosen = r['prefix'][:quota]
            if not set(chosen) <= selected or target in chosen:
                raise ValueError('Diagnostic target must be currently missing')
            refs = r['reference_coarse_indices']
            weights = np.asarray(r['reference_weights'], dtype=np.float64)

            def coverage(indices):
                """按冻结原问题加权覆盖重算，逐列更新以匹配原计算精度。"""
                covered = np.zeros(len(refs), dtype=np.float32)
                for index in indices:
                    covered = np.maximum(covered, np.clip(features[refs] @ features[index], 0, 1))
                return float(weights @ covered)

            # 步骤2：先复现旧目标，再保持锚点和名额，列出所有合法单帧替换。
            before = coverage(chosen)
            if abs(before - r['coverage_curve'][quota]) > 1e-6:
                raise ValueError('Frozen coverage reconstruction mismatch')
            alternatives = []
            for removed in chosen[1:]:
                changed = [i for i in chosen if i != removed] + [target]
                after = coverage(changed)
                alternatives.append(dict(removed_index=removed, removed_frame=pool['candidates'][removed],
                    coverage_after=after, coverage_delta=after-before,
                    weighted_objective_delta=r['density_weight']*(after-before)))
            # 步骤3：报告全部替换，不隐瞒最佳目标值仍下降或多帧关系未修复的情况。
            records.append(dict(question_id=q, target_index=target, target_frame=pool['candidates'][target],
                region_id=r['region_id'], quota=quota, selected_indices=chosen,
                prefix_position=r['prefix'].index(target)+1 if target in r['prefix'] else None,
                original_coverage=before, alternatives=alternatives,
                best_objective_delta=max((a['weighted_objective_delta'] for a in alternatives), default=None)))
    sources[str(Path(__file__))] = sha256(__file__)
    result = dict(sources=sources, records=records, new_model_calls=0,
        limitation='事后人工目标、同区单帧替换；不证明多帧组合或其他区域预算的最优性')
    output = ROOT / 'outputs/analysis/witness_swap_objective_20261004_r1/report.json'
    if output.exists():
        if read_json(output) != result:
            raise ValueError('Frozen diagnostic changed')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        durable(output, result)
    for r in records:
        print(r['question_id'], r['target_index'], 'quota', r['quota'], 'prefix', r['prefix_position'],
              'best_objective_delta', r['best_objective_delta'], flush=True)


if __name__ == '__main__':
    main()
