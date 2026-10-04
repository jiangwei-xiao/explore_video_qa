"""复算已冻结目标帧与实际赢家的竞争差异，不产生新选择或模型调用。"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
import argparse
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE, FULL
from videoqa_methods.weighted_seed_support import weighted_support

CURRENT = ROOT / 'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
TARGETS = {'409-3': [77, 80], '556-1': [42], '395-2': [276],
           '584-3': [582], '807-2': [1646], '383-2': [944]}
CONTROL_TARGETS = {'538-1': [9, 10, 11], '705-1': [2383],
                   '893-1': [699, 704, 706], '305-3': [198, 199],
                   '170-2': [10], '191-3': [6, 17, 28]}


def main(run, panel='initial'):
    """接口：复算首批或完整十二题面板，追踪支持种子与去重参照。

    目标只用于诊断，不进入选择规则；当前指标描述数值竞争，不解释语义关系。
    """
    sources = {str(Path(__file__)): sha256(__file__)}
    records = []
    # 步骤1：只加载历史池、特征和exp12轨迹，绑定全部读取来源。
    targets_by_question = TARGETS if panel == 'initial' else {**TARGETS, **CONTROL_TARGETS}
    for q, targets in targets_by_question.items():
        pool_path = BASE / 'pools' / (q + '.json')
        trace_path = CURRENT / 'traces' / (q + '.json')
        feature_path = BASE / 'features' / (q + '.npy')
        prefix_path = FULL / 'traces' / (q + '.json')
        for path in [pool_path, trace_path, feature_path, prefix_path]:
            sources[str(path)] = sha256(path)
        pool, trace = read_json(pool_path), read_json(trace_path)
        prefix_trace = read_json(prefix_path)
        if {r['region_id']: r['prefix'] for r in trace['regions']} != {r['region_id']: r['prefix'] for r in prefix_trace['regions']}:
            raise ValueError('exp07与exp12局部前缀不一致')
        f = np.load(feature_path, allow_pickle=False)
        plans = {r['region_id']: r for r in pool['plan']['regions']}
        for target in targets:
            region, = [r for r in prefix_trace['regions'] if target in r['members']]
            ids = region['members']
            rid = region['region_id']
            seeds = plans[rid]['seed_indices']
            support, weights, _ = weighted_support(f, ids, seeds, pool['initial_scores'])
            lookup = {i: j for j, i in enumerate(ids)}
            chosen = [region['prefix'][0]]
            D = np.clip(f[ids] @ f[chosen[0]], 0, 1)
            steps = []

            def describe(i):
                """返回单候选数值及具体来源，不把相似度称为证据相关性。"""
                seed = max(seeds, key=lambda s: weights[s] * float(np.clip(f[i] @ f[s], 0, 1)))
                ref = max(chosen, key=lambda s: float(np.clip(f[i] @ f[s], 0, 1)))
                j = lookup[i]
                return dict(candidate_index=i, source_frame=pool['candidates'][i],
                            support=float(support[j]), redundancy=float(D[j]),
                            score=float(support[j]) * math.sqrt(1 - float(D[j])),
                            strongest_seed=seed, nearest_selected=ref,
                            original_itm=pool['initial_scores'][i] if i < pool['initial_count'] else None)

            # 步骤2：逐次独立计算所有候选，实际胜者必须与冻结轨迹一致。
            for expected in region['local_steps']:
                if target in chosen:
                    break
                values = {i: float(support[lookup[i]]) * math.sqrt(1-float(D[lookup[i]]))
                          for i in ids if i not in chosen}
                winner = min(values, key=lambda i: (-values[i], pool['candidates'][i]['source_pts'], i))
                if winner != expected['candidate_index'] or abs(values[winner] - expected['local_score']) > 1e-6:
                    raise ValueError('历史胜者或分数未复现')
                steps.append(dict(position=len(chosen)+1, target=describe(target), winner=describe(winner)))
                chosen.append(winner)
                D = np.maximum(D, np.clip(f[ids] @ f[winner], 0, 1))
            records.append(dict(question_id=q, target=target, region_id=rid,
                quota=trace['region_budgets'][str(rid)], anchor=region['prefix'][0],
                target_position=region['prefix'].index(target)+1 if target in region['prefix'] else None,
                steps=steps))
    # 步骤3：派生结果原子落盘；恢复核对来源与全部轨迹，不覆盖不一致结果。
    result = dict(sources=sources, new_model_calls=0, records=records)
    path = run / 'competition.json'
    if path.exists():
        if read_json(path) != result:
            raise ValueError('恢复指纹或分析结果改变')
    else:
        run.mkdir(parents=True, exist_ok=True)
        durable(path, result)
    for r in records:
        print(r['question_id'], r['target'], 'quota', r['quota'], 'position', r['target_position'])
        for s in r['steps'][:min(r['quota'], 4)]:
            print(s['position'], [(k, s[k]['candidate_index'], round(s[k]['support'], 4),
                                  round(s[k]['redundancy'], 4), s[k]['nearest_selected']) for k in ['target','winner']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--panel', choices=['initial', 'all'], default='initial')
    args = parser.parse_args()
    main(args.run, args.panel)
