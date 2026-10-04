"""冻结13题的无人工目标覆盖交换探针，仅离线选择与路径评价。"""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE, FULL
from videoqa_audit.coverage_exchange_probe import best_single_exchange


def main():
    """接口：复用已有面板，每区域仅一次交换；人工见证仅在求解结束后评价。"""
    panel = ROOT / 'outputs/analysis/joint_temporal_support13_20261004_r1/analysis.json'
    parent = ROOT / 'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
    sources = {str(panel): sha256(panel), str(Path(__file__)): sha256(__file__),
               str(ROOT / 'src/videoqa_audit/coverage_exchange_probe.py'):
               sha256(ROOT / 'src/videoqa_audit/coverage_exchange_probe.py')}
    records = []
    # 步骤1：面板继承已冻结的13题，不按本探针结果重新选题。
    for row in read_json(panel)['records']:
        q = row['question_id']
        paths = dict(pool=BASE / 'pools' / (q + '.json'), features=BASE / 'features' / (q + '.npy'),
                     old_trace=FULL / 'traces' / (q + '.json'), trace=parent / 'traces' / (q + '.json'),
                     selection=parent / 'selections/RD-1.2-exp12' / (q + '.json'))
        sources.update({str(p): sha256(p) for p in paths.values()})
        pool = read_json(paths['pool']);features = np.load(paths['features'], allow_pickle=False)
        trace = read_json(paths['trace']);old_trace = read_json(paths['old_trace'])
        old = read_json(paths['selection'])['selected_indices']
        result, changes = [], []
        # 步骤2：候选归属与配额原样复用，图像相似矩阵只描述原覆盖目标。
        for region in trace['regions']:
            prior = next(r for r in old_trace['regions'] if r['region_id'] == region['region_id'])
            if prior['prefix'] != region['prefix']:
                raise ValueError('Frozen local prefix identity mismatch')
            ids = sorted(prior['members'], key=lambda i: (pool['candidates'][i]['source_pts'], i))
            lookup = {i:j for j,i in enumerate(ids)}
            quota = trace['region_budgets'][str(region['region_id'])]
            chosen = region['prefix'][:quota]
            if not set(chosen) <= set(old):
                raise ValueError('Old selection does not match quota')
            refs = region['reference_coarse_indices']
            sim = np.clip(features[refs] @ features[ids].T, 0, 1)
            selected, change = best_single_exchange(sim, region['reference_weights'],
                                                    [lookup[i] for i in chosen], lookup[region['prefix'][0]])
            if abs(change['before'] - region['coverage_curve'][quota]) > 1e-6:
                raise ValueError('Coverage differs from frozen implementation')
            result.extend(ids[i] for i in selected)
            change.update(region_id=region['region_id'], quota=quota,
                          weighted_delta=region['density_weight']*change['delta'])
            if change['changed']:
                change.update(added_index=ids[change['added_local']], removed_index=ids[change['removed_local']])
            changes.append(change)
        result.sort(key=lambda i: (pool['candidates'][i]['source_pts'], i))
        if len(result) != len(set(result)) or len(result) != len(old):
            raise ValueError('Budget or source uniqueness changed')
        # 步骤3：只保存反事实输入，不附用旧预测或写入正式方法选择目录。
        records.append(dict(question_id=q, old_selected=old, selected_indices=result,
                            selection=dict(video=pool['video'], selected_frames=[pool['candidates'][i] for i in result]),
                            changes=changes, weighted_delta=sum(c['weighted_delta'] for c in changes)))
        print(q, 'exchanges', sum(c['changed'] for c in changes), flush=True)
    output = ROOT / 'outputs/analysis/coverage_exchange13_20261004_r1/analysis.json'
    data = dict(sources=sources, records=records, new_model_calls=0, new_answers=0,
                rule='one best strict coverage exchange per original region; fixed quotas and anchors',
                limitation='候选和参考沿用exp12；目标增加不证明证据充分或准确率改善')
    if output.exists():
        if read_json(output) != data:
            raise ValueError('Resume fingerprint or result mismatch')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        durable(output, data)


if __name__ == '__main__':
    main()
