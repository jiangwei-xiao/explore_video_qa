"""量化exp12局部支持的几何约束，不改变指数、种子或预算。"""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE


def main():
    """接口：固定200题统计种子权重与单种子有效范围，原问题仍通过锚点和预算参与。"""
    manifest = ROOT / 'data/manifests/video_mme_development_200_20260920.json'
    parent = ROOT / 'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
    sources = {str(p): sha256(p) for p in [manifest, Path(__file__)]}
    records = []
    # 步骤1：全量读取冻结开发区域，不只统计具有多个补充名额的有利子集。
    for row in read_json(manifest)['rows']:
        q = row['question_id']
        pp, tp = BASE / 'pools' / (q + '.json'), parent / 'traces' / (q + '.json')
        sources.update({str(p): sha256(p) for p in [pp, tp]})
        pool, trace = read_json(pp), read_json(tp)
        for r in pool['plan']['regions']:
            raw = np.asarray([pool['initial_scores'][i] for i in r['seed_indices']], dtype=np.float64)
            weights = raw / raw.max() if raw.max() > 0 else np.ones_like(raw)
            records.append(dict(question_id=q, region_id=r['region_id'], seeds=len(raw),
                                minimum_relative_weight=float(weights.min()),
                                quota=trace['region_budgets'][str(r['region_id'])]))
    # 步骤2：单种子时只对“锚点后的第一张”有c√(1-c)等式，不推广至整个序列。
    single = [r for r in records if r['seeds'] == 1]
    multi = [r for r in records if r['seeds'] > 1]
    summary = dict(questions=len({r['question_id'] for r in records}), regions=len(records),
        total_slots=sum(r['quota'] for r in records), single_seed_regions=len(single),
        single_seed_slots=sum(r['quota'] for r in single),
        single_seed_regions_with_extra_slot=sum(r['quota'] > 1 for r in single),
        single_seed_extra_slots=sum(max(0, r['quota']-1) for r in single),
        quantile_levels=[0, .1, .25, .5, .75, .9, 1],
        all_region_min_weight_quantiles=np.quantile([r['minimum_relative_weight'] for r in records],
                                                  [0, .1, .25, .5, .75, .9, 1]).tolist(),
        multi_seed_min_weight_quantiles=np.quantile([r['minimum_relative_weight'] for r in multi],
                                                  [0, .1, .25, .5, .75, .9, 1]).tolist())
    if summary['questions'] != 200 or summary['total_slots'] != 6400:
        raise ValueError('Frozen development denominator changed')
    # 步骤3：验证解析极值的数值一致性；此极值不是用于修改算法的阈值。
    grid = np.linspace(0, 1, 10001)
    maximum = float(grid[np.argmax(grid * np.sqrt(1-grid))])
    if abs(maximum - 2/3) > 1e-4:
        raise ValueError('Analytic first-step geometry mismatch')
    result = dict(sources=sources, summary=summary, records=records, new_model_calls=0,
        first_step=dict(function='c*sqrt(1-c)', analytic_peak=2/3, grid_peak=maximum),
        limitation='几何偏好不等于错误；单种子锚点和配额仍受原问题影响，不宣称局部选择完全无关问题')
    path = ROOT / 'outputs/analysis/seed_support_geometry200_20261004_r1/report.json'
    if path.exists():
        if read_json(path) != result:
            raise ValueError('Frozen geometry report changed')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        durable(path, result)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
