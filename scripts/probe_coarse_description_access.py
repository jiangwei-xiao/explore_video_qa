"""两题零调用资格对照；负结果也冻结，不自动转入问答或参数搜索。"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from videoqa_audit.query_input_probe import BASE, FULL
from videoqa_budget32.selection import config
from videoqa_full.state import durable, exclusive
from videoqa_methods.coarse_description_access import select_coarse_description_access
from videoqa_runtime.common import read_json, sha256


def main():
    """接口：只重放两道已修复题，以目标帧保留评价一个固定资格改动，不使用答案选帧。"""
    directory = ROOT / 'outputs/analysis/coarse_description_access_probe_20261003_r1'
    source = ROOT / 'outputs/analysis/local_query_coverage200_20261003_r1'
    sources, comparisons = {}, []
    with exclusive(directory / 'probe.lock'):
        # 步骤1：读取冻结原池及已计算描述分数；目标索引只在选择结束后评价。
        for q, targets in [('409-3', [77, 80]), ('556-1', [42])]:
            paths = [BASE / 'pools' / (q + '.json'), BASE / 'features' / (q + '.npy'),
                     FULL / 'traces' / (q + '.json'), source / 'raw_scores' / (q + '.json')]
            sources.update({str(p): sha256(p) for p in paths})
            pool, features, old, scores = read_json(paths[0]), np.load(paths[1]), read_json(paths[2]), read_json(paths[3])['scores']
            indices, trace = select_coarse_description_access(pool, features, config(32, 32), old['region_budgets'], scores)
            # 步骤2：重放必须保持已有结果字节；完成记录不允许因结果不理想而替换。
            record = dict(selected_indices=indices, trace=trace, new_model_calls=0)
            path = directory / (q + '.json')
            if path.exists():
                assert read_json(path) == record
            else:
                durable(path, record)
            comparisons.append(dict(question_id=q, targets=targets, retained=[i for i in targets if i in indices],
                                    result_sha256=sha256(path)))
        # 步骤3：固定来源和代码，不将本次探针登记为已经问答的算法版本。
        for path in [Path(__file__).resolve(), ROOT / 'src/videoqa_methods/coarse_description_access.py']:
            sources[str(path)] = sha256(path)
        spec = dict(sources=sources, comparisons=comparisons, new_model_calls=0,
                    decision='No target witness retained; do not launch QA or expand this rule to 200 questions')
        path = directory / 'protocol.json'
        if path.exists():
            assert read_json(path) == spec
        else:
            durable(path, spec)
        print(comparisons, flush=True)


if __name__ == '__main__':
    main()
