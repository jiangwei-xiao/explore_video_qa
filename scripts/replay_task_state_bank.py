"""问题条件状态的银行内重放：零前向，非完整候选池，旧输入必须精确复现。"""
import argparse
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from videoqa_full.state import durable
from videoqa_runtime.common import read_json, sha256
from videoqa_methods.task_conditioned_state import conditional_redundancy
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.density_selection import frame_key
from probe_task_conditioned_state import verify, verify_completed


def replay(root):
    """接口：银行子集上重算局部贪心，L和配额固定；不把子集结果冒充全池输入。"""
    spec = verify(root); verify_completed(root); reports = []
    # 步骤1：按冻结源对应特征；所有父胜出帧已在银行，因此旧规则可精确控制重放。
    for name, item in spec['profiles'].items():
        profile, q = name.split(':'); pool = read_json(item['pool'])
        parent = read_json(item['parent_trace']); parent = parent['exp07'] if item['budget'] == 8 else parent
        old = np.load(item['features'], allow_pickle=False)[item['global_mapping']]
        representation = read_json(root / 'features' / (q + '.json'))
        mapping = item['global_mapping']; new_all = []; control_all = []
        for region, plan in zip(parent['regions'], pool['plan']['regions']):
            support, _, _ = weighted_support(old, region['members'], plan['seed_indices'], pool['initial_scores'])
            support = dict(zip(region['members'], support))
            eligible = [i for i in region['members'] if str(mapping[i]) in representation['vectors']]
            quota = parent['region_budgets'][str(region['region_id'])]
            # 步骤2：先运行旧D控制，再仅换新D；银行没有其他源帧，不新增计算或补采样。
            for task_conditioned in [False, True]:
                chosen = [plan['anchor_index']]
                while len(chosen) < quota:
                    values = []
                    for i in eligible:
                        if i in chosen:
                            continue
                        if task_conditioned:
                            context = [str(mapping[j]) for j in chosen]; index = str(mapping[i])
                            d = conditional_redundancy(old[i], old[chosen], representation['vectors'][index],
                                np.asarray([representation['vectors'][j] for j in context]),
                                representation['valid'][index], [representation['valid'][j] for j in context])
                        else:
                            d = float(np.clip(old[chosen] @ old[i], 0, 1).max())
                        values.append((float(support[i]) * math.sqrt(1 - d), i))
                    _, winner = min(values, key=lambda value: (-value[0], *frame_key(pool['candidates'][value[1]])))
                    chosen.append(winner)
                if task_conditioned:
                    new_all += chosen
                else:
                    control_all += chosen
        expected = sorted(item['parent_selected_indices'], key=lambda i: frame_key(pool['candidates'][i]))
        control = sorted(control_all, key=lambda i: frame_key(pool['candidates'][i]))
        if control != expected:
            raise ValueError('Restricted-bank original control does not replay parent: ' + name)
        ids = sorted(new_all, key=lambda i: frame_key(pool['candidates'][i]))
        value = dict(question_id=q, profile=profile,
            selection=dict(video=pool['video'], selected_frames=[pool['candidates'][i] for i in ids]),
            selected_indices=ids, old_control_exact=True,
            scope='Frozen bank restricted competition, not full candidate selection',
            new_model_calls=0, new_QA_calls=0)
        # 步骤3：恢复只接受相同结果，不覆盖第一次重放证据或使用历史答案填分数。
        path = root / 'bank_replay' / profile / (q + '.json')
        if path.exists() and read_json(path) != value:
            raise ValueError('Completed bank replay changed')
        durable(path, value); reports.append(dict(question_id=q, profile=profile, ids=ids))
    summary = dict(reports=reports, old_controls_exact=len(reports), new_model_calls=0, new_QA_calls=0)
    path = root / 'bank_replay/summary.json'
    if path.exists() and read_json(path) != summary:
        raise ValueError('Completed bank summary changed')
    durable(path, summary)
    durable(root / 'bank_replay/source.json', dict(script=str(Path(__file__)),
        script_sha256=sha256(Path(__file__)), protocol_sha256=sha256(root / 'protocol.json')))
    return summary


def main():
    """接口：指定已完成能力运行，旧源/完整返回变化时暂停，始终不调用模型。"""
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum():
        parser.error('Invalid run ID')
    result = replay(ROOT / 'outputs/diagnostics' / args.run_id)
    print(dict(controls=result['old_controls_exact'], new_model_calls=0, new_QA_calls=0))


if __name__ == '__main__':
    main()
