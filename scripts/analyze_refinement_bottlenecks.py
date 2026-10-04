"""从冻结全量配对结果分解收益来源，不据标签路由方法或重新筛开发题。"""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable


def main():
    """接口：报告全量和开发视频外的题型配对得失；仅作描述，不声称子组显著性。"""
    source = ROOT / 'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1/per_question.json'
    manifest = ROOT / 'data/manifests/video_mme_development_200_20260920.json'
    # 步骤1：复用已独立审计的逐题结果，显式验证主分母和视频隔离分母。
    rows = read_json(source)
    dev = {r['video_id'] for r in read_json(manifest)['rows']}
    outside = [r for r in rows if r['video_id'] not in dev]
    if len(rows) != 2700 or len({r['question_id'] for r in rows}) != 2700 or len(dev) != 200 or len(outside) != 2100:
        raise ValueError('Frozen population mismatch')
    methods = ['RD-1.2-exp12', 'BASE-Uniform', 'BASE-BLIP-TopK']
    totals = {m: sum(r['methods'][m]['correct'] for r in rows) for m in methods}
    if totals != dict(zip(methods, [1710, 1644, 1650])):
        raise ValueError('Full result totals do not match accepted audit')
    tables = {}
    # 步骤2：全部官方题型同时报告，不只选择有收益的类别；共同错不等于可由选帧修复。
    for scope, group in [('full2700', rows), ('outside_development_videos2100', outside)]:
        table = []
        for task in sorted({r['task_type'] for r in group}):
            part = [r for r in group if r['task_type'] == task]
            count = Counter()
            for r in part:
                a, u, t = [bool(r['methods'][m]['correct']) for m in methods]
                count.update(n=1, exp12=int(a), uniform=int(u), topk=int(t),
                             gained_vs_uniform=int(a and not u), lost_vs_uniform=int(u and not a),
                             all_three_wrong=int(not a and not u and not t))
            table.append(dict(task_type=task, **count, net_vs_uniform=count['exp12']-count['uniform'],
                              difference_pp=100*(count['exp12']-count['uniform'])/count['n'],
                              question_ids=[r['question_id'] for r in part]))
        if sum(r['n'] for r in table) != len(group):
            raise ValueError('Task grouping not exhaustive')
        tables[scope] = table
    # 步骤3：冻结来源与结果；恢复只检查，不改变历史计分或启动生成。
    output = ROOT / 'outputs/analysis/refinement_bottlenecks_20261004_r1/task_pairs.json'
    result = dict(sources={str(p): sha256(p) for p in [source, manifest, Path(__file__)]},
                  tables=tables, totals=totals, new_model_calls=0,
                  limitation='官方题型不等于证据布局；多子组描述性结果，不作显著性或盲测声明')
    if output.exists():
        if read_json(output) != result:
            raise ValueError('Result or source fingerprint changed')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        durable(output, result)
    print(totals, 'full2700 and outside2100 task tables verified', flush=True)


if __name__ == '__main__':
    main()
