"""将冻结事实见证迁移到exp12实际输入，只定位结构缺口，不生成新选择或答案。"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE, FULL

CURRENT = ROOT / 'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
FACTS = ROOT / 'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'


def witness_path(index, trace, selected):
    """接口：返回精确见证的前缀位置、现配额和理论配额上限，不判语义充分性。"""
    # 步骤1：粗参考归属必须唯一；来源缺失不能被静默归为未命中。
    regions = [r for r in trace['regions'] if index in r['reference_coarse_indices']]
    if len(regions) != 1:
        raise ValueError('Witness must have exactly one coarse reference owner')
    region = regions[0]
    prefix = region['prefix']
    position = prefix.index(index) + 1 if index in prefix else None
    quota = trace['region_budgets'][str(region['region_id'])]
    budget = sum(trace['region_budgets'].values())
    # 步骤2：每区至少保留一个锚点，所以单区最多取得K-(区域数-1)个名额。
    cap = min(len(prefix), budget - len(trace['regions']) + 1)
    chosen = index in selected
    if chosen != (position is not None and position <= quota):
        raise ValueError('Selection does not match frozen prefix and quota')
    if chosen:
        state = 'exact_witness_selected'
    elif position is None:
        state = 'outside_saved_prefix'
    elif position > cap:
        state = 'beyond_feasible_quota'
    else:
        state = 'quota_reachable_but_unselected'
    return dict(candidate_index=index, region_id=region['region_id'], prefix_position=position,
                quota=quota, max_feasible_quota=cap, selected=chosen, structural_state=state,
                additional_slots=None if position is None else max(0, position-quota))


def build(run):
    """接口：固定42题75事实与三张人工修复粗帧，恢复仅核对同一输出，不调用模型。"""
    sources = {str(Path(__file__)): sha256(__file__)}

    def read(path):
        sources[str(path)] = sha256(path)
        return read_json(path)

    # 步骤1：事实只供评价；固定旧目录，不按当前预测删改分母。
    entries = read(FACTS)['entries']
    if len(entries) != 75 or len({f['question_id'] for f in entries}) != 42:
        raise ValueError('Frozen proxy denominator changed')
    fact_rows, supplemental, cache = [], [], {}

    def inputs(q):
        if q not in cache:
            pool = read(BASE / 'pools' / (q+'.json'))
            old = read(FULL / 'traces' / (q+'.json'))
            trace = read(CURRENT / 'traces' / (q+'.json'))
            item = read(CURRENT / 'selections/RD-1.2-exp12' / (q+'.json'))
            if any(a['prefix'] != b['prefix'] for a, b in zip(old['regions'], trace['regions'])):
                raise ValueError('Frozen prefixes changed')
            old_selected = {i for r in old['regions'] for i in r['prefix'][:old['region_budgets'][str(r['region_id'])]]}
            cache[q] = pool, old, trace, old_selected, set(item['selected_indices'])
        return cache[q]

    def paths(q, indices):
        pool, old, trace, old_selected, selected = inputs(q)
        result = []
        for i in sorted(set(indices)):
            if not 0 <= i < pool['initial_count']:
                raise ValueError('Witness is not an original coarse candidate')
            result.append(dict(source_frame=pool['candidates'][i],
                exp07=witness_path(i, old, old_selected), exp12=witness_path(i, trace, selected)))
        return result

    # 步骤2：逐事实检查全部已列见证，重复PTS不能造成重复计数。
    for fact in entries:
        q = fact['question_id']
        pool = inputs(q)[0]
        lookup = {c['source_pts']:i for i,c in enumerate(pool['candidates'][:pool['initial_count']])}
        indices = [lookup[p] for p in fact['witness_pts']]
        evidence = paths(q, indices)
        if not evidence:
            raise ValueError('Fact has no frozen witness')
        fact_rows.append(dict(question_id=q, fact_id=fact['fact_id'], role=fact['role'],
            description=fact['description'], paths=evidence,
            old_hit=any(p['exp07']['selected'] for p in evidence),
            current_hit=any(p['exp12']['selected'] for p in evidence),
            any_quota_reachable=any(p['exp12']['structural_state']=='quota_reachable_but_unselected' for p in evidence)))
    for q, indices in [('409-3',[77,80]), ('556-1',[42])]:
        supplemental.append(dict(question_id=q, paths=paths(q, indices),
            basis='Previously frozen manual Qwen32 repair; separate from 75-fact denominator'))
    # 步骤3：精确见证计数与答案好坏分离，不将未标注的细帧当作无证据。
    summary = dict(questions=42, facts=75, exp07_exact_hits=sum(r['old_hit'] for r in fact_rows),
        exp12_exact_hits=sum(r['current_hit'] for r in fact_rows),
        lost=[dict(question_id=r['question_id'],fact_id=r['fact_id']) for r in fact_rows if r['old_hit'] and not r['current_hit']],
        gained=[dict(question_id=r['question_id'],fact_id=r['fact_id']) for r in fact_rows if not r['old_hit'] and r['current_hit']],
        missing_quota_reachable=sum(not r['current_hit'] and r['any_quota_reachable'] for r in fact_rows),
        missing_not_quota_reachable=sum(not r['current_hit'] and not r['any_quota_reachable'] for r in fact_rows),
        new_model_calls=0, semantic_sufficiency='not_inferred',
        limitation='Exact coarse witnesses omit alternative fine frames; theoretical quota reachability is not a proposed allocation')
    result = dict(sources=sources, summary=summary, facts=fact_rows, supplemental=supplemental)
    output = run / 'witness_paths.json'
    if output.exists():
        if read_json(output) != result:
            raise ValueError('Resume fingerprint or result mismatch')
    else:
        run.mkdir(parents=True, exist_ok=True)
        durable(output, result)
    print(summary, flush=True)
    for r in supplemental:
        print(r['question_id'], [p['exp12'] for p in r['paths']], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    build(Path(parser.parse_args().run))
