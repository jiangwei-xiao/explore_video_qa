"""核查exp23的冻结粗帧见证变化；仅评价，不向选帧器提供人工事实。"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable


def audit(output):
    """接口：输出42题75事实的精确源见证得失及已知修复帧路径，无模型调用。"""
    # 步骤1：固定评价资产和方法输入，记录哈希；不根据新答案改变分母。
    old = ROOT / 'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
    new = ROOT / 'outputs/methods/rd12_exp23__m64k32__prepare200__20261004_r1'
    facts = ROOT / 'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'
    sources, cache = {}, {}

    def read(path):
        sources[str(path)] = sha256(path)
        return read_json(path)

    entries = read(facts)['entries']
    if len(entries) != 75 or len({f['question_id'] for f in entries}) != 42:
        raise ValueError('Frozen evaluation denominator changed')

    def inputs(q):
        if q not in cache:
            a = read(old / 'selections/RD-1.2-exp12' / (q + '.json'))
            b = read(new / 'selections' / (q + '.json'))
            for s in (a, b):
                frames = s['selection']['selected_frames']
                if len(frames) != 32 or len({f['source_pts'] for f in frames}) != 32:
                    raise ValueError('Unexpected frame count or duplicate PTS')
            if a['selection']['video'] != b['selection']['video']:
                raise ValueError('Video identity mismatch')
            cache[q] = a, b
        return cache[q]

    # 步骤2：同题同视频以真实PTS匹配，细帧替代见证的语义价值留给画面审核。
    rows = []
    for f in entries:
        a, b = inputs(f['question_id'])
        hits = [bool(set(f['witness_pts']) &
                     {c['source_pts'] for c in s['selection']['selected_frames']}) for s in (a, b)]
        rows.append(dict(question_id=f['question_id'], fact_id=f['fact_id'],
                         description=f['description'], witness_pts=f['witness_pts'],
                         old_hit=hits[0], new_hit=hits[1]))

    # 步骤3：仅对共用初始粗池的既有修复帧定位前缀与配额，不跨池对应细帧编号。
    paths = []
    for q, indices in [('409-3', [77, 80]), ('556-1', [42]), ('191-3', [6])]:
        _, selected = inputs(q)
        trace = read(new / 'traces' / (q + '.json'))
        for i in indices:
            owners = [r for r in trace['regions'] if i in r['reference_coarse_indices']]
            if len(owners) != 1:
                raise ValueError('Coarse witness must have one owner')
            r = owners[0]
            position = r['prefix'].index(i) + 1 if i in r['prefix'] else None
            quota = trace['region_budgets'][str(r['region_id'])]
            hit = i in selected['selected_indices']
            if hit != (position is not None and position <= quota):
                raise ValueError('Prefix, quota and final selection disagree')
            paths.append(dict(question_id=q, coarse_index=i, region_id=r['region_id'],
                              prefix_position=position, quota=quota, selected=hit))

    # 步骤4：恢复只校验已有结果；禁止覆盖旧审计或触发任何问答。
    result = dict(sources=sources, facts=rows, supplemental_paths=paths,
                  summary=dict(questions=42, facts=75, old_hits=sum(r['old_hit'] for r in rows),
                               new_hits=sum(r['new_hit'] for r in rows),
                               gains=[r for r in rows if r['new_hit'] and not r['old_hit']],
                               losses=[r for r in rows if r['old_hit'] and not r['new_hit']],
                               new_model_calls=0, semantic_sufficiency='not_inferred'))
    output = Path(output)
    if output.exists():
        if read_json(output) != result:
            raise ValueError('Resume result or source fingerprint mismatch')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        durable(output, result)
    print(result['summary'], flush=True)
    print(paths, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    audit(parser.parse_args().output)
