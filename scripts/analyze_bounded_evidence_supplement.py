"""开发200题有界补充离线检查；仅复用缓存，无新模型调用或问答入口。"""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive, utc
from videoqa_audit.query_input_probe import BASE, FULL
from videoqa_methods.bounded_evidence_supplement import supplement


def main():
    """接口：固定200题、最多4次替换；先冻结规则，事实仅用于事后评价，不选择参数。"""
    run = ROOT / 'outputs/analysis/bounded_evidence_supplement200_20261003_r2'
    with exclusive(run / 'analysis.lock'):
        dataset = ROOT / 'data/manifests/video_mme_development_200_20260920.json'
        fact_path = ROOT / 'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'
        ids = [r['question_id'] for r in read_json(dataset)['rows']]
        facts = read_json(fact_path)['entries']
        assert len(ids) == len(set(ids)) == 200 and len(facts) == 75
        sources = {str(p): sha256(p) for p in [dataset, fact_path, Path(__file__).resolve(),
                   ROOT / 'src/videoqa_methods/bounded_evidence_supplement.py']}
        protocol = dict(sources=sources, question_ids=ids, max_swaps=4, frame_budget=32,
                        new_model_calls=0, scope='offline; addition by positive conditional gain times novelty; deletion by original coverage loss')
        p = run / 'protocol.json'
        if p.exists():
            assert read_json(p) == protocol
        else:
            durable(p, protocol)
        records = []
        # 步骤1：同一选择函数对全200题执行，读取答案文件仅为恢复冻结源帧，不向函数提供答案。
        for q in ids:
            pp, fp = BASE / 'pools' / (q+'.json'), BASE / 'features' / (q+'.npy')
            tp = FULL / 'traces' / (q+'.json')
            rp = FULL / 'imported/results/RD-1.2-exp07' / (q+'.json')
            sp = ROOT / 'outputs/analysis/option_rank_dev200_20261001_r1/results' / (q+'.json')
            sources_q = {str(p): sha256(p) for p in [pp, fp, tp, rp, sp]}
            check = read_json(BASE / 'selection_checkpoints' / (q+'.json'))
            for p in [pp, fp]:
                assert sources_q[str(p)] == check['files'][str(p.relative_to(BASE))]
            dest = run / 'questions' / (q+'.json')
            if dest.exists():
                record = read_json(dest)
                assert record['sources'] == sources_q and record['protocol_sha256'] == sha256(run/'protocol.json')
            else:
                pool, trace, result = read_json(pp), read_json(tp), read_json(rp)
                old = [f['candidate_index'] for f in result['selection']['selected_frames']]
                selected, changes = supplement(pool, np.load(fp, allow_pickle=False), old, trace, read_json(sp)['scores'])
                old_pts = {pool['candidates'][i]['source_pts'] for i in old}
                new_pts = {pool['candidates'][i]['source_pts'] for i in selected}
                fact_records = [dict(fact_id=f['fact_id'], role=f['role'],
                                    old_hit=bool(old_pts.intersection(f['witness_pts'])),
                                    new_hit=bool(new_pts.intersection(f['witness_pts']))) for f in facts if f['question_id']==q]
                record = dict(question_id=q, original_indices=old, selected_indices=selected, trace=changes,
                              facts=fact_records, sources=sources_q, protocol_sha256=sha256(run/'protocol.json'), new_model_calls=0)
                durable(dest, record)
            records.append(record)
            durable(run/'status.json', dict(status='running', completed=len(records), total=200,
                                           pid=os.getpid(), sid=os.getsid(0), utc=utc()))
        # 步骤2：事实分母保持42题75项；精确见证损失需再看替代画面，不能自动判语义丢失。
        fact_rows = [dict(question_id=r['question_id'], **f) for r in records for f in r['facts']]
        assert len(fact_rows)==75 and len({f['question_id'] for f in fact_rows})==42
        grouped = [r['facts'] for r in records if r['facts']]
        summary = dict(status='completed', question_count=200,
                       changed_questions=sum(bool(r['trace']['steps']) for r in records),
                       total_replacements=sum(len(r['trace']['steps']) for r in records),
                       old_hits=sum(f['old_hit'] for f in fact_rows), new_hits=sum(f['new_hit'] for f in fact_rows),
                       old_macro=float(np.mean([np.mean([f['old_hit'] for f in fs]) for fs in grouped])),
                       new_macro=float(np.mean([np.mean([f['new_hit'] for f in fs]) for fs in grouped])),
                       gained=[f for f in fact_rows if f['new_hit'] and not f['old_hit']],
                       lost=[f for f in fact_rows if f['old_hit'] and not f['new_hit']],
                       new_model_calls=0, interpretation='Not algorithm accuracy; exact known witness proxy only')
        # 步骤3：完成恢复不覆写统计；返回结果供有界画面审核，不触发QA。
        if (run/'summary.json').exists():
            assert read_json(run/'summary.json') == summary
        else:
            durable(run/'summary.json', summary)
        durable(run/'status.json', dict(status='completed', completed=200, pid=os.getpid(), sid=os.getsid(0), utc=utc()))
        print(summary, flush=True)


if __name__ == '__main__':
    main()
