"""exp07完整结果补充核验与性能统计；必须完成2500新调用后执行，不读取中途成绩。"""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import rows,durable,utc
from videoqa_runtime.baseline_report import distribution
from videoqa_budget32.prefix_full import BASE,DEV,METHOD


def main(run):
    """接口：独立字母计分、导入来源、同输入复现与显存/阶段成本的全量检查。"""
    run = Path(run); spec = read_json(run/'protocol.json'); data = rows()
    assert read_json(run/'completion_audit.json')['status']=='passed'
    assert read_json(run/'independent_selection_audit.json')['status']=='passed'
    imported,new = set(spec['imported_ids']),set(spec['new_ids'])
    assert len(imported)==200 and len(new)==2500 and not imported&new
    assert imported|new=={r['question_id'] for r in data}
    # 步骤1：结果分母、已开始调用数和来源身份闭合，导入不伪装为新调用。
    for folder in ('results','calls','raw_generations'):
        assert {p.stem for p in (run/folder/METHOD).glob('*.json')}==new
        assert {p.stem for p in (run/'imported'/folder/METHOD).glob('*.json')}==imported
    full = {}; references = {m:{} for m in ('BASE-Uniform','BASE-BLIP-TopK','RD-1.2')}
    same = {m:[] for m in references}; parity = {m:[] for m in references}
    hashes = {}
    for row in data:
        q = row['question_id']; p = (run/'imported' if q in imported else run)/'results'/METHOD/(q+'.json')
        x = read_json(p); full[q] = x; gold = 'ABCD'[row['answer_index']]
        assert x['correct']==(x['answer']['parsed_answer']==gold)
        assert x['question']==row['question'] and x['options']==row['options']
        assert x['protocol_sha256']==sha256((DEV if q in imported else run)/'protocol.json')
        for m in references:
            r = read_json(BASE/'results'/m/(q+'.json')); references[m][q] = r
            assert r['correct']==(r['answer']['parsed_answer']==gold)
            a,b = x['answer']['observed'],r['answer']['observed']
            parity[m].append(dict(question_id=q,equal_frames=a['source_frames']==b['source_frames'],
                equal_size=a['processed_size']==b['processed_size'],equal_visual_tokens=a['visual_tokens']==b['visual_tokens'],
                equal_prefill=a['prefill_tokens']==b['prefill_tokens']))
            key = lambda result:[(f['source_pts'],f['source_frame_index']) for f in result['selection']['selected_frames']]
            if key(x)==key(r):
                same[m].append(dict(question_id=q,rgb_equal=x['rgb_sha256']==r['rgb_sha256'],
                    tensor_equal=a['processor_pixel_sha256']==b['processor_pixel_sha256'],
                    token_ids_equal=a['input_token_ids']==b['input_token_ids'],
                    answer_equal=x['answer']['parsed_answer']==r['answer']['parsed_answer']))
        hashes[str(p.relative_to(run))] = sha256(p)
    totals = {METHOD:sum(x['correct'] for x in full.values()),**{m:sum(x['correct'] for x in rs.values()) for m,rs in references.items()}}
    assert [totals[m] for m in references]==[1644,1650,1670]
    summary = read_json(run/'summary.json')
    for m,c in totals.items(): assert summary['accuracy'][m]['all2700']['all']['correct']==c
    # 步骤2：只把2500条本批次计入主耗时分布；旧200题不混作本批GPU实测。
    performance = {}
    fields = ['qa_through_generation_seconds','generation_seconds','peak_allocated_gib','peak_reserved_gib','visual_tokens','prefill_tokens']
    for layer in ('all','short','medium','long'):
        group = [full[r['question_id']]['answer']['observed'] for r in data if r['question_id'] in new and (layer=='all' or r['stratum']==layer)]
        performance[layer] = dict(n=len(group),**{k:distribution([x[k] for x in group]) for k in fields})
    failures = list((run/'failures').glob('*.json')); assert not failures
    durable(run/'result_supplement.json',dict(status='passed',audit_code_sha256=sha256(Path(__file__)),direct_correct=totals,new_generated=2500,imported=200,
        performance_new_batch=performance,same_input_checks=same,input_budget_parity=parity,
        differing_same_input_answers={m:[x for x in xs if not x['answer_equal']] for m,xs in same.items()},
        failure_count=0,selection_audit_sha256=sha256(run/'independent_selection_audit.json'),
        completion_audit_sha256=sha256(run/'completion_audit.json'),utc=utc()))
    durable(run/'result_file_hashes.json',hashes)
    print('Direct totals verified',totals,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run',required=True)
    main(parser.parse_args().run)
