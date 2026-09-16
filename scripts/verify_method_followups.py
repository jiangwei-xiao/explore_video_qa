"""首轮改进交付验收：只核验并封存，不执行生成或改写历史证据。"""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import shutil
import sys
from datetime import datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT, read_json, sha256, write_json
from videoqa_methods.followups import protocol, freeze_proxy, validate_e, utc
from videoqa_methods.run import verify_reference_and_baseline


def verify(run):
    """接口：核对调用预算/顺序、特征、旧资产、人工review绑定和测试，并保存交付清单。"""
    run=Path(run).resolve(); rows=read_json(ROOT/'data/manifests/video_mme_development.json')['rows']
    spec=read_json(run/'protocol.json'); fingerprint=sha256(run/'protocol.json')
    if protocol()!=spec: raise ValueError('Execution identity changed')
    verify_reference_and_baseline()
    assert read_json(run/'proxy_manifest.json')==freeze_proxy(rows)
    # 步骤1：每题三类生成恰好一次且明确返回，所有输入/特征和上游通过。
    question_ids={r['question_id'] for r in rows}; calls={}
    for kind in ('scope','answer','query'):
        paths=list((run/'calls'/kind).glob('*.json')); assert len(paths)==50
        records=[read_json(p) for p in paths]; calls[kind]=records
        assert {x['question_id'] for x in records}==question_ids
        assert all(x['status']=='completed' and x['generation_state'].get('started') and x['generation_state'].get('returned') for x in records)
    assert max(x['completed_at_utc'] for x in calls['answer'])<=min(x['started_at_utc'] for x in calls['query'])
    freeze=read_json(run/'query_freeze.json')
    assert max(x['completed_at_utc'] for x in calls['query'])<=freeze['utc']
    for row in rows:
        qid=row['question_id']; e=read_json(run/'results/E'/f'{qid}.json'); q=read_json(run/'results/Q'/f'{qid}.json')
        validate_e(e,row,fingerprint)
        assert e['timing_kind']=='direct_no_application_cache' and not e['calls_reused']
        assert sha256(run/q['feature_file'])==q['feature_sha256']
        assert sha256(run/'queries'/f'{qid}.json')==freeze['sha256'][qid]==q['query_sha256']
        assert q['protocol_sha256']==fingerprint and len({r['source_pts'] for r in q['pool']['selected_frames']})==16
    assert not list((run/'failures').glob('*.json'))
    # 步骤2：验证语义review与实际读取图板哈希，不把自动目录失配改写为新事实。
    fidelity=read_json(run/'review/query_fidelity.json'); assert len(fidelity['questions'])==50
    for item in fidelity['questions']: assert sha256(run/'queries'/f'{item["question_id"]}.json')==item['query_sha256']
    for item in read_json(run/'review/case_review.json')['cases']: assert sha256(run/item['board'])==item['board_sha256']
    decision=read_json(run/'review/decisions.json')
    elapsed=(datetime.fromisoformat(decision['review_finished_at_utc'])-datetime.fromisoformat(decision['review_time_upper_bound_start'])).total_seconds()
    assert elapsed<=1800
    tests=(run/'tests_final.log').read_text(); assert '129 passed, 1 skipped' in tests
    # 步骤3：将非执行分析代码和文档另存快照，避免改变已冻结执行指纹。
    paths=[ROOT/'src/videoqa_methods/followups_report.py',ROOT/'tests/test_method_followups.py']
    paths += [ROOT/'scripts'/p for p in ('summarize_method_followups.py','render_followup_cases.py','finalize_method_followups.py','record_followup_review.py','verify_method_followups.py')]
    paths += list((ROOT/'docs/experiments/video_mme_50/2026-09-15/method_followups').glob('*.md'))
    hashes={}
    for p in paths:
        target=run/'analysis_snapshot'/p.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
        hashes[str(p.relative_to(ROOT))]=sha256(p)
    write_json(run/'analysis_code_manifest.json',hashes)
    acceptance=dict(status='completed_with_explicit_interpretation_limits',run_id=run.name,utc=utc(),
        E_results=50,Q_results=50,final_answer_calls=50,scope_calls=50,query_calls=50,query_final_answers=0,
        generation_failures=0,generation_retries=0,history_unchanged=True,execution_identity_unchanged=True,
        all_upstream_and_16frame_3360token_checks_passed=True,tests=dict(passed=129,skipped=1,sha256=sha256(run/'tests_final.log')),
        review_conservative_upper_bound_seconds=elapsed,query_semantic_drift=['491-2'],user_agreement=None,
        resume_check='all completed; no model load or generation repeated',
        analysis_recovery='first statistics run used different CPU threads; replay exact with frozen 2-thread settings; no experiment repeated',
        new_commits=0,pushes=0)
    write_json(run/'acceptance.json',acceptance)
    manifest={str(p.relative_to(run)):sha256(p) for p in sorted(run.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and not p.name.endswith('.lock')}
    write_json(run/'artifact_manifest.json',manifest)
    print(acceptance,flush=True); print('artifact_files',len(manifest),flush=True)


if __name__=='__main__': verify(sys.argv[1])
