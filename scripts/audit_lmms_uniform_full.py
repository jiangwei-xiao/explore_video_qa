#!/usr/bin/env python3
"""全量均匀验收：仅CPU核查与已完成恢复，不允许新增模型调用。"""
import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, rows, utc
from videoqa_full.prepare import verify_frozen_rd
from videoqa_lmms_baseline.run import verify, validate, METHOD, PYTHON
from videoqa_lmms_bridge.frames import FrameProvider


def audit(run):
    """接口：完整结果才可验收；核对源索引、历史指纹和恢复前后账本字节。"""
    # 步骤1：重算2700题，并用独立linspace及完整PTS索引核验900视频。
    assert read_json(run / 'execution.json')['status'] == 'completed'
    spec = verify(run)
    fp = sha256(run / 'protocol.json')
    data = rows()
    result = {r['question_id']: validate(run, r, spec, fp) for r in data}
    assert len(result) == 2700
    assert Counter(r['stratum'] for r in data) == dict(short=900, medium=900, long=900)
    for vid, source in spec['source_uniform'].items():
        record = read_json(source['path'])
        assert sha256(source['pts_path']) == source['pts_sha256']
        pts = np.load(source['pts_path'], allow_pickle=False)
        indices = np.linspace(0, len(pts)-1, min(16, len(pts)), dtype=int).tolist()
        assert indices == [f['source_frame_index'] for f in record['new_frames']]
        assert [int(pts[i]) for i in indices] == [f['source_pts'] for f in record['new_frames']]
    for name, h in spec['code_sha256'].items():
        assert sha256(run / 'code_snapshot' / name) == h
    # 步骤2：历史基线及RD冻结资产必须保持字节不变，不能用改写后的旧结果对照。
    old = ROOT / 'outputs/full_videoqa/rd_1_2__videomme2700__crossmodel_20260922__r01'
    oldspec = read_json(old / 'protocol.json')
    assert sha256(old / 'historical_baselines.json') == oldspec['selection_exports']['historical_baselines.json']
    for name, h in oldspec['source_results'].items():
        assert sha256(ROOT / name) == h
    release = verify_frozen_rd()
    frozen = read_json(ROOT / release['frozen_manifest_path'])['sha256']
    for name, h in frozen.items():
        assert sha256(ROOT / name) == h, name
    # 步骤3：有限RGB重解码核查；不把抽查说成2700题独立重解码。
    sample = list(dict.fromkeys(spec['pilot_question_ids'] + ['395-2', '844-1', '872-2']))
    for q in sample:
        images, hashes = FrameProvider(result[q]['selection']).decode()
        try:
            assert hashes == result[q]['rgb_sha256'], q
        finally:
            for image in images:
                image.close()
    # 步骤4：只有全部完成时才调用恢复入口；它应在加载模型前返回。
    paths = sorted((run / 'results').glob('*.json')) + sorted((run / 'calls' / METHOD).glob('*.json'))
    assert len(paths) == 5400
    before = {str(p.relative_to(run)): sha256(p) for p in paths}
    workers = {str(p): sha256(p) for p in (run / 'workers').glob('*.json')}
    with (run / 'resume_acceptance.log').open('w') as log:
        subprocess.run([PYTHON, str(ROOT / 'scripts/run_lmms_uniform_full.py'), '--run-id', run.name, '--resume'],
                       check=True, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    after_paths = sorted((run / 'results').glob('*.json')) + sorted((run / 'calls' / METHOD).glob('*.json'))
    assert before == {str(p.relative_to(run)): sha256(p) for p in after_paths}
    assert workers == {str(p): sha256(p) for p in (run / 'workers').glob('*.json')}
    summary = read_json(run / 'summary.json')
    assert summary['accuracy']['all']['correct'] == sum(r['correct'] for r in result.values())
    assert not summary['failures'] and not summary['recovered']
    report = dict(status='passed', questions=2700, videos=900, calls=2700,
        result_call_files_unchanged_after_resume=5400, new_resume_calls=0,
        source_history_hashes_checked=len(oldspec['source_results']), rd_frozen_files_checked=len(frozen),
        rgb_redecoded_questions=sample, protocol_sha256=fp, script_sha256=sha256(__file__),
        frame_counts=dict(Counter(len(r['selection']['selected_frames']) for r in result.values())),
        visual_tokens=dict(Counter(r['result']['observed']['visual_tokens'] for r in result.values())),
        worker_records_unchanged=True, utc=utc())
    durable(run / 'completion_audit.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id and args.run_id not in ('.', '..')
    audit(ROOT / 'outputs/baselines' / args.run_id)
