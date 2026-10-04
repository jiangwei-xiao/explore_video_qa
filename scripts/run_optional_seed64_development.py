"""exp23冻结M64/K32输入的200题问答协调器，准备与报告不生成额外答案。"""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive, utc
from videoqa_audit.query_input_probe import BASE
from videoqa_budget32 import development

METHOD = 'RD-1.2-exp23'
NAME = 'rd12_exp23__qwen32__dev200__20261004_r1'
RUN = ROOT / 'outputs/methods' / NAME
QA = ROOT / 'outputs/diagnostics' / NAME
PREP = ROOT / 'outputs/methods/rd12_exp23__m64k32__prepare200__20261004_r1'
DATA = ROOT / 'data/manifests/video_mme_development_200_20260920.json'
PYTHON = '/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'


def immutable(path, value):
    """接口：检查点只写一次，恢复时拒绝来源或内容变化。"""
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Frozen content changed: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        durable(path, value)


def prepare():
    """接口：核验完整准备资产并冻结200份输入，不解码重选或导入旧答案。"""
    # 步骤1：准备记录、实际文件和控制验收同时成立才允许进入问答。
    ids = [r['question_id'] for r in read_json(DATA)['rows']]
    if len(ids) != len(set(ids)) or len(ids) != 200:
        raise ValueError('Expected frozen 200 unique questions')
    summary = read_json(PREP / 'summary.json')
    if summary['status'] != 'passed' or summary['questions'] != 200 or summary['qa_calls'] != 0:
        raise ValueError('Feature preparation incomplete')
    sources = dict(read_json(PREP / 'protocol.json')['sources'])
    for name in ['protocol', 'summary', 'overlap_audit', 'reencoded_control', 'reencoded_input_equivalence']:
        p = PREP / (name + '.json')
        sources[str(p)] = sha256(p)
    for name in ['overlap_audit', 'reencoded_input_equivalence']:
        if read_json(PREP / (name + '.json'))['status'] != 'passed':
            raise ValueError('Control acceptance failed')
    if read_json(PREP / 'overlap_audit.json')['source_summary_sha256'] != sha256(PREP / 'summary.json'):
        raise ValueError('Overlap audit belongs to a different preparation')
    if read_json(PREP / 'reencoded_input_equivalence.json')['control_sha256'] != sha256(PREP / 'reencoded_control.json'):
        raise ValueError('Input control provenance mismatch')
    for path, digest in summary['files'].items():
        sources[path] = digest
        checkpoint = read_json(ROOT / path)
        sources.update(checkpoint['files'])
    for p in [Path(__file__), DATA, ROOT / 'scripts/report_external_support.py']:
        sources[str(p)] = sha256(p)
    for path, digest in sources.items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f'Preparation asset changed: {path}')
    if METHOD not in {r['id'] for r in read_json(ROOT / 'configs/method_versions.json')['entries']}:
        raise ValueError('Unregistered method')
    spec = dict(method=METHOD, question_ids=ids, sources=sources, max_qa_calls=200,
                allowed_gpus=list(map(str, range(8))), configuration='M64/K32 optional region quota; frozen exp23',
                stop='No generation retries, no parameter search, no automatic full evaluation')
    immutable(RUN / 'protocol.json', spec)
    # 步骤2：输入只引用已验收的源帧和RGB；正式process再次精确解码核验。
    directory = RUN / 'qa_inputs'
    for q in ids:
        path = PREP / 'selections' / (q + '.json')
        item = read_json(path)
        if item['question_id'] != q or item['method'] != METHOD:
            raise ValueError('Selection identity mismatch')
        frames = item['selection']['selected_frames']
        pts = [f['source_pts'] for f in frames]
        if len(frames) != 32 or len(set(pts)) != 32 or pts != sorted(pts) or len(item['rgb_sha256']) != 32:
            raise ValueError('Frozen input budget, order or identity invalid')
        immutable(directory / (q + '.json'), dict(question_id=q, method=METHOD,
            selection=item['selection'], selected_indices=item['selected_indices'], rgb_sha256=item['rgb_sha256'],
            sources={str(path): sha256(path), str(RUN / 'protocol.json'): sha256(RUN / 'protocol.json')}))
    immutable(directory / 'selection_manifest.json', dict(status='completed', question_count=200,
        frame_budget=32, question_ids=ids, dataset_manifest=str(DATA), manifest_sha256=sha256(DATA),
        parent_run=str(BASE), parent_protocol_sha256=sha256(BASE / 'protocol.json'), method=METHOD,
        configuration=spec['configuration'], diagnostic=False,
        exports={q: sha256(directory / (q + '.json')) for q in ids}))
    # 步骤3：代码快照不可改写；模型和官方评测依赖由既有入口单独冻结。
    for path, digest in sources.items():
        p = ROOT / path
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            continue
        if rel.parts[0] not in {'src', 'scripts', 'configs'}:
            continue
        dest = RUN / 'code_snapshot' / rel
        if dest.exists():
            if sha256(dest) != digest:
                raise ValueError('Code snapshot mismatch')
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    development.prepare(QA, directory, spec['allowed_gpus'])


def report_only():
    """接口：在正式Qwen环境仅完成配对报告，核对600份生成文件没有变化。"""
    # 步骤1：必须已有正式完成审计，不能从未知生成状态绕过恢复约束。
    if read_json(QA / 'completion_audit.json')['status'] != 'passed':
        raise ValueError('QA not complete')
    spec = development.verify(QA)
    if spec['methods'] != [METHOD] or spec['max_calls'] != 200:
        raise ValueError('QA identity mismatch')
    files = [p for d in ['calls', 'raw_generations', 'results'] for p in (QA / d / METHOD).glob('*.json')]
    if len(files) != 600:
        raise ValueError('Generation ledger is incomplete')
    before = {str(p): sha256(p) for p in files}
    # 步骤2：旧通用汇总器仅替换展示运行身份，算法和原始输出不变。
    import report_external_support as report
    report.NAME, report.METHOD, report.RUN, report.COORD = NAME, METHOD, QA, RUN
    durable(RUN / 'status.json', dict(status='completed', qa_complete=True, report_status='running', utc=utc()))
    report.main(False)
    if before != {str(p): sha256(p) for p in files}:
        raise ValueError('Reporting changed generation records')
    durable(QA / 'report_recovery.json', dict(status='passed', unchanged_records=600, new_model_calls=0,
                                             record_hashes=before, utc=utc()))
    durable(RUN / 'status.json', dict(status='completed', qa_complete=True, report_status='passed', utc=utc()))


def main(stage):
    """接口：后台先行验收及200题独立运行；已完成只恢复报告，不重新生成。"""
    with exclusive(RUN / 'run.lock'):
        # 步骤1：完成恢复优先，不重新准备输入或加载模型。
        completed = QA / 'completion_audit.json'
        if stage == 'report':
            report_only()
            return
        if not (completed.exists() and read_json(completed)['status'] == 'passed'):
            prepare()
            if stage == 'prepare':
                return
            # 步骤2：子进程负责空闲GPU、前五题和持久调用账本，最多200次生成。
            durable(RUN / 'launcher_identity.json', dict(pid=os.getpid(), sid=os.getsid(0), utc=utc()))
            durable(RUN / 'status.json', dict(status='qa', pid=os.getpid(), sid=os.getsid(0), utc=utc()))
            subprocess.run([PYTHON, str(ROOT / 'scripts/run_frozen_development.py'), '--run-id', NAME,
                            '--stage', 'qa', '--resume'], check=True)
    # 步骤3：报告也使用排他锁，但不在父进程持锁时等待子进程锁。
    subprocess.run([PYTHON, str(Path(__file__)), '--stage', 'report'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'run', 'report'], default='run')
    try:
        main(parser.parse_args().stage)
    except BaseException as exc:
        durable(RUN / 'status.json', dict(status='paused', error=repr(exc), utc=utc()))
        raise
