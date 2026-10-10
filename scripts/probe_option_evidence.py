"""锚点条件的非生成式选项支持：固定原银行/Qwen32配额，辅助评分不作正式QA。"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import shutil
import sys
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from videoqa_full.state import rows, durable, exclusive, utc
from videoqa_runtime.common import read_json, sha256
from videoqa_lmms_bridge.common import framework_identity, WFS
from videoqa_methods.option_evidence_support import OptionEvidenceScorer, evidence_support
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.density_selection import frame_key
from probe_task_conditioned_state import verify as verify_bank, verify_completed

BANK = ROOT / 'outputs/diagnostics/question_state__pilot8__20261009_r1'
FILES = ('scripts/probe_option_evidence.py', 'src/videoqa_methods/option_evidence_support.py',
    'src/videoqa_full/qwen_fa2_backend.py', 'src/videoqa_full/qwen_backend.py',
    'src/videoqa_methods/weighted_seed_support.py', 'src/videoqa_methods/density_selection.py')


def prepare(root):
    """接口：复用冻结原RGB银行，公共问题/选项与任务身份先冻结，不读取gold评分。"""
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'protocol.json').exists():
        return verify(root)
    old = verify_bank(BANK); verify_completed(BANK)
    public_rows = {r['question_id']: dict(question=r['question'], options=list(r['options'])) for r in rows()}
    sources = dict(old['sources']); sources[str(BANK / 'protocol.json')] = sha256(BANK / 'protocol.json')
    profiles = {}; jobs = []; baselines = []
    # 步骤1：只处理Qwen32中存在可替换名额的单元，单锚点配额等价保留，不浪费辅助前向。
    for q, bank in old['banks'].items():
        item = old['profiles']['qwen32:' + q]; pool = read_json(item['pool']); parent = read_json(item['parent_trace'])
        ordinal = {index: i for i, index in enumerate(bank['global_indices'])}
        profile = dict(item, public_input=public_rows[q], bank=bank)
        active = []
        for region, plan in zip(parent['regions'], pool['plan']['regions']):
            rid = region['region_id']; quota = parent['region_budgets'][str(rid)]
            if quota <= 1:
                continue
            anchor = plan['anchor_index']; anchor_global = item['global_mapping'][anchor]
            active.append(rid)
            baseline = dict(question_id=q, region_id=rid, anchor=anchor,
                source_indices=[anchor, anchor], bank_ordinals=[ordinal[anchor_global]] * 2,
                source_keys=[[pool['candidates'][anchor]['source_frame_index'], pool['candidates'][anchor]['source_pts']]] * 2,
                physical_unique_sources=1, scope='Repeated-anchor baseline; no added evidence')
            baselines.append(baseline)
            for index in region['members']:
                global_index = item['global_mapping'][index]
                if global_index not in ordinal or index == anchor:
                    continue
                ordered = sorted([anchor, index], key=lambda i: frame_key(pool['candidates'][i]))
                jobs.append(dict(question_id=q, region_id=rid, anchor=anchor, candidate=index,
                    source_indices=ordered, bank_ordinals=[ordinal[item['global_mapping'][i]] for i in ordered],
                    source_keys=[[pool['candidates'][i]['source_frame_index'], pool['candidates'][i]['source_pts']] for i in ordered],
                    physical_unique_sources=2, scope='Anchor and candidate, chronological; no semantic relation guarantee'))
        profiles[q] = dict(profile, active_regions=active)
    # 步骤2：原题/原选项完整入账，正确答案不进入协议或模型；旧分数/特征不重算。
    for name in FILES:
        sources[str(ROOT / name)] = sha256(ROOT / name)
        destination = root / 'code_snapshot' / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    sources[str(WFS / 'utils.py')] = sha256(WFS / 'utils.py')
    spec = dict(parent='RD-1.2-exp07', framework=framework_identity(), profiles=profiles,
        baselines=baselines, jobs=jobs, sources=sources, input_gold=False, input_options=True,
        evidence='Valid-letter mass times normalized five-outcome JS against repeated-anchor baseline',
        local_rule='Original L * E * sqrt(1 - original visual D); original anchors and quotas fixed',
        no_signal='If all bank E exactly zero in a region, replay original local rule',
        pair_forward_limit=len(jobs), anchor_baseline_limit=len(baselines), synthetic_forward_limit=4,
        final_QA_limit=0, formal_profile='Qwen32 only; helper is the same model as final QA',
        wider_gate='Cross-video visible evidence gains before development; explicitly report regressions, not require all pilot questions to be lossless',
        development_gate='One fixed Qwen32 candidate, at most200 new final answers; full only with >=142/200 and mechanism support',
        scope='Frozen bank diagnostic; not full-pool selection or a method accuracy')
    durable(root / 'protocol.json', spec)
    return spec


def verify(root):
    """接口：字节身份与公开问题/选项核验；发生变化暂停，不自动重建或重评分。"""
    spec = read_json(root / 'protocol.json')
    for path, expected in spec['sources'].items():
        if sha256(path) != expected:
            raise ValueError('Source changed: ' + path)
    public = {r['question_id']: dict(question=r['question'], options=list(r['options'])) for r in rows()}
    for q, item in spec['profiles'].items():
        if item['public_input'] != public[q]:
            raise ValueError('Original public question/options changed')
    return spec


def call(root, relative, frames, public_input, scorer, category, source_identity):
    """接口：辅助单步分布前向账本，未知开始禁止重复；原概率先于任何E后处理落盘。"""
    path = root / 'calls' / relative
    hashes = [hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest() for image in frames]
    if path.exists():
        value = read_json(path)
        if value['status'] != 'returned' or value['public_input'] != public_input or value['source_identity'] != source_identity:
            raise ValueError('Uncertain or mismatched support forward; repetition forbidden')
        if value['metadata']['rgb_sha256'] != hashes or value['category'] != category:
            raise ValueError('Cached pair source differs')
        return value
    def started(metadata):
        """接口：实际前向前持久化题/选项/两个源身份，辅助评分与正式生成额度分开。"""
        durable(path, dict(status='started', public_input=public_input, category=category,
            source_identity=source_identity, metadata=metadata, pid=os.getpid(), utc=utc()))
    probabilities, meta = scorer.score(frames, public_input['question'], public_input['options'], before_forward=started)
    value = dict(status='returned', public_input=public_input, category=category,
        source_identity=source_identity, probabilities=probabilities.tolist(), metadata=meta, utc=utc())
    durable(path, value)
    return value


def extract(root, spec):
    """接口：单GPU长期辅助评分进程，配额1跳过；不生成分类或最终答题文本。"""
    from PIL import Image
    scorer = OptionEvidenceScorer(); scorer.torch.set_num_threads(2)
    durable(root / 'environment.json', scorer.load_report)
    public = dict(question='Which color is visible?', options=['A. Red.', 'B. Blue.', 'C. Green.', 'D. Gray.'])
    synthetic = []
    # 步骤1：两个相同蓝帧重复控制、蓝红及竖屏接口；仅核验分布，不按颜色答案放行。
    for i, (size, colors) in enumerate([((1280, 720), ['blue', 'blue']), ((1280, 720), ['red', 'red']),
        ((1280, 720), ['blue', 'blue']), ((720, 1280), ['blue', 'red'])]):
        images = [Image.new('RGB', size, color) for color in colors]
        value = call(root, f'synthetic/{i}.json', images, public, scorer, 'synthetic', dict(synthetic_id=i))
        synthetic.append(value['probabilities'])
        for image in images:
            image.close()
    if synthetic[0] != synthetic[2] or synthetic[0] == synthetic[1]:
        raise ValueError('Synthetic repeatability or image injection failed')
    # 步骤2：原银行无损像素及真实时序，基线和候选条件的文本、网格必须相同。
    baselines = {}; evidence = {}; completed = 0
    for category, jobs in [('anchor_baseline', spec['baselines']), ('candidate_pair', spec['jobs'])]:
        for job in jobs:
            q = job['question_id']; rid = job['region_id']; public = spec['profiles'][q]['public_input']
            images = [Image.open(BANK / 'bank' / q / f'{ordinal:03d}.png').convert('RGB') for ordinal in job['bank_ordinals']]
            identifier = f'{rid}_anchor' if category == 'anchor_baseline' else f'{rid}_{job["candidate"]}'
            value = call(root, f'{category}/{q}/{identifier}.json', images, public, scorer, category, job)
            for image in images:
                image.close()
            expected = [spec['profiles'][q]['bank']['rgb_sha256'][ordinal] for ordinal in job['bank_ordinals']]
            if value['metadata']['rgb_sha256'] != expected:
                raise ValueError('Pair pixels differ from frozen source bank')
            if category == 'anchor_baseline':
                baselines[q, rid] = value
            else:
                baseline = baselines[q, rid]
                if value['metadata']['token_sha256'] != baseline['metadata']['token_sha256'] or value['metadata']['video_grid_thw'] != baseline['metadata']['video_grid_thw']:
                    raise ValueError('Candidate and baseline input positions differ')
                evidence[q, job['candidate']] = dict(evidence_support(value['probabilities'], baseline['probabilities']),
                    region_id=rid, anchor=job['anchor'], pair_probabilities=value['probabilities'],
                    baseline_probabilities=baseline['probabilities'])
            completed += 1
            durable(root / 'status.json', dict(status='scoring', completed=completed,
                total=len(spec['jobs']) + len(spec['baselines']), active_question=q,
                pid=os.getpid(), sid=os.getsid(0), utc=utc()))
    for q in spec['profiles']:
        durable(root / 'evidence' / (q + '.json'), {str(i): v for (question, i), v in evidence.items() if question == q})


def select_bank(root, spec):
    """接口：银行内只乘新E，原L/D/锚点/配额不变；先重放旧规则，结果非正式输入。"""
    reports = []; results = {}
    for q, item in spec['profiles'].items():
        pool = read_json(item['pool']); parent = read_json(item['parent_trace'])
        old = np.load(item['features'], allow_pickle=False)[item['global_mapping']]
        scores = read_json(root / 'evidence' / (q + '.json')); bank_ids = set(item['bank']['global_indices'])
        outcomes = {}
        # 步骤1：同银行旧控制包含所有父胜者，离散选择必须完全复现，避免采样域混杂。
        for condition in ['control', 'evidence']:
            selected = []; traces = []
            for region, plan in zip(parent['regions'], pool['plan']['regions']):
                members = [i for i in region['members'] if item['global_mapping'][i] in bank_ids]
                support, _, _ = weighted_support(old, region['members'], plan['seed_indices'], pool['initial_scores'])
                support = dict(zip(region['members'], support)); quota = parent['region_budgets'][str(region['region_id'])]
                chosen = [plan['anchor_index']]; steps = []
                fallback = not any(scores.get(str(i), {}).get('value', 0) > 0 for i in members if i != chosen[0])
                # 步骤2：新旧图共用视觉D，E不是新奇距离；无信号时原式等价回退。
                while len(chosen) < quota:
                    proposals = []
                    for i in members:
                        if i in chosen:
                            continue
                        d = float(np.clip(old[chosen] @ old[i], 0, 1).max())
                        e = 1. if condition == 'control' or fallback else scores[str(i)]['value']
                        proposals.append(dict(candidate_index=i, L=float(support[i]), D=d, E=e,
                            score=float(support[i]) * e * math.sqrt(1 - d)))
                    winner = min(proposals, key=lambda value: (-value['score'], *frame_key(pool['candidates'][value['candidate_index']])))
                    steps.append(dict(winner=winner, selected_before=list(chosen))); chosen.append(winner['candidate_index'])
                selected += chosen; traces.append(dict(region_id=region['region_id'], quota=quota,
                    prefix=chosen, no_signal_fallback=fallback, steps=steps))
            outcomes[condition] = sorted(selected, key=lambda i: frame_key(pool['candidates'][i]))
            if condition == 'evidence':
                results[q] = dict(question_id=q, selected_indices=outcomes[condition],
                    selection=dict(video=pool['video'], selected_frames=[pool['candidates'][i] for i in outcomes[condition]]),
                    trace=traces, scope='Frozen bank only, not full-pool input or formal accuracy')
        expected = sorted(item['parent_selected_indices'], key=lambda i: frame_key(pool['candidates'][i]))
        if outcomes['control'] != expected:
            raise ValueError('Same-bank original control does not replay parent')
        durable(root / 'bank_selection' / (q + '.json'), results[q])
        reports.append(dict(question_id=q, old_control_exact=True,
            changed_frames=len(set(expected) ^ set(outcomes['evidence'])) // 2,
            selected_indices=outcomes['evidence']))
    calls = [read_json(p) for p in (root / 'calls').rglob('*.json')]
    if any(value['status'] != 'returned' for value in calls) or len(calls) != len(spec['jobs']) + len(spec['baselines']) + 4:
        raise ValueError('Auxiliary call accounting differs')
    summary = dict(reports=reports, auxiliary_prefill_calls=len(calls), final_QA_calls=0,
        candidate_pair_calls=len(spec['jobs']), anchor_baselines=len(spec['baselines']), synthetic_calls=4,
        model_seconds=sum(v['metadata']['forward_seconds'] for v in calls),
        artifacts={str(p): sha256(p) for folder in ['calls', 'evidence', 'bank_selection'] for p in (root / folder).rglob('*.json')},
        semantic_review='pending', development_or_full_started=False)
    durable(root / 'summary.json', summary); durable(root / 'status.json', dict(status='completed', utc=utc()))
    return summary


def main():
    """接口：独立后台准备/辅助评分/银行重放；只读恢复不加载模型或重发不确定调用。"""
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True); args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum():
        parser.error('Invalid run ID')
    root = ROOT / 'outputs/diagnostics' / args.run_id
    try:
        with exclusive(root / 'coordinator.lock'):
            spec = prepare(root); verify(root)
            if (root / 'summary.json').exists():
                saved = read_json(root / 'summary.json')
                if any(sha256(path) != expected for path, expected in saved['artifacts'].items()):
                    raise ValueError('Completed auxiliary output changed')
                durable(root / 'restore_audit.json', dict(new_model_forwards=0, new_final_QA_calls=0, utc=utc()))
            else:
                durable(root / 'launch.json', dict(pid=os.getpid(), sid=os.getsid(0), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), utc=utc()))
                extract(root, spec); select_bank(root, spec)
    except BaseException as error:
        durable(root / 'failure.json', dict(status='paused', error=repr(error), utc=utc())); raise


if __name__ == '__main__':
    main()
