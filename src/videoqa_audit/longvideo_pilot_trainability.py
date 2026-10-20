"""长视频先导CPU诊断：固定质量掩码、五折折外预测和194题敏感性复算。"""
from collections import Counter
import hashlib
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import digest, durable, exclusive, utc
from videoqa_audit.set_value_trainability import (
    SPEC, Sources, TrainingAssets, cluster_comparison, cross_validate,
    score_inputs, select_methods,
)
from videoqa_set_value.data import METHODS
from videoqa_set_value.learner import SetValueHead
from videoqa_set_value.longvideo_task import strict_option

SOURCE = ROOT / 'outputs/set_value/set_value_longvideo__llava8__train800_val400__20261019_r1'
OLD_HEAD = ROOT / 'outputs/analysis/set_value_trainability__nextqa600__20261019_r1/full_fit/state.pt'
OLD_HEAD_SHA = '8f069aa8288dbe012d1e7e9b69e2e18e46126ab904dfd6d8c02f54d5f398655f'
MANIFEST_SHA = '0b737590cffe589d993a0fbf6468f2a50efee2e8b56560be8cd32490b7202eaf'
SCOPE = 'set_value_longvideo_pilot_cv_v1'
EXCLUDED = {'lvr-train-37437', 'lvr-train-46919', 'lvr-train-18121'}
UNCERTAIN = {'lvr-train-35296', 'lvr-train-39579', 'lvr-train-30424'}
TIME = r'\d{1,2}:\d{2}:\d{2}(?:\.\d+)?'
TITLE_ONLY = re.compile(r'\s*#{1,6}\s*From\s*\[\s*' + TIME +
                        r'\s+to\s+' + TIME + r'\s*\]\s*:\s*', re.I)


def option_defects(row):
    """接口：只读原选项字符串和展示字母；任何空项/纯时间标题使整题失效。"""
    # 步骤1检查展示结构；步骤2逐项记录缺陷，不访问正确答案或历史预测。
    options, letters = row['options'], row['option_letters']
    if len(options) != 4 or len(letters) != 4 or set(letters) != set('ABCD'):
        raise ValueError('Original four-option mapping differs')
    return [dict(letter=letter, option=value,
                 reason='empty_option' if not value.strip() else 'timestamp_title_only')
            for letter, value in zip(letters, options)
            if not value.strip() or TITLE_ONLY.fullmatch(value)]


def freeze_population(rows):
    """接口：原200题公开身份→质量掩码及五折；先分折后掩码，完全不读取标签。"""
    # 步骤1固定原总体和排除理由；同内容视频不能重复或被移入其他折。
    if len(rows) != 200 or any(r['split'] != 'train' or not r['pilot'] for r in rows):
        raise ValueError('Expected original pilot 200 training rows')
    if len({r['question_id'] for r in rows}) != 200 or len({r['content_group'] for r in rows}) != 200:
        raise ValueError('Pilot source/question identity is not unique')
    defects = {r['question_id']: option_defects(r) for r in rows}
    excluded = {q for q, entries in defects.items() if entries}
    if excluded != EXCLUDED:
        raise ValueError('Frozen option quality partition differs')
    valid = [r for r in rows if r['question_id'] not in excluded]
    if not UNCERTAIN.issubset({r['question_id'] for r in valid}):
        raise ValueError('Uncertain cases must remain in training and primary evaluation')
    ordered = sorted(rows, key=lambda r: hashlib.sha256(
        ('2027/' + SCOPE + '/' + r['content_group']).encode()).hexdigest())
    membership = {r['question_id']: i % 5 for i, r in enumerate(ordered)}
    # 步骤2仅用原顺序构造有效题索引；排除题占据的原始折位置不向前填补。
    folds = []
    for fold in range(5):
        held = [i for i, r in enumerate(valid) if membership[r['question_id']] == fold]
        train = [i for i, r in enumerate(valid) if membership[r['question_id']] != fold]
        folds.append(dict(fold=fold, train_indices=train, heldout_indices=held,
                          train_question_ids=[valid[i]['question_id'] for i in train],
                          heldout_question_ids=[valid[i]['question_id'] for i in held]))
    if [len(f['heldout_indices']) for f in folds] != [40, 40, 39, 40, 38]:
        raise ValueError('Frozen fold population differs')
    quality = dict(rule='any_empty_or_complete_timestamp_heading_option_v1',
                   reads_answers=False, original_questions=200, primary_questions=197,
                   sensitivity_questions=194, excluded=[dict(question_id=q, defects=defects[q])
                                                         for q in sorted(excluded)],
                   uncertain_retained=sorted(UNCERTAIN),
                   primary_question_ids=[r['question_id'] for r in valid],
                   sensitivity_question_ids=[r['question_id'] for r in valid if r['question_id'] not in UNCERTAIN])
    return valid, quality, dict(seed=2027, scope=SCOPE, membership_before_mask=membership, folds=folds)


def freeze_json(path, value):
    """接口：首次原子发布；恢复时配置/来源变化立即暂停，禁止覆盖旧冻结文件。"""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Frozen diagnostic record differs: ' + str(path))
    else:
        durable(path, value)


def verify_answer(row, result, reference):
    """接口：按原四选项字母复算历史输出；参考字母不是选项列表中的位置。"""
    parsed = strict_option(result['answer']['raw_output'])
    if reference not in row['option_letters']:
        raise ValueError('Scoring-only reference not among original labels')
    correct = parsed == reference
    if (result['question_id'] != row['question_id'] or result['phase'] != 'pilot'
            or result['answer']['parsed_answer'] != parsed or bool(result['correct']) != correct
            or bool(result['answer']['framework_metric']['correct']) != correct):
        raise ValueError('Original four-option answer binding differs')
    return int(correct)


def load_pilot_assets(source, output, tracker):
    """接口：只加载先导200资产，输出197题训练对象；不访问独立验证或剩600题反馈。"""
    source, output = Path(source), Path(output)
    # 步骤1先发布不含答案的质量掩码/分折，再读取已有先导反馈。
    manifest = tracker.read(source / 'manifest.json', MANIFEST_SHA)
    original = [r for r in manifest['rows'] if r['split'] == 'train' and r['pilot']]
    valid, quality, folds = freeze_population(original)
    freeze_json(output / 'quality_mask.json', quality)
    freeze_json(output / 'folds.json', folds)
    protocol = tracker.read(source / 'protocol.json')
    frozen = tracker.read(source / 'source_freeze.json', protocol['source_freeze_sha256'])
    for rel, expected in frozen['code'].items():
        tracker.check(ROOT / rel, expected)
    tracker.check(OLD_HEAD, OLD_HEAD_SHA)
    if frozen['old_head_sha256'] != OLD_HEAD_SHA or protocol['manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('Parent protocol/head identity differs')
    acceptance = tracker.read(source / 'completed_restore_acceptance.json')
    protection = tracker.read(source / 'restore_protection_before.json', acceptance['before_sha256'])
    protected = protection['files']
    for name in ('STOP.json', 'pilot_report.json', 'pilot_review.json'):
        tracker.check(source / name)
    if read_json(source / 'STOP.json')['status'] != 'stopped_pilot_visual_gate':
        raise ValueError('Original stopped trial status differs')
    label_path = next(Path(p) for p in manifest['identity_sources'] if p.endswith('/scoring_labels.json'))
    labels = tracker.read(label_path, manifest['identity_sources'][str(label_path)])
    pilot_ids = {r['question_id'] for r in original}
    references = {}
    for record in labels['rows']:
        if record['question_id'] not in pilot_ids:
            continue
        match = re.fullmatch(r'\s*<answer>\s*([A-D])\s*</answer>\s*', record['reference_answer'])
        if not match:
            raise ValueError('Ambiguous original reference')
        if record['question_id'] in references:
            raise ValueError('Duplicate pilot scoring row')
        references[record['question_id']] = match.group(1)
    if set(references) != pilot_ids:
        raise ValueError('Pilot scoring population incomplete')

    def protected_read(relative):
        """接口：仅实际读取文件核验原完成时哈希，不扫描视频或未使用的验证资产。"""
        return tracker.read(source / relative, protected[relative])

    xx, yy, representatives, orders = [], [], [], []
    valid_ids = {r['question_id'] for r in valid}
    # 步骤2核对原结果、实际像素/提示、调用与特征；共享输入不重复投票。
    for row in original:
        q = row['question_id']
        plan = protected_read('inputs/pilot/' + q + '.json')
        if plan['question_id'] != q or plan['phase'] != 'pilot' or plan['partition_uses_labels']:
            raise ValueError('Input plan identity differs')
        rel = 'value_features/' + q + '.npz'
        with np.load(tracker.check(source / rel, protected[rel]), allow_pickle=False) as archive:
            arrays = [archive[m] for m in METHODS]
            if any(a.shape != (8, 770) or a.dtype != np.float32 for a in arrays):
                raise ValueError('Frozen FP32 feature shape differs')
            x = np.stack(arrays)
        if (not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x[:, :, :768], axis=-1), 1., atol=1e-5, rtol=0)
                or np.any((x[:, :, 768:] < 0) | (x[:, :, 768:] > 1))):
            raise ValueError('Frozen features outside protocol')
        y = []
        for method in METHODS:
            result = protected_read('qa/pilot/results/' + method + '/' + q + '.json')
            entry = plan['methods'][method]
            if result['method'] != method or result['input_key'] != entry['input_key']:
                raise ValueError('Method/input binding differs')
            y.append(verify_answer(row, result, references[q]))
            observed = result['answer']['observed']
            for key in ('prompt', 'pixel_sha256', 'input_token_ids', 'source_frames', 'visual_tokens'):
                if observed[key] != entry['identity'][key]:
                    raise ValueError('Actual input identity differs: ' + key)
            if observed['visual_tokens'] != 1680 or observed['max_new_tokens'] != 16:
                raise ValueError('Historical teacher protocol differs')
            order = entry['source_order']
            if len(order) != 8 or order != sorted(order) or len({tuple(s) for s in order}) != 8:
                raise ValueError('Frozen source ordering differs')
            call_path = Path(result['call_path'])
            relative = str(call_path.relative_to(source))
            if not relative.startswith('qa/pilot/calls/'):
                raise ValueError('Non-pilot call access forbidden')
            call = protected_read(relative)
            if (call['status'] != 'completed' or call['generation_state'] != dict(started=True, returned=True)
                    or call['question_id'] != q or call['answer']['raw_output'] != result['answer']['raw_output']
                    or call['answer']['observed'] != observed):
                raise ValueError('Original/shared raw call binding differs')
        reps, covered = [], []
        for group in plan['groups']:
            members = [METHODS.index(m) for m in group['methods'] if m in METHODS]
            if not members:
                continue
            leader = members[0]
            if any(y[j] != y[leader] or not np.array_equal(x[j], x[leader])
                   or plan['methods'][METHODS[j]]['identity'] != plan['methods'][METHODS[leader]]['identity']
                   for j in members):
                raise ValueError('Identical input has conflicting feature/answer identity')
            reps.append(leader)
            covered.extend(members)
        if sorted(covered) != list(range(4)):
            raise ValueError('Input group coverage differs')
        if q in valid_ids:
            xx.append(x); yy.append(y); representatives.append(reps)
            orders.append([plan['methods'][m]['source_order'] for m in METHODS])
    # 步骤3仅返回有效训练题，坏题原结果和清单均不改写。
    data = TrainingAssets(valid, np.stack(xx), np.asarray(yy, dtype=np.float32), representatives, orders)
    pairs = sum(sum(data.labels[i, j] for j in reps) *
                sum(1 - data.labels[i, j] for j in reps) for i, reps in enumerate(representatives))
    if len(data.usable()) != 40 or int(pairs) != 108 or data.labels.sum(0).astype(int).tolist() != [132, 136, 135, 136]:
        raise ValueError('Frozen effective supervision counts differ')
    if [len(data.subset(f['heldout_indices']).usable()) for f in folds['folds']] != [9, 8, 9, 7, 7]:
        raise ValueError('Frozen fold preference counts differ')
    return data, quality, folds


def evaluate_subset(data, predictions, indices):
    """接口：固定折外预测→指定197/194题评价；敏感性复算不训练、不重新选模型。"""
    # 步骤1同步筛选行、标签和原预测，主固定参照仍是该题所属训练折选出的策略。
    ids = list(indices); rows = [data.rows[i] for i in ids]
    values = {m: [int(data.labels[i, j]) for i in ids] for j, m in enumerate(METHODS)}
    for name, field in (('learned_oof', 'chosen_index'), ('training_fold_best', 'baseline_index'),
                        ('mean_itm', 'mean_itm_index'), ('old_nextqa_head', 'old_head_index')):
        values[name] = [int(data.labels[i, predictions[i][field]]) for i in ids]
    preferences = []
    for i in ids:
        pairs = []
        for a in data.representatives[i]:
            for b in data.representatives[i]:
                if data.labels[i, a] == 1 and data.labels[i, b] == 0:
                    delta = predictions[i]['scores'][a] - predictions[i]['scores'][b]
                    pairs.append(1. if delta > 1e-8 else .5 if abs(delta) <= 1e-8 else 0.)
        if pairs:
            preferences.append(float(np.mean(pairs)))
    # 步骤2复用视频簇配对统计，当前每视频一题；区间不包含重新训练的不确定性。
    comparisons = {name: cluster_comparison(rows, values['learned_oof'], v)
                   for name, v in values.items() if name != 'learned_oof'}
    correct = {name: sum(v) for name, v in values.items()}
    gains = set(comparisons['training_fold_best']['new_correct'])
    gain_videos = len({r['content_group'] for r in rows if r['question_id'] in gains})
    criteria = dict(preference_at_least_sixty=bool(preferences) and np.mean(preferences) >= .6,
                    net_gain_over_training_fold_best=comparisons['training_fold_best']['net_correct'] > 0,
                    no_lower_than_best_fixed=correct['learned_oof'] >= max(correct[m] for m in METHODS),
                    improvements_in_at_least_three_videos=gain_videos >= 3)
    by_duration = {}
    for low, high, name in ((180., 600., '180–600s'), (600., 1800.000001, '600–1800s')):
        selected = [j for j, i in enumerate(ids) if low <= data.rows[i]['duration_seconds'] < high]
        by_duration[name] = dict(questions=len(selected),
                                correct={k: sum(v[j] for j in selected) for k, v in values.items()})
    return dict(questions=len(ids), preference_questions=len(preferences),
                preference_accuracy=float(np.mean(preferences)) if preferences else None,
                correct=correct, accuracy={k: v / len(ids) for k, v in correct.items()},
                comparisons=comparisons, criteria={k: bool(v) for k, v in criteria.items()},
                improvement_videos=gain_videos, oracle_correct=sum(any(data.labels[i]) for i in ids),
                unique_preference_pairs=int(sum(sum(data.labels[i, j] for j in data.representatives[i]) *
                    sum(1 - data.labels[i, j] for j in data.representatives[i]) for i in ids)),
                choices=dict(Counter(METHODS[predictions[i]['chosen_index']] for i in ids)),
                by_duration=by_duration, question_ids=[r['question_id'] for r in rows])


def recommend(main, sensitivity, all_fitted):
    """接口：严格执行事前资源门；194题不另加偏好60%门，不自动接续800/400。"""
    gates = dict(all_folds_fitted=bool(all_fitted), **main['criteria'])
    for key in ('net_gain_over_training_fold_best', 'no_lower_than_best_fixed',
                'improvements_in_at_least_three_videos'):
        gates['sensitivity_' + key] = sensitivity['criteria'][key]
    passed = all(gates.values())
    decision = ('propose_800_400_protocol_review' if passed else 'training_not_fully_fitted'
                if not all_fitted else 'close_current_frozen_blip_set_head')
    return dict(criteria=gates, passed=passed, decision=decision,
                automatic_teacher_extension=False, automatic_target_evaluation=False)


def render_report(output, result):
    """接口：输出五折训练曲线与可读对照；原200题和新197/194分母明确区分。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    # 步骤1曲线只展示训练指标，留出成绩不用于选择曲线片段或检查点。
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for i, summary in enumerate(result['fold_summaries']):
        history = summary['history']; steps = [h['step'] for h in history]
        for ax, key in zip(axes, ('mean_preference_loss', 'preference_accuracy', 'usable_top1_accuracy')):
            ax.plot(steps, [h[key] for h in history], label='Fold ' + str(i + 1))
    for ax, title in zip(axes, ('Training loss', 'Training preference accuracy', 'Training top-1 accuracy')):
        ax.set_title(title); ax.set_xlabel('Optimizer updates'); ax.grid(alpha=.2)
    axes[0].legend()
    for ax in axes[1:]:
        ax.axhline(.95, color='gray', ls='--'); ax.set_ylim(0, 1.03)
    fig.savefig(output / 'training_curves.png', dpi=160); plt.close(fig)
    # 步骤2所有比较及其完整题单留在JSON，Markdown只呈现主结论和明确限制。
    lines = ['# 长视频先导CPU五折诊断', '', '决策：`' + result['recommendation']['decision'] + '`。', '',
             '仅已有先导反馈的回顾性诊断；新增大模型调用0。原试点画面门停止记录保留。', '',
             '| 口径 | 题数 | 偏好视频 | 折外偏好正确率 | 新头正确数 | 训练折固定参照 | 最好固定方法 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for label, key in (('主分析', 'primary'), ('排除关系争议题', 'sensitivity')):
        value = result[key]; counts = value['correct']
        lines.append(f"| {label} | {value['questions']} | {value['preference_questions']} | "
                     f"{100 * value['preference_accuracy']:.2f}% | {counts['learned_oof']} | "
                     f"{counts['training_fold_best']} | {max(counts[m] for m in METHODS)} |")
    lines += ['', '| 参照 | 主197题正确数 | 新头净增 | 新增／退化 | 配对95%区间（百分点） |',
              '|---|---:|---:|---:|---|']
    main = result['primary']
    for name, value in main['comparisons'].items():
        lo, hi = np.asarray(value['cluster_95_interval']) * 100
        lines.append(f"| {name} | {main['correct'][name]} | {value['net_correct']:+d} | "
                     f"{len(value['new_correct'])}／{len(value['degraded'])} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ['', '![训练曲线](training_curves.png)', '',
              '五折更新数：' + '／'.join(str(s['updates']) for s in result['fold_summaries']) + '。', '',
              '197题排除37437、46919、18121；194题再排除35296、39579、30424，仅重算评价。',
              '三道关系争议题仍曾参与部分折训练；敏感性口径没有消除其训练影响。',
              '视频簇bootstrap 10000次，种子2027；固定折外预测区间不包含重新训练的不确定性。',
              '原200题先导成绩保留；独立400题、剩余600训练题及目标基准未访问或新增问答。',
              '完整来源、全部配对题号及投入门见report.json、oof.json和sources.json。', '']
    (output / 'report.md').write_text('\n'.join(lines))


def run(output, source=SOURCE):
    """接口：CPU单进程完整执行/恢复；完成恢复只验哈希，不重新更新参数或重写产物。"""
    output, source = Path(output), Path(source)
    output.mkdir(parents=True, exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU diagnostic requires CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    with exclusive(output / 'analysis.lock'):
        # 步骤1完成恢复只读检查，避免任何重复训练或原试点协调器调用。
        if (output / 'completion.json').exists():
            completion = read_json(output / 'completion.json')
            for path, expected in completion['sources'].items():
                if sha256(path) != expected:
                    raise ValueError('Completed source fingerprint changed: ' + path)
            for rel, expected in completion['artifacts'].items():
                if sha256(output / rel) != expected:
                    raise ValueError('Completed diagnostic artifact changed: ' + rel)
            return read_json(output / 'report.json')
        if not (output / 'launcher.json').exists():
            durable(output / 'launcher.json', dict(pid=os.getpid(), sid=os.getsid(0),
                    cpu_threads=2, cuda_visible_devices='', source=str(source), utc=utc()))
        began = time.monotonic(); tracker = Sources()
        tracker.check(Path(__file__))
        tracker.check(ROOT / 'scripts/analyze_longvideo_pilot_trainability.py')
        data, quality, folds = load_pilot_assets(source, output, tracker)
        protocol = dict(task='longvideo_pilot_cpu_trainability_v1', source=str(source), spec=SPEC,
                        scope=SCOPE, max_trajectories=5, max_optimizer_updates=10000,
                        parameter_count=sum(p.numel() for p in SetValueHead().parameters()),
                        quality_mask_sha256=sha256(output / 'quality_mask.json'),
                        folds_sha256=sha256(output / 'folds.json'), old_head_sha256=OLD_HEAD_SHA,
                        cpu_threads=2, new_backbone_forwards=0, new_generations=0,
                        source_identity=digest(tracker.files),
                        bootstrap=dict(replicates=10000, seed=2027, cluster='video', fixed_predictions=True))
        if protocol['parameter_count'] != 107073:
            raise ValueError('Frozen score head changed')
        freeze_json(output / 'protocol.json', protocol)
        freeze_json(output / 'sources.json', tracker.files)
        durable(output / 'heartbeat.json', dict(stage='five_fold_cpu_training', pid=os.getpid(),
                                               sid=os.getsid(0), utc=utc()))
        # 步骤2复用冻结训练器；每折只收到训练子对象，不额外训练全197题头。
        oof = cross_validate(data, folds, output / 'cv')
        summaries = oof['fold_summaries']
        if len(summaries) != 5 or sum(s['updates'] for s in summaries) > 10000:
            raise ValueError('Formal CPU training budget exceeded')
        if len({s['initialization_sha256'] for s in summaries}) != 1:
            raise ValueError('Fold initialization differs')
        # 步骤3旧头只推理；保存折外分数后统一评价，敏感性不重新训练。
        old = SetValueHead().float()
        old.load_state_dict(torch.load(OLD_HEAD, map_location='cpu', weights_only=False)['model'])
        old_scores = score_inputs(old, data.features)
        old_choices = select_methods(old_scores, data.source_orders)
        predictions = oof['predictions']
        for i, prediction in enumerate(predictions):
            prediction.update(old_head_index=int(old_choices[i]), old_head_scores=old_scores[i].tolist())
        durable(output / 'oof.json', dict(predictions=predictions,
                                        fold_summaries=summaries, new_training_trajectories=5))
        main = evaluate_subset(data, predictions, range(len(data.rows)))
        sensitivity = evaluate_subset(data, predictions, [i for i, r in enumerate(data.rows)
                                                         if r['question_id'] not in UNCERTAIN])
        all_fitted = all(s['status'] == 'fit_passed' for s in summaries)
        result = dict(status='completed', primary=main, sensitivity=sensitivity,
                      recommendation=recommend(main, sensitivity, all_fitted), fold_summaries=summaries,
                      total_optimizer_updates=sum(s['updates'] for s in summaries),
                      trajectories=5, new_backbone_forwards=0, new_generations=0,
                      independent_validation_re_evaluated=False, source_manifest_sha256=MANIFEST_SHA,
                      wall_seconds=time.monotonic() - began, utc=utc())
        durable(output / 'report.json', result); render_report(output, result)
        # 步骤4只复查实际访问的来源；不解码视频、不重扫189GB或原封存验证数据。
        tracker.verify()
        ignored = {'analysis.lock', 'heartbeat.json', 'coordinator.log', 'launcher.json', 'completion.json'}
        artifacts = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*')
                     if p.is_file() and p.name not in ignored}
        durable(output / 'completion.json', dict(status='completed', sources=tracker.files,
                                                 artifacts=artifacts, new_generations=0,
                                                 total_optimizer_updates=result['total_optimizer_updates'], utc=utc()))
        durable(output / 'heartbeat.json', dict(stage='completed', pid=os.getpid(), utc=utc()))
        return result
