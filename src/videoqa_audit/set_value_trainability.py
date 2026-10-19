"""集合价值学习CPU诊断：监督审计、40步重放、固定拟合与视频级折外评价。"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, digest, exclusive, utc
from videoqa_set_value.data import METHODS
from videoqa_set_value.data_acquisition import stable_key
from videoqa_set_value.learner import SetValueHead, preference_loss
from videoqa_set_value.task import strict_option

SOURCE = ROOT / 'outputs/set_value/set_value_learning__nextqa800__llava8__20261019_r2'
REVIEW = ROOT / 'outputs/analysis/dual_dataset_selection_review_20261015_r2'
SPEC = dict(seed=2027, batch_questions=32, learning_rate=.001, weight_decay=.0001,
            max_updates=2000, check_every=20, required_accuracy=.95, consecutive_checks=3)
TIME_BINS = ((180., '0–180s'), (600., '180–600s'), (1800., '600–1800s'), (math.inf, '>1800s'))


class Sources:
    """接口：每个读取来源先计算字节摘要；恢复和收尾只读核验，不加载骨干模型。"""
    def __init__(self):
        self.files = {}

    def check(self, path, expected=None):
        """步骤1记录或核对哈希；步骤2拒绝来源变化或错误的历史绑定。"""
        path = Path(path); value = sha256(path)
        if expected is not None and value != expected: raise ValueError('Frozen source differs: '+str(path))
        if str(path) in self.files and self.files[str(path)] != value: raise ValueError('Source changed during analysis')
        self.files[str(path)] = value
        return path

    def read(self, path, expected=None):
        """接口：仅返回已绑定内容身份的JSON，不执行原实验入口。"""
        return read_json(self.check(path, expected))

    def verify(self):
        """接口：收尾复查实际读取的训练及描述统计来源。"""
        for path, expected in self.files.items():
            if sha256(path) != expected: raise ValueError('Source mutated: '+path)


@dataclass
class TrainingAssets:
    """接口：仅容纳原训练600题；子集直接隔离特征、标签和完整输入去重索引。"""
    rows: list
    features: np.ndarray
    labels: np.ndarray
    representatives: list
    source_orders: list

    def __post_init__(self):
        if not self.rows or any(r['split'] != 'train' for r in self.rows):
            raise ValueError('Only original training rows are allowed')
        n = len(self.rows)
        if self.features.shape != (n, 4, 8, 770) or self.labels.shape != (n, 4):
            raise ValueError('Training tensor identity differs')
        if self.features.dtype != np.float32 or self.labels.dtype != np.float32:
            raise ValueError('Only frozen FP32 tensors are allowed')
        if not np.isfinite(self.features).all() or not np.isin(self.labels, (0, 1)).all():
            raise ValueError('Nonfinite features or invalid correctness')
        if len(self.representatives) != n or len(self.source_orders) != n:
            raise ValueError('Missing input grouping')

    def subset(self, indices):
        """接口：生成真正独立的训练折对象，训练器不会持有留出题标签。"""
        ids = list(map(int, indices))
        return TrainingAssets([self.rows[i] for i in ids], self.features[ids].copy(), self.labels[ids].copy(),
                              [self.representatives[i] for i in ids], [self.source_orders[i] for i in ids])

    def usable(self):
        """接口：仅唯一输入的正负并存题产生偏好，不对全对/全错伪造标签。"""
        return [i for i, reps in enumerate(self.representatives)
                if len({int(self.labels[i, j]) for j in reps}) == 2]


def load_training_assets(source, tracker):
    """接口：核验冻结训练特征和原始答案；拒绝读取qa/val及验证特征。"""
    source = Path(source)
    # 步骤1：只投影原清单中的训练行，标签核验使用训练调用的原始返回。
    manifest = tracker.read(source/'manifest.json')
    rows = [r for r in manifest['rows'] if r['split'] == 'train']
    if len(rows) != 600 or len({r['video_id'] for r in rows}) != 300:
        raise ValueError('Expected frozen 600 questions / 300 videos')
    model = tracker.read(source/'head/training.json')
    audit = tracker.read(source/'qa/train/completion_audit.json')
    inputs = tracker.read(source/'inputs_complete.json')
    arrays, labels, reps, orders = [], [], [], []
    for row in rows:
        q = row['question_id']
        plan = tracker.read(source/'inputs'/(q+'.json'), inputs['exports'][q])
        feature_path = tracker.check(source/'value_features'/(q+'.npz'), model['identity']['train_features'][q])
        with np.load(feature_path, allow_pickle=False) as z:
            features = np.stack([z[m] for m in METHODS]).astype(np.float32)
        if not np.allclose(np.linalg.norm(features[:, :, :768], axis=-1), 1., atol=1e-5, rtol=0):
            raise ValueError('Fused feature normalization differs')
        if np.any((features[:, :, 768:] < 0) | (features[:, :, 768:] > 1)):
            raise ValueError('ITM/time features outside protocol')
        y = []
        for method in METHODS:
            rel = 'results/'+method+'/'+q+'.json'
            result = tracker.read(source/'qa/train'/rel, audit['artifacts'][rel])
            answer = result['answer']; prediction = strict_option(answer['raw_output'])
            correct = prediction == 'ABCDE'[row['answer_index']]
            if correct != result['correct'] or prediction != answer['parsed_answer']:
                raise ValueError('Training answer/label mismatch')
            if result['input_key'] != plan['methods'][method]['input_key']:
                raise ValueError('Training feature/input binding differs')
            y.append(int(correct))
        group_indices = [g['representative_method_index'] for g in plan['groups']]
        for group in plan['groups']:
            ids = group['method_indices']; representative = group['representative_method_index']
            if len({y[i] for i in ids}) != 1: raise ValueError('Identical training input has conflicting labels')
            for i in ids:
                if not np.array_equal(features[i], features[representative]):
                    raise ValueError('Identical training input has different features')
        arrays.append(features); labels.append(y); reps.append(group_indices)
        orders.append([plan['methods'][m]['source_order'] for m in METHODS])
    # 步骤2：模型类和损失仍使用原冻结实现，作为重放的控制来源。
    protocol = tracker.read(source/'protocol.json')
    tracker.check(ROOT/'src/videoqa_set_value/learner.py', protocol['code']['src/videoqa_set_value/learner.py'])
    tracker.check(source/'head/checkpoint.pt', model['checkpoint_sha256'])
    return TrainingAssets(rows, np.stack(arrays), np.asarray(labels, dtype=np.float32), reps, orders)


def freeze_folds(rows):
    """接口：仅按视频身份和种子分五折；不读取答案或为偏好数量调整分折。"""
    if any(r['split'] != 'train' for r in rows): raise ValueError('Validation rows cannot enter folds')
    videos = sorted({r['video_id'] for r in rows}, key=lambda v:stable_key('set_value_diagnostic_cv', v))
    if len(videos) != 300 or len(rows) != 600: raise ValueError('Fixed fold population differs')
    membership = {v:i % 5 for i, v in enumerate(videos)}
    # 步骤1按视频分组；步骤2同步冻结题目索引，防止同视频两题跨折。
    folds = []
    for fold in range(5):
        held = [i for i, row in enumerate(rows) if membership[row['video_id']] == fold]
        train = [i for i in range(len(rows)) if i not in set(held)]
        if len(held) != 120 or len({rows[i]['video_id'] for i in held}) != 60:
            raise ValueError('Fold sizes differ')
        folds.append(dict(fold=fold, train_indices=train, heldout_indices=held,
                          train_question_ids=[rows[i]['question_id'] for i in train],
                          heldout_question_ids=[rows[i]['question_id'] for i in held]))
    return dict(seed=2027, scope='set_value_diagnostic_cv', video_membership=membership, folds=folds)


def select_methods(scores, source_orders):
    """接口：仅分数与源顺序→方法索引；没有正确性标签，沿用1e-8数值同分规则。"""
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError('Invalid set scores')
    return [min((j for j, s in enumerate(row) if float(row.max()-s) <= 1e-8), key=lambda j:order[j])
            for row, order in zip(values, source_orders)]


def score_inputs(model, features):
    """接口：只接收输入特征，返回固定小头分数；预测阶段不接收留出标签。"""
    model.eval()
    with torch.inference_mode():
        return model(torch.as_tensor(features, dtype=torch.float32).reshape(-1, 8, 770)).reshape(len(features), 4).numpy()


def evaluate_training(model, data, include_gradients=True):
    """接口：训练对象内部计算损失/偏好/选组表现，停止条件只依赖本对象。"""
    # 步骤1：按原600题矩阵形状评分，保证40步选择控制与原实现一致。
    scores = score_inputs(model, data.features); choices = select_methods(scores, data.source_orders)
    usable = data.usable(); preferences, margins = [], []
    for i in usable:
        pairs = []
        for a in data.representatives[i]:
            for b in data.representatives[i]:
                if data.labels[i, a] == 1 and data.labels[i, b] == 0:
                    delta = float(scores[i, a]-scores[i, b]); margins.append(delta)
                    pairs.append(1. if delta > 1e-8 else .5 if abs(delta) <= 1e-8 else 0.)
        preferences.append(float(np.mean(pairs)))
    if not usable: raise ValueError('No within-question training preference')
    # 步骤2：全有效题平均损失和梯度单独记录，不混用epoch内批次损失。
    model.train(); model.zero_grad(set_to_none=True)
    xx = torch.from_numpy(data.features[usable]); yy = torch.from_numpy(data.labels[usable])
    ss = model(xx.reshape(-1, 8, 770)).reshape(len(usable), 4)
    loss = preference_loss(ss, yy, [data.representatives[i] for i in usable])
    gradients = {}
    if include_gradients:
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all(): raise ValueError('Missing/nonfinite gradient')
            gradients[name] = float(p.grad.norm())
    result = dict(questions=len(data.rows), usable_questions=len(usable),
                  preference_accuracy=float(np.mean(preferences)),
                  usable_top1_accuracy=float(np.mean([data.labels[i, choices[i]] for i in usable])),
                  all_correct_count=int(sum(data.labels[i, c] for i, c in enumerate(choices))),
                  mean_preference_loss=float(loss.detach()), gradient_norms=gradients,
                  margin_quantiles=np.quantile(margins, [0, .1, .5, .9, 1]).tolist(), choices=choices)
    model.zero_grad(set_to_none=True)
    return result


def passed_fit(metrics, spec):
    """接口：仅有效训练题上的两个95%门槛决定连续通过计数。"""
    return (metrics['preference_accuracy'] >= spec['required_accuracy'] and
            metrics['usable_top1_accuracy'] >= spec['required_accuracy'])


def parameter_digest(state):
    """接口：稳定摘要不依赖torch.save容器时间，用于确认各折独立同种子初始化。"""
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def state_error(left, right):
    """接口：递归比较参数/优化器，不忽略momentum、step或参数组配置。"""
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError('Checkpoint tensor shape/dtype differs')
        if not torch.isfinite(left).all() or not torch.isfinite(right).all(): raise ValueError('Nonfinite checkpoint')
        return float((left-right).abs().max()) if left.numel() else 0.
    if isinstance(left, dict):
        if set(left) != set(right): raise ValueError('Checkpoint keys differ')
        return max((state_error(left[k], right[k]) for k in left), default=0.)
    if isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right): raise ValueError('Checkpoint list differs')
        return max((state_error(a, b) for a, b in zip(left, right)), default=0.)
    if left != right: raise ValueError('Optimizer scalar/config differs')
    return 0.


def save_state(path, value):
    """接口：每次参数更新后原子持久化；恢复从最近已提交step继续，不重复应用更新。"""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix('.part')
    with temp.open('wb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def train_trajectory(data, output, allowed_question_ids, spec=None, reference=None, replay_choices=None,
                     stop_after=None):
    """接口：仅给定训练折的小头轨迹；留出数据不在参数中，完成恢复不训练。

    stop_after仅供恢复测试模拟中断，正式入口不暴露此项。reference只用于全训练40步重放，
    交叉验证必须传None，因此不会继承包含留出标签的旧检查点。
    """
    output = Path(output); output.mkdir(parents=True, exist_ok=True); spec = dict(SPEC if spec is None else spec)
    if [r['question_id'] for r in data.rows] != list(allowed_question_ids): raise ValueError('Training fold membership differs')
    identity = digest(dict(spec=spec, questions=allowed_question_ids,
                           x=hashlib.sha256(data.features.tobytes()).hexdigest(),
                           y=data.labels.tolist(), reps=data.representatives, reference=reference is not None))
    summary_path = output/'summary.json'; checkpoint = output/'state.pt'
    if summary_path.exists():
        result = read_json(summary_path)
        if result['identity'] != identity or result['checkpoint_sha256'] != sha256(checkpoint):
            raise ValueError('Completed trajectory fingerprint differs')
        return result
    # 步骤1：每条轨迹独立初始化，不从全数据检查点加载；重放参考只做比较。
    torch.manual_seed(spec['seed']); model = SetValueHead().float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    initialization = parameter_digest(model.state_dict()); usable = data.usable()
    if not usable: raise ValueError('Cannot train a fold without preferences')
    features = torch.from_numpy(data.features); labels = torch.from_numpy(data.labels)
    batches = math.ceil(len(usable)/spec['batch_questions']); previous_seconds = 0.
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['identity'] != identity: raise ValueError('Partial trajectory fingerprint differs')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        previous_seconds = state['training_seconds']
    else:
        initial = evaluate_training(model, data); initial.pop('choices')
        state = dict(identity=identity, step=0, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     initialization_sha256=initialization, history=[dict(step=0, **initial)], consecutive=0,
                     replay_40=None, status='running', training_seconds=0.)
        save_state(checkpoint, state)
    began = time.monotonic()
    # 步骤2：按原epoch散列顺序确定每批；每step提交游标、参数与优化器状态。
    while state['step'] < spec['max_updates'] and state['status'] == 'running':
        step = state['step']; epoch, batch_index = divmod(step, batches)
        order = np.random.RandomState(spec['seed']+epoch).permutation(usable)
        batch = order[batch_index*spec['batch_questions']:(batch_index+1)*spec['batch_questions']]
        model.train(); scores = model(features[batch].reshape(-1, 8, 770)).reshape(len(batch), 4)
        loss = preference_loss(scores, labels[batch], [data.representatives[int(i)] for i in batch])
        if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite/missing optimization gradient')
        optimizer.step(); state['step'] += 1
        if state['step'] % spec['check_every'] == 0:
            metrics = evaluate_training(model, data)
            # 步骤3：40步控制同时核对原权重、完整优化器和600题训练选择。
            if reference is not None and state['step'] == 40:
                model_error = state_error(model.state_dict(), reference['model'])
                optimizer_error = state_error(optimizer.state_dict(), reference['optimizer'])
                if max(model_error, optimizer_error) > 1e-6: raise ValueError('Original 40-step replay differs')
                if replay_choices is not None and metrics['choices'] != list(replay_choices):
                    raise ValueError('Original 40-step training choices differ')
                state['replay_40'] = dict(model_abs_error=model_error, optimizer_abs_error=optimizer_error,
                                          training_choices_exact=True if replay_choices is not None else None)
            metrics.pop('choices'); state['history'].append(dict(step=state['step'], **metrics))
            state['consecutive'] = state['consecutive']+1 if passed_fit(metrics, spec) else 0
            if state['consecutive'] >= spec['consecutive_checks']:
                state['status'] = 'fit_passed'
            durable(output/'progress.json', dict(step=state['step'], maximum=spec['max_updates'],
                    preference_accuracy=metrics['preference_accuracy'], usable_top1_accuracy=metrics['usable_top1_accuracy'],
                    consecutive=state['consecutive'], status=state['status'], pid=os.getpid(), utc=utc()))
        if state['step'] == spec['max_updates'] and state['status'] == 'running': state['status'] = 'budget_exhausted'
        state.update(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     training_seconds=previous_seconds+time.monotonic()-began)
        save_state(checkpoint, state)
        if stop_after is not None and state['step'] >= stop_after and state['status'] == 'running':
            return dict(status='interrupted_for_test', step=state['step'])
    if reference is not None and state['replay_40'] is None: raise ValueError('40-step control was not reached')
    result = dict(identity=identity, status=state['status'], updates=state['step'], spec=spec,
                  train_questions=len(data.rows), usable_questions=len(usable),
                  usable_videos=len({data.rows[i]['video_id'] for i in usable}),
                  initialization_sha256=initialization, checkpoint_sha256=sha256(checkpoint),
                  replay_40=state['replay_40'], history=state['history'],
                  training_seconds=state['training_seconds'], utc=utc())
    durable(summary_path, result)
    return result


def describe_case(question_id, video_id, task_type, stratum, duration, labels, keys, frame_sets):
    """接口：四候选的反馈/完整输入/源帧差异，仅描述统计，不生成训练标签。"""
    groups = []
    for i, key in enumerate(keys):
        group = next((g for g in groups if g['key'] == key), None)
        if group is None: groups.append(dict(key=key, indices=[i]))
        else: group['indices'].append(i)
    conflict = any(len({labels[i] for i in g['indices']}) != 1 for g in groups)
    reps = [g['indices'][0] for g in groups]; positives = sum(labels[i] for i in reps)
    pairs = 0 if conflict else positives*(len(reps)-positives)
    overlaps = [len(a & b)/len(a | b) for a, b in itertools.combinations(frame_sets, 2)]
    state = 'all_correct' if all(labels) else 'all_wrong' if not any(labels) else 'mixed'
    return dict(question_id=question_id, video_id=video_id, task_type=task_type, stratum=stratum,
                duration_seconds=float(duration), duration_bin=next(name for bound, name in TIME_BINS if duration <= bound),
                correct=list(map(int, labels)), full_input_keys=keys, unique_inputs=len(groups),
                complete_input_equal_pairs=sum(len(g['indices'])*(len(g['indices'])-1)//2 for g in groups),
                complete_input_conflict=conflict, preference_pairs=int(pairs), state=state,
                source_frame_jaccard=overlaps, mean_source_frame_jaccard=float(np.mean(overlaps)))


def aggregate_cases(cases):
    """接口：题目/视频分母和去重偏好密度；状态视频数允许交叠，不伪造视频划分。"""
    if not cases: return dict(questions=0, videos=0)
    counts = Counter(c['state'] for c in cases); best = np.asarray([c['correct'] for c in cases]).sum(0)
    return dict(questions=len(cases), videos=len({c['video_id'] for c in cases}), states=dict(counts),
                state_videos={s:len({c['video_id'] for c in cases if c['state'] == s}) for s in ('all_correct','all_wrong','mixed')},
                mixed_fraction=counts['mixed']/len(cases),
                all_four_inputs_same=sum(c['unique_inputs'] == 1 for c in cases),
                any_complete_input_pair_same=sum(c['unique_inputs'] < 4 for c in cases),
                complete_input_equal_pair_fraction=sum(c['complete_input_equal_pairs'] for c in cases)/(6*len(cases)),
                duplicate_method_inputs=sum(4-c['unique_inputs'] for c in cases),
                conflicting_same_input_questions=sum(c['complete_input_conflict'] for c in cases),
                unique_preference_pairs=sum(c['preference_pairs'] for c in cases),
                unambiguous_mixed_questions=sum(c['state'] == 'mixed' and not c['complete_input_conflict'] for c in cases),
                fixed_method_correct=dict(zip(METHODS, map(int, best))), any_candidate_correct=sum(any(c['correct']) for c in cases),
                bank_gap_to_best_fixed=int(sum(any(c['correct']) for c in cases)-best.max()),
                duration_quantiles=np.quantile([c['duration_seconds'] for c in cases], [0,.5,.9,1]).tolist(),
                mean_source_frame_jaccard=float(np.mean([c['mean_source_frame_jaccard'] for c in cases])),
                source_frame_jaccard_quantiles=np.quantile([v for c in cases for v in c['source_frame_jaccard']], [.1,.5,.9]).tolist())


def grouped_description(cases):
    """步骤1总体；步骤2固定时长分箱；步骤3原任务/官方时长标签，不重分类问题。"""
    return dict(all=aggregate_cases(cases),
                by_duration={name:aggregate_cases([c for c in cases if c['duration_bin'] == name]) for _, name in TIME_BINS},
                by_task={name:aggregate_cases([c for c in cases if c['task_type'] == name]) for name in sorted({c['task_type'] for c in cases})},
                by_stratum={name:aggregate_cases([c for c in cases if c['stratum'] == name]) for name in sorted({c['stratum'] for c in cases})})


def nextqa_cases(data, source, tracker):
    """接口：仅训练600题的输入计划和旧时钟元数据，未访问原验证特征或答案。"""
    cases = []
    for i, row in enumerate(data.rows):
        plan = tracker.read(Path(source)/'inputs'/(row['question_id']+'.json'))
        duration = tracker.read(row['index_path'])['video']['duration_seconds']
        frames = [{tuple((p[1], p[0])) for p in order} for order in data.source_orders[i]]
        cases.append(describe_case(row['question_id'], row['video_id'], row['task_type'], 'official_train', duration,
                                  data.labels[i].astype(int).tolist(), [plan['methods'][m]['input_key'] for m in METHODS], frames))
    return cases


def target_cases(review, dataset, profile, tracker, summary, historical_sources):
    """接口：目标数据只生成描述记录；原始实际输入身份绑定官方处理像素和提示。"""
    review = Path(review); cases = []
    paths = sorted((review/'records'/dataset/profile).glob('*.json'))
    expected = 2700 if dataset == 'videomme' else 2174
    if len(paths) != expected: raise ValueError('Target descriptive population incomplete')
    for ordinal, path in enumerate(paths):
        record = tracker.read(path, summary['exports'][str(path.relative_to(review))]); labels, keys, frames = [], [], []
        duration = None
        for method in METHODS:
            original_path = record['answer_sources'][method]
            original = tracker.read(original_path, historical_sources[original_path])
            answer = original['answer']; obs = answer['observed']; inp = record['inputs'][method]
            if original['question_id'] != record['question_id'] or answer['raw_output'] != record['answers'][method]['raw_output']:
                raise ValueError('Frozen target answer binding differs')
            if bool(original['correct']) != record['answers'][method]['native_correct']:
                raise ValueError('Target native score differs')
            actual_frames = [[f['source_frame_index'], f['source_pts']] for f in original['selection']['selected_frames']]
            if actual_frames != inp['source_frames'] or original['rgb_sha256'] != inp['rgb_sha256']:
                raise ValueError('Target input/source mapping differs')
            pixel = obs.get('pixel_sha256', obs.get('model_pixel_sha256'))
            if pixel is None: raise ValueError('Actual target processed pixel identity missing')
            identity = dict(profile=profile, source_frames=actual_frames, rgb=inp['rgb_sha256'],
                            pixel_sha256=pixel, prompt=obs['prompt'], input_token_ids=obs['input_token_ids'],
                            visual_tokens=obs['visual_tokens'], model_generation_config=obs['model_generation_config'],
                            generation_kwargs=obs.get('generation_kwargs'), max_new_tokens=obs.get('max_new_tokens',16),
                            video_grid_thw=obs.get('video_grid_thw'), encoding_fps=obs.get('encoding_fps'))
            keys.append(digest(identity)); labels.append(int(record['answers'][method]['correct']))
            frames.append(set(map(tuple, actual_frames)))
            value = original['selection']['video']['duration_seconds']
            if duration is not None and abs(duration-value) > 1e-8: raise ValueError('Target video duration changed')
            duration = value
        cases.append(describe_case(record['question_id'], record['video_id'], record['task_type'], record['stratum'],
                                  duration, labels, keys, frames))
    return cases


def signal_audit(data, source):
    """接口：只读原40步检查点与训练偏好，检查信号碰撞、更新及梯度。"""
    torch.manual_seed(2027); model = SetValueHead().float(); initial = {k:v.detach().clone() for k,v in model.state_dict().items()}
    saved = torch.load(Path(source)/'head/checkpoint.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model']); stats = evaluate_training(model, data)
    metrics = dict(stats); metrics.pop('choices'); differences, collisions = [], 0
    for i in data.usable():
        for a in data.representatives[i]:
            for b in data.representatives[i]:
                if data.labels[i,a] == 1 and data.labels[i,b] == 0:
                    differences.append(float(np.linalg.norm(data.features[i,a]-data.features[i,b])))
                    collisions += bool(np.array_equal(data.features[i,a], data.features[i,b]))
    return dict(optimizer_steps=sorted({int(v['step']) for v in saved['optimizer']['state'].values()}),
                unique_positive_negative_pairs=len(differences), exact_tensor_collisions=collisions,
                feature_difference_norm_quantiles=np.quantile(differences,[0,.1,.5,.9,1]).tolist(),
                parameter_update_norms={k:float((v-initial[k]).norm()) for k,v in model.state_dict().items()},
                training_metrics=metrics), saved, stats['choices']


def cluster_comparison(rows, learned, baseline):
    """接口：600折外题按300个视频簇重抽，保留每视频两题及完整四格题号。"""
    a = np.asarray(learned, dtype=int); b = np.asarray(baseline, dtype=int); groups = defaultdict(list)
    for i, row in enumerate(rows): groups[row['video_id']].append(i)
    means = np.asarray([float((a[ids]-b[ids]).mean()) for ids in groups.values()])
    rng = np.random.default_rng(2027); sampled = []
    for _ in range(50): sampled.extend(means[rng.integers(0,len(means),size=(200,len(means)))].mean(1).tolist())
    masks = dict(new_correct=(a==1)&(b==0), degraded=(a==0)&(b==1), both_correct=(a==1)&(b==1), both_wrong=(a==0)&(b==0))
    return dict(net_correct=int((a-b).sum()), difference=float((a-b).mean()),
                cluster_95_interval=np.quantile(sampled,[.025,.975]).tolist(),
                **{name:[rows[i]['question_id'] for i in np.flatnonzero(mask)] for name,mask in masks.items()})


def cross_validate(data, folds, output, spec=None):
    """接口：最多五条独立初始化轨迹，冻结留出预测后才读取标签计算折外结果。"""
    output = Path(output); predictions = [None]*len(data.rows); summaries = []
    for fold in folds['folds']:
        directory = output/('fold_'+str(fold['fold'])); train = data.subset(fold['train_indices'])
        result = train_trajectory(train, directory, fold['train_question_ids'], spec=spec)
        summaries.append(result)
        # 步骤1：训练结束后才取留出输入；预测函数没有标签参数。
        path = directory/'heldout_predictions.json'
        if path.exists(): frozen = read_json(path)
        else:
            model = SetValueHead().float(); saved = torch.load(directory/'state.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model'])
            ids = fold['heldout_indices']; scores = score_inputs(model, data.features[ids])
            choices = select_methods(scores, [data.source_orders[i] for i in ids])
            itm = select_methods(data.features[ids,:,:,768].mean(2),
                                 [data.source_orders[i] for i in ids])
            baseline = int(np.argmax(train.labels.sum(0)))
            frozen = dict(fold=fold['fold'], checkpoint_sha256=sha256(directory/'state.pt'),
                          training_selected_fixed_method=METHODS[baseline],
                          predictions=[dict(question_id=data.rows[i]['question_id'],scores=[float(v) for v in s],
                                            chosen_index=int(c),mean_itm_index=int(m),baseline_index=baseline)
                                       for i,s,c,m in zip(ids,scores,choices,itm)], uses_heldout_labels=False,utc=utc())
            durable(path,frozen)
        if frozen['checkpoint_sha256'] != sha256(directory/'state.pt'): raise ValueError('Fold prediction model changed')
        # 步骤2：只在预测发布后匹配旧标签；这些结果不能进入训练或停止条件。
        for i, p in zip(fold['heldout_indices'], frozen['predictions']):
            if p['question_id'] != data.rows[i]['question_id'] or predictions[i] is not None:
                raise ValueError('OOF prediction pairing differs')
            predictions[i] = p
    if any(p is None for p in predictions): raise ValueError('Incomplete OOF population')
    # 步骤3：统一合并600题；按题平均偏好而非按比较对数加权。
    values = {m:data.labels[:,j].astype(int).tolist() for j,m in enumerate(METHODS)}
    for name, key in [('learned_oof','chosen_index'),('training_fold_best','baseline_index'),('mean_itm','mean_itm_index')]:
        values[name] = [int(data.labels[i,p[key]]) for i,p in enumerate(predictions)]
    preferences = []; preference_videos = set()
    for i in data.usable():
        p = predictions[i]; pairs=[]
        for a in data.representatives[i]:
            for b in data.representatives[i]:
                if data.labels[i,a] == 1 and data.labels[i,b] == 0:
                    d = p['scores'][a]-p['scores'][b]; pairs.append(1. if d>1e-8 else .5 if abs(d)<=1e-8 else 0.)
        preferences.append(float(np.mean(pairs))); preference_videos.add(data.rows[i]['video_id'])
    comparisons = {m:cluster_comparison(data.rows,values['learned_oof'],v) for m,v in values.items() if m!='learned_oof'}
    gains=set(comparisons['training_fold_best']['new_correct'])
    gain_videos=len({r['video_id'] for r in data.rows if r['question_id'] in gains})
    correct={k:sum(v) for k,v in values.items()}
    criteria=dict(preference_at_least_sixty=float(np.mean(preferences))>=.6,
                  net_gain_over_training_fold_best=comparisons['training_fold_best']['net_correct']>0,
                  improvements_in_at_least_three_videos=gain_videos>=3,
                  no_lower_than_best_fixed=correct['learned_oof']>=max(correct[m] for m in METHODS))
    return dict(questions=len(data.rows), videos=len({r['video_id'] for r in data.rows}),
                correct=correct,preference_questions=len(preferences),preference_videos=len(preference_videos),
                preference_accuracy=float(np.mean(preferences)),comparisons=comparisons,
                choice_counts=dict(Counter(METHODS[p['chosen_index']] for p in predictions)),
                criteria={k:bool(v) for k,v in criteria.items()},suggest_independent_supervision=all(criteria.values()),
                fold_summaries=summaries,predictions=predictions,
                limitation='已观察训练数据上的回顾性视频交叉验证；不是原200题或新独立基准成绩。')


def sealed_artifacts(source):
    """接口：仅校验原恢复审计中的文件哈希；验证200题内容不进入新模型或计分。"""
    source = Path(source); audit = read_json(source/'completed_resume_audit.json')
    expected = audit['artifacts']
    for rel, value in expected.items():
        if sha256(source/rel) != value: raise ValueError('Original sealed artifact changed: '+rel)
    return dict(files=len(expected), digest=digest(expected), validation_re_evaluated=False)


def render_artifacts(output, result):
    """接口：用标准绘图工具输出训练曲线与监督密度，静态页只展示已有统计。"""
    import html
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output); series=[('Full training',result['full_fit'])]
    if result['oof']:
        series += [('CV fold '+str(i),v) for i,v in enumerate(result['oof']['fold_summaries'])]
    # 步骤1：完整训练轨迹和折训练轨迹共用坐标，不用留出分数挑曲线片段。
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    for name,summary in series:
        h=summary['history'];steps=[p['step'] for p in h]
        for ax,key in zip(axes,('mean_preference_loss','preference_accuracy','usable_top1_accuracy')):
            ax.plot(steps,[p[key] for p in h],label=name,lw=1.5)
    for ax,title in zip(axes,('Question-balanced training loss','Training preference accuracy','Training top-1 on mixed questions')):
        ax.set_title(title);ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2)
    for ax in axes[1:]:ax.axhline(.95,color='gray',ls='--',lw=1);ax.set_ylim(0,1.03)
    axes[0].legend(fontsize=8);fig.savefig(output/'training_curves.png',dpi=170)
    fig.savefig(output/'training_curves.svg');plt.close(fig)
    # 步骤2：监督密度的分母直接写在柱子上，不能把目标配置差异归因于单一预算。
    groups=list(result['supervision'].items());fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    bars=ax.bar(range(len(groups)),[100*v['all']['mixed_fraction'] for _,v in groups])
    ax.set_xticks(range(len(groups)),[k for k,_ in groups],rotation=16,ha='right')
    ax.set_ylabel('Mixed-correctness questions (%)');ax.grid(axis='y',alpha=.2)
    for bar,(_,value) in zip(bars,groups):
        a=value['all'];ax.annotate(str(a['states'].get('mixed',0))+'/'+str(a['questions']),
                                  (bar.get_x()+bar.get_width()/2,bar.get_height()),ha='center',va='bottom',fontsize=9)
    fig.savefig(output/'supervision_density.png',dpi=170);fig.savefig(output/'supervision_density.svg');plt.close(fig)
    # 步骤3：浏览页仅链接本轮CPU产物，不重新请求任何模型回答。
    escaped=html.escape(json.dumps({k:v for k,v in result.items() if k not in ('oof','supervision','full_fit')},ensure_ascii=False,indent=2))
    page='<!doctype html><meta charset="utf-8"><title>集合价值训练诊断</title><style>body{max-width:1100px;margin:24px auto;font:16px/1.6 sans-serif}img{max-width:100%}pre{white-space:pre-wrap;background:#eee;padding:14px}</style>'
    page+='<h1>集合价值学习：训练充分性与监督适配诊断</h1><p>仅CPU小头训练；原200题封存。折外结果属于已观察训练数据的回顾性诊断。</p><pre>'+escaped+'</pre>'
    page+='<img src="training_curves.png" alt="Training curves"><img src="supervision_density.png" alt="Supervision density">'
    page+='<p><a href="report.json">完整报告</a> · <a href="folds.json">固定分折</a> · <a href="supervision.json">分组统计</a></p>'
    if result['oof']:
        page+='<h2>600题折外结果</h2><pre>'+html.escape(json.dumps({k:v for k,v in result['oof'].items() if k not in ('fold_summaries','predictions')},ensure_ascii=False,indent=2))+'</pre>'
    (output/'index.html').write_text(page)


def run(output, source=SOURCE, review=REVIEW):
    """接口：CPU后台独立诊断，先冻结来源/折，再重放拟合，通过后只在600题内交叉验证。"""
    output=Path(output); output.mkdir(parents=True,exist_ok=True); source=Path(source); review=Path(review)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    with exclusive(output/'analysis.lock'):
        # 步骤1：完成恢复仅核验来源/派生物，不再次参数更新。
        if (output/'completion.json').exists():
            result=read_json(output/'completion.json')
            for p,h in result['sources'].items():
                if sha256(p)!=h:raise ValueError('Completed diagnostic source changed')
            for rel,h in result['artifacts'].items():
                if sha256(output/rel)!=h:raise ValueError('Completed diagnostic artifact changed')
            sealed_artifacts(source); return read_json(output/'report.json')
        tracker=Sources(); data=load_training_assets(source,tracker)
        tracker.check(Path(__file__))
        tracker.check(ROOT/'scripts/analyze_set_value_trainability.py')
        protection=sealed_artifacts(source); folds=freeze_folds(data.rows)
        heldout_mixed=[len(data.subset(f['heldout_indices']).usable()) for f in folds['folds']]
        if heldout_mixed!=[6,13,10,8,10]:raise ValueError('Frozen fold preference counts differ')
        protocol=dict(task='set_value_trainability',source_run=str(source),descriptive_review=str(review),spec=SPEC,
                      original_validation_sealed=True,new_backbone_forwards=0,new_generations=0,
                      heldout_mixed_questions=heldout_mixed,max_trajectories=6,max_updates=12000,code={str(Path(__file__)):sha256(__file__),
                      str(ROOT/'scripts/analyze_set_value_trainability.py'):sha256(ROOT/'scripts/analyze_set_value_trainability.py')})
        for name,value in [('protocol.json',protocol),('folds.json',folds)]:
            path=output/name
            if path.exists() and read_json(path)!=value:raise ValueError('Diagnostic configuration changed')
            if not path.exists():durable(path,value)
        durable(output/'heartbeat.json',dict(stage='supervision_description',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤2：描述银行与训练严格分开，目标结果只传给汇总器。
        if not (output/'supervision.json').exists():
            cases={'nextqa/train/llava8':nextqa_cases(data,source,tracker)}
            summary=tracker.read(review/'summary.json'); historical=tracker.read(review/'sources.json',summary['sources_sha256'])
            for dataset,profile in itertools.product(('videomme','mlvu'),('llava8','qwen32')):
                durable(output/'heartbeat.json',dict(stage='describing_'+dataset+'_'+profile,pid=os.getpid(),utc=utc()))
                cases[dataset+'/'+profile]=target_cases(review,dataset,profile,tracker,summary,historical)
            descriptions={name:grouped_description(items) for name,items in cases.items()}
            durable(output/'descriptive_cases.json',cases); durable(output/'supervision.json',descriptions)
            durable(output/'descriptive_sources.json',tracker.files)
        else:
            tracker.files.update(read_json(output/'descriptive_sources.json'))
            descriptions=read_json(output/'supervision.json')
        audit,reference,old_choices=signal_audit(data,source); durable(output/'training_signal.json',audit)
        tracker.check(source/'head/checkpoint.pt')
        # 步骤3：一条完全相同的原40步路径，随后固定上限与训练停止规则。
        durable(output/'heartbeat.json',dict(stage='full_training_fit',pid=os.getpid(),utc=utc()))
        full=train_trajectory(data,output/'full_fit',[r['question_id'] for r in data.rows],reference=reference,replay_choices=old_choices)
        oof=None
        if full['status']=='fit_passed':
            durable(output/'heartbeat.json',dict(stage='video_cross_validation',pid=os.getpid(),utc=utc()))
            oof=cross_validate(data,folds,output/'cv')
            durable(output/'oof.json',oof)
        # 步骤4：所有条件由事前规则决定，原200题和目标数据不参与选检查点。
        total=full['updates']+(sum(f['updates'] for f in oof['fold_summaries']) if oof else 0)
        if total>12000:raise ValueError('Diagnostic update budget exceeded')
        if oof and len({f['initialization_sha256'] for f in oof['fold_summaries']}|{full['initialization_sha256']})!=1:
            raise ValueError('Fold initialization differs')
        tracker.verify(); after=sealed_artifacts(source)
        if after!=protection:raise ValueError('Original experiment changed')
        result=dict(status='completed',decision='training_not_fitted' if oof is None else
                    'propose_independent_supervision' if oof['suggest_independent_supervision'] else 'generalization_not_supported',
                    full_fit=full,oof=oof,supervision=descriptions,training_signal=audit,
                    total_optimizer_updates=total,trajectories=1+(5 if oof else 0),
                    original_experiment_protection=after,new_backbone_forwards=0,new_generations=0,
                    new_original_validation_predictions=0,utc=utc())
        durable(output/'report.json',result)
        render_artifacts(output,result)
        artifacts={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()
                   and p.name not in ('analysis.lock','heartbeat.json','coordinator.log','launcher.json')}
        durable(output/'completion.json',dict(status='completed',sources=tracker.files,artifacts=artifacts,utc=utc()))
        durable(output/'heartbeat.json',dict(stage='completed',pid=os.getpid(),utc=utc()))
        return result
