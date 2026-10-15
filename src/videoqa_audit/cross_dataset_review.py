"""双数据集的只读计分与选择路径审计；不导入任何模型或调用生成入口。"""
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import re

import numpy as np

METHODS = ('BASE-Uniform', 'BASE-BLIP-TopK', 'RD-1.2-exp07', 'RD-1.2-exp12')
TASKS = ('plotQA', 'needle', 'ego', 'count', 'order', 'anomaly_reco', 'topic_reasoning')
PERIOD_POLICY = 'single-uppercase-option-period-v1'


def normalize_option_period(raw_output, native_prediction):
    """接口：仅把完整A./B./C./D.映射为字母；不从解释文字猜答案，也不接收标准答案。"""
    # 步骤1：只允许首尾空白和一个英文句点；旧解析有效的预测优先保持。
    if native_prediction in ('A', 'B', 'C', 'D'):
        return native_prediction
    match = re.fullmatch(r'\s*([ABCD])\.\s*', raw_output)
    # 步骤2：其他输出完整保留，避免把格式诊断扩展为宽松内容解析。
    return match.group(1) if match else native_prediction


def freeze_balanced_subset(identity_rows, per_task=30, seed=2027):
    """接口：仅接收题号/任务/内容视频身份，冻结每类30题；不读取问题答案或历史成绩。"""
    if not identity_rows or per_task < 1:
        raise ValueError('Invalid subset scope')
    required = {'question_id', 'task_type', 'video_id'}
    if any(set(row) != required for row in identity_rows):
        raise ValueError('Subset input must contain identity fields only')
    if len({r['question_id'] for r in identity_rows}) != len(identity_rows):
        raise ValueError('Duplicate subset question')
    selected = []
    used_videos = set()
    # 步骤1：固定任务顺序；种子和身份的SHA256排序可跨Python版本重放。
    for task in TASKS:
        group = [r for r in identity_rows if r['task_type'] == task]
        def key(row):
            payload = f"{seed}|{task}|{row['video_id']}|{row['question_id']}".encode()
            return hashlib.sha256(payload).hexdigest(), row['question_id']
        count = 0
        for row in sorted(group, key=key):
            if row['video_id'] in used_videos:
                continue
            selected.append(dict(row))
            used_videos.add(row['video_id'])
            count += 1
            if count == per_task:
                break
        # 步骤2：不足时显式停止，不读取答案换题、不跨任务挪用名额。
        if count != per_task:
            raise ValueError('Insufficient independent videos for task: ' + task)
    return dict(rows=selected, seed=seed, per_task=per_task,
                ordering='SHA256(seed|task|content_video_id|question_id), fixed task order',
                question_count=len(selected), unique_content_videos=len(used_videos),
                answer_fields_read=False, history_predictions_read=False,
                limitation='Previously evaluated public data; not a blind holdout')


def audit_selection_paths(pool, left, right, budget):
    """接口：同池共同前缀下追踪Top-K粗帧的配额可达性，不推断语义证据或答案上限。"""
    from videoqa_audit.coverage_objective_review import compare_allocations, selected_indices
    from videoqa_methods.density_selection import frame_key
    comparison = compare_allocations(pool, left, right)
    candidates, scores = pool['candidates'], pool['initial_scores']
    n = pool['initial_count']
    if len(scores) != n or not 0 < n <= len(candidates):
        raise ValueError('Coarse identity mismatch')
    # 步骤1：稳定排序重建原Top-K定位，全部初始粗帧必须具有唯一选择归属。
    top = sorted(range(n), key=lambda i: (-scores[i], *frame_key(candidates[i])))[:budget]
    owners = {}
    for region in left['regions']:
        for i in region['reference_coarse_indices']:
            if i in owners:
                raise ValueError('Duplicate coarse owner')
            owners[i] = region
    if set(owners) != set(range(n)):
        raise ValueError('Coarse ownership is incomplete')
    selected = {name: set(selected_indices(trace)) for name, trace in [('exp07', left), ('exp12', right)]}
    actual_budget = comparison['budget']
    if actual_budget != min(budget, len(candidates)):
        raise ValueError('Actual budget mismatch')
    paths = []
    # 步骤2：实际前缀已截到K；考虑其余每区一个保护锚点后计算最大可行配额。
    for rank, i in enumerate(top, 1):
        region = owners[i]
        rid = str(region['region_id'])
        position = region['prefix'].index(i) + 1 if i in region['prefix'] else None
        capacity = min(len(region['prefix']), actual_budget - len(left['regions']) + 1)
        parents = {}
        for name, trace in [('exp07', left), ('exp12', right)]:
            quota = trace['region_budgets'][rid]
            chosen = i in selected[name]
            if chosen != (position is not None and position <= quota):
                raise ValueError('Prefix and source selection disagree')
            state = ('selected' if chosen else 'outside_prefix' if position is None else
                     'beyond_feasible_quota' if position > capacity else 'quota_reachable')
            parents[name] = dict(state=state, quota=quota,
                                 additional_slots=None if position is None else max(0, position-quota))
        paths.append(dict(candidate_index=i, source_frame_index=candidates[i]['source_frame_index'],
                          source_pts=candidates[i]['source_pts'], seconds=candidates[i]['timestamp_seconds'],
                          rank=rank, itm=scores[i], region_id=region['region_id'],
                          prefix_position=position, max_feasible_quota=capacity, **parents))
    # 步骤3：这些指标仅描述检索帧被保留的路径；近似替代须另经实际图像审核。
    return dict(actual_budget=actual_budget, coarse_count=n, fine_count=len(candidates)-n,
                region_count=len(left['regions']), anchor_fraction=len(left['regions'])/actual_budget,
                transferred_slots=comparison['transferred_slots'], selected_overlap=comparison['selected_overlap'],
                selected_jaccard=comparison['selected_jaccard'], retained_seeds=comparison['retained_seeds'],
                selected_fine=comparison['selected_fine'], topk_paths=paths,
                selected_indices={k: sorted(v) for k, v in selected.items()},
                regions=[dict(region_id=r['region_id'], prefix=r['prefix'],
                              coarse_references=len(r['reference_coarse_indices']),
                              members=len(r['members']), density_weight=r['density_weight'],
                              exp07_quota=left['region_budgets'][str(r['region_id'])],
                              exp12_quota=right['region_budgets'][str(r['region_id'])]) for r in left['regions']],
                limitation='Top-K retention is not evidence recall; fixed-prefix capacity only')


def cluster_report(rows, banks, macro=False, repetitions=10000, seed=2027):
    """接口：四方法配对计分；按内容视频为簇，Video-MME另按时长分层重抽。"""
    methods = list(banks)
    tasks = list(TASKS) if macro else ['all']
    videos = sorted({r['video_id'] for r in rows})
    vpos = {v: i for i, v in enumerate(videos)}
    sums = np.zeros((len(videos), len(methods), len(tasks)), dtype=np.float64)
    counts = np.zeros((len(videos), len(tasks)), dtype=np.float64)
    strata = {}
    bytask = {m: {} for m in methods}
    # 步骤1：同内容视频跨任务共享簇；每次抽样重新计算宏平均的七个分母。
    for row in rows:
        q, v = row['question_id'], vpos[row['video_id']]
        task = row['task_type'] if macro else 'all'
        j = tasks.index(task)
        counts[v, j] += 1
        layer = 'all' if macro else row['stratum']
        if v in strata and strata[v] != layer:
            raise ValueError('A video crosses duration strata')
        strata[v] = layer
        for k, method in enumerate(methods):
            correct = bool(banks[method][q]['correct'])
            sums[v, k, j] += correct
            name = row['task_type']
            entry = bytask[method].setdefault(name, dict(n=0, correct=0))
            entry['n'] += 1
            entry['correct'] += correct
    denominators = counts.sum(axis=0)
    if np.any(denominators == 0):
        raise ValueError('Missing evaluation task')
    scores = {}
    totals = sums.sum(axis=0)
    for k, method in enumerate(methods):
        scores[method] = dict(n=len(rows), correct=int(totals[k].sum()),
                              accuracy_percent=100*totals[k].sum()/len(rows),
                              main_metric_percent=float(np.mean(100*totals[k]/denominators)),
                              tasks={t: dict(x, accuracy_percent=100*x['correct']/x['n'])
                                     for t, x in bytask[method].items()})
    rng = np.random.default_rng(seed)
    groups = [[i for i, label in strata.items() if label == s] for s in sorted(set(strata.values()))]
    draws = []
    # 步骤2：有界批量生成簇权重，不把同帧引用当作独立样本。
    while sum(len(x) for x in draws) < repetitions:
        size = min(250, repetitions-sum(len(x) for x in draws))
        weights = np.zeros((size, len(videos)), dtype=np.float64)
        for group in groups:
            weights[:, group] = rng.multinomial(len(group), np.full(len(group), 1/len(group)), size)
        d = weights @ counts
        s = (weights @ sums.reshape(len(videos), -1)).reshape(size, len(methods), len(tasks))
        valid = np.all(d > 0, axis=1)
        draws.append(np.mean(100*s[valid]/d[valid, None, :], axis=2))
    samples = np.concatenate(draws)
    pairs = {}
    for ia, ib in itertools.combinations(range(len(methods)), 2):
        a, b = methods[ia], methods[ib]
        cells = {k: [] for k in ('both_correct', 'left_only', 'right_only', 'both_wrong')}
        for row in rows:
            q = row['question_id']
            x, y = banks[a][q]['correct'], banks[b][q]['correct']
            key = 'both_correct' if x and y else 'left_only' if x else 'right_only' if y else 'both_wrong'
            cells[key].append(q)
        pairs[a+' minus '+b] = dict(cells=cells,
            difference_pp=scores[a]['main_metric_percent']-scores[b]['main_metric_percent'],
            video_cluster_95_pp=np.percentile(samples[:, ia]-samples[:, ib], [2.5, 97.5]).tolist())
    return dict(metric='M-Avg' if macro else 'accuracy', scores=scores, pairs=pairs,
                repetitions=repetitions, seed=seed, clusters=len(videos),
                strata='content_video' if macro else 'duration_stratified_video')
