"""exp12双配置组件消融：关闭细帧、重算真实第一遍配额；不修改冻结算法。"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import copy
import os
import shutil
import time

import numpy as np

from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, exclusive, rows, utc, digest
from videoqa_methods.budget_contract import budget_config
from videoqa_methods.coarse_eligibility import coarse_selection_cells, select_global_coarse
from videoqa_methods.coarse_only_ablation import feasible_reference, select_coarse_only
from videoqa_methods.density_selection import frame_key, build_regions
from videoqa_methods.prefix_coverage_budget import select_prefix_coverage
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_audit.density_reward_decay import select_density_decay
from videoqa_budget32.frames import validate_frames

CONTROLS = {'A': 'RD-1.2-exp20', 'B': 'RD-1.2-exp04'}
PROFILES = ('llava8', 'qwen32')
BASE = ROOT/'outputs/full_videoqa/uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01'
PARENTS = {
    'llava8': ROOT/'outputs/full_videoqa/llava8__five_methods__videomme2700__20261005_r1',
    'qwen32': ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1',
}
PREFIX32 = ROOT/'outputs/full_videoqa/rd12_exp07__qwen32__videomme2700__20260930_r1'
HISTORY = {
    'A': ROOT/'outputs/diagnostics/rd12_exp20__qwen32__dev200__20261003_r1',
    'B': ROOT/'outputs/diagnostics/rd12_exp04__qwen__dev200__k32__20260928_r1',
}
DEV = ROOT/'data/manifests/video_mme_development_200_20260920.json'
PYTHONS = {'llava8': '/App/conda/envs/explore_video_qa_lmms/bin/python',
           'qwen32': '/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'}
CORE_FILES = (
    'src/videoqa_generalization/component_ablations.py',
    'src/videoqa_generalization/component_ablation_runtime.py',
    'src/videoqa_generalization/component_ablation_report.py',
    'scripts/run_rd_component_ablations.py',
    'src/videoqa_methods/budget_contract.py', 'src/videoqa_methods/coarse_eligibility.py',
    'src/videoqa_methods/coarse_only_ablation.py', 'src/videoqa_methods/prefix_coverage_budget.py',
    'src/videoqa_methods/relevance_coverage_budget.py', 'src/videoqa_methods/weighted_seed_support.py',
    'src/videoqa_methods/density_selection.py', 'src/videoqa_audit/density_reward_decay.py',
    'configs/retrieval_density_v1.json', 'configs/method_versions.json',
)


def budget(profile):
    """接口：固定配置名→唯一预算K，不暴露独立种子数量或可搜索参数。"""
    if profile not in PROFILES:
        raise ValueError('Unknown fixed ablation profile')
    return 8 if profile == 'llava8' else 32


def source_identity(selection):
    """接口：候选编号可变，跨池比较只用源时钟和有序源帧身份。"""
    video = selection['video']
    return dict(video={k: video[k] for k in ('path', 'start_pts', 'time_base', 'duration_fraction')},
                frames=[[f['source_frame_index'], f['source_pts']] for f in selection['selected_frames']])


def freeze(path, value):
    """接口：首次原子发布；恢复仅接受完全相同的冻结协议或选择。"""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Frozen ablation record changed: ' + str(path))
    else:
        durable(path, value)


def cut_prefixes(prefix, quotas, target):
    """接口：固定父前缀和真实配额→源索引，拒绝零配额、超容量和预算不闭合。"""
    # 步骤1校验每区保护下限与容量；步骤2只截取前缀，不采用第一遍具体选帧。
    if set(quotas) != {str(r['region_id']) for r in prefix['regions']}:
        raise ValueError('Quota region identity differs')
    chosen = []
    for region in prefix['regions']:
        q = quotas[str(region['region_id'])]
        if isinstance(q, bool) or not isinstance(q, int) or not 1 <= q <= len(region['prefix']):
            raise ValueError('Invalid real first-pass quota')
        chosen.extend(region['prefix'][:q])
    if len(chosen) != target or len(set(chosen)) != target:
        raise ValueError('Prefix quota budget is not closed')
    return chosen


def select_component(control, pool, features, prefix, frame_budget):
    """接口：冻结池/特征/实际父前缀/K→控制选择和轨迹，不接收答案或历史对错。"""
    if control not in CONTROLS:
        raise ValueError('Unknown fixed ablation control')
    f = np.asarray(features, dtype=np.float32)
    cfg = budget_config(frame_budget)
    # 步骤1 A仅移除追加细帧；复用原实现重建次序与预算，不简单删最终帧。
    started = time.perf_counter()
    if control == 'A':
        selected, trace = select_coarse_only(pool, f, prefix, frame_budget)
        trace.update(control='A', quota_origin='same weighted-coverage DP on coarse-only pool',
                     selection_seconds=time.perf_counter()-started)
        return selected, trace
    # 步骤2 B在全局粗资格单元上重新运行真实第一遍；绝不读reference_quotas占位字段。
    cells = coarse_selection_cells(pool['plan'], pool['video'])
    first_indices, first = select_density_decay(pool['candidates'], f, cells, pool['video'], cfg)
    first_seconds = time.perf_counter()-started
    target = min(frame_budget, len(pool['candidates']))
    selected = cut_prefixes(prefix, first['region_budgets'], target)
    selected.sort(key=lambda i: frame_key(pool['candidates'][i]))
    # 步骤3独立第二遍复算，证明该前缀截取就是同条件exp04策略，不用更弱占位配额。
    second_started = time.perf_counter()
    direct, direct_trace = select_global_coarse(pool['candidates'], f, pool['plan'], pool['video'],
        cfg, first['region_budgets'], pool['initial_scores'])
    if direct != selected:
        raise ValueError('Real first-pass allocation does not reproduce fixed local prefixes')
    if not set(prefix['protected_indices']) <= set(selected):
        raise ValueError('Protected anchor lost')
    trace = dict(control='B', quota_origin='recomputed_select_density_decay_on_global_coarse_cells',
                 region_budgets=first['region_budgets'], first_pass_indices=first_indices,
                 first_pass_trace=first, protected_indices=prefix['protected_indices'],
                 first_pass_seconds=first_seconds, direct_second_pass_seconds=time.perf_counter()-second_started,
                 direct_second_pass_exact=True, local_prefixes_unchanged=True,
                 selection_seconds=time.perf_counter()-started,
                 limitation='Budget-component comparison, not the isolated causal effect of the DP solver')
    return selected, trace


@lru_cache(maxsize=4)
def parent_context(profile):
    """接口：每进程只载入一次父协议，避免逐题重复解析大型来源索引。"""
    parent = PARENTS[profile]
    return dict(spec=read_json(parent/'protocol.json'), protocol_sha256=sha256(parent/'protocol.json'))


def load_assets(profile, question_id):
    """接口：只投影输入资产和父选择，不将任何答案字段传给选择函数。"""
    k = budget(profile); parent = PARENTS[profile]; context = parent_context(profile)
    pp = BASE/'pools'/(question_id+'.json') if k == 32 else parent/'inputs/pools'/(question_id+'.json')
    fp = BASE/'features'/(question_id+'.npy')
    tp = PREFIX32/'traces'/(question_id+'.json') if k == 32 else parent/'inputs/traces'/(question_id+'.json')
    sp = parent/'selections/RD-1.2-exp12'/(question_id+'.json')
    paths = [pp, fp, tp, sp]; sources = {str(p): sha256(p) for p in paths}
    # 步骤1检查原始全量资产检查点绑定；8帧特征必须按原映射取行。
    check = read_json(BASE/'selection_checkpoints'/(question_id+'.json'))
    for p in (BASE/'pools'/(question_id+'.json'), fp):
        expected = check['files'][str(p.relative_to(BASE))]
        actual = sources[str(p)] if str(p) in sources else sha256(p)
        if actual != expected:
            raise ValueError('Original coarse/fine asset fingerprint differs')
    if k == 8:
        for p in (pp, tp, sp):
            expected = context['spec']['selection_exports'][str(p.relative_to(parent))]
            if sources[str(p)] != expected:
                raise ValueError('Frozen K8 asset changed')
    else:
        saved = read_json(sp)
        if saved['trace_sha256'] != sha256(parent/saved['trace_file']):
            raise ValueError('Parent weighted trace fingerprint differs')
    pool = read_json(pp); global_features = np.load(fp, allow_pickle=False)
    features = global_features if k == 32 else global_features[pool['original_candidate_indices']]
    trace = read_json(tp); prefix = trace if k == 32 else trace['exp07']
    original = read_json(sp)['selection']
    return pool, features, prefix, original, sources


def offline_one(job):
    """接口：一题父重放、两控制选择和历史开发核对；无评分或生成调用。"""
    root, profile, row = Path(job[0]), job[1], job[2]
    q = row['question_id']; k = budget(profile); dest = root/'offline'/profile/(q+'.json')
    pool, features, saved_prefix, original, sources = load_assets(profile, q)
    if dest.exists():
        saved = read_json(dest)
        if any(saved['sources'].get(p) != h for p, h in sources.items()):
            raise ValueError('Offline source fingerprint differs')
        for path, expected in saved['sources'].items():
            if sha256(path) != expected:
                raise ValueError('Historical control source changed')
        for path, expected in saved['exports'].items():
            if sha256(root/path) != expected:
                raise ValueError('Offline output changed')
        return saved
    # 步骤1按单预算重建种子/区域，父重放只用可行配额作目标下界验收。
    cfg = budget_config(k)
    if build_regions(pool['candidates'][:pool['initial_count']], pool['initial_scores'], pool['video'], cfg) != pool['plan']:
        raise ValueError('Frozen seed/window plan differs')
    capacities = [len(r['members']) for r in saved_prefix['regions']]
    ref = dict(zip((str(r['region_id']) for r in saved_prefix['regions']),
                   feasible_reference(capacities, min(k, len(pool['candidates'])))))
    parent_started = time.perf_counter()
    _, prefix = select_prefix_coverage(pool['candidates'], features, pool['plan'], pool['video'],
                                     cfg, ref, pool['initial_scores'])
    parent_ids, parent_trace = select_relevance_coverage(pool, features, prefix)
    parent_seconds = time.perf_counter()-parent_started
    for a, b in zip(prefix['regions'], saved_prefix['regions']):
        if a['region_id'] != b['region_id'] or a['prefix'] != b['prefix'] or not np.allclose(
                a['coverage_curve'], b['coverage_curve'], atol=1e-6, rtol=0):
            raise ValueError('Frozen parent prefix/coverage differs')
    replay = dict(video=pool['video'], selected_frames=[pool['candidates'][i] for i in parent_ids])
    if source_identity(replay) != source_identity(original):
        raise ValueError('Frozen exp12 parent selection differs')
    # 步骤2两项均从相同父输入出发；历史200题只核验选择，不据答案调整规则。
    exports = {}; values = {}; historical_checks = {}
    dev_ids = {r['question_id'] for r in read_json(DEV)['rows']}
    for control, method in CONTROLS.items():
        ids, trace = select_component(control, pool, features, prefix, k)
        selection = dict(video=pool['video'], selected_frames=[pool['candidates'][i] for i in ids])
        validate_frames(selection, k)
        if profile == 'qwen32' and q in dev_ids:
            history = HISTORY[control]/'results'/method/(q+'.json')
            expected = read_json(history)['selection']
            if source_identity(selection) != source_identity(expected):
                raise ValueError('Historical development selection mismatch: '+control+'/'+q)
            sources[str(history)] = sha256(history)
            historical_checks[control] = True
        tp = root/'traces'/profile/control/(q+'.json'); durable(tp, trace)
        item = dict(question_id=q, video_id=row['video_id'], profile=profile, method=method, control=control,
                    frame_budget=k, selection=selection, selected_indices=ids,
                    parent_source_equal=source_identity(selection)==source_identity(original),
                    fine_selected=sum(i >= pool['initial_count'] for i in ids),
                    coarse_count=pool['initial_count'], parent_pool_size=len(pool['candidates']),
                    actual_frame_count=len(ids), trace_file=str(tp.relative_to(root)), trace_sha256=sha256(tp))
        sp = root/'selections'/profile/method/(q+'.json'); durable(sp, item)
        exports[str(tp.relative_to(root))] = sha256(tp)
        exports[str(sp.relative_to(root))] = sha256(sp)
        values[control] = dict(selected_sources=source_identity(selection)['frames'],
                               frame_count=len(ids), same_as_parent=item['parent_source_equal'],
                               fine_selected=item['fine_selected'])
    # 步骤3每题原子提交，来源映射只记录，不复制大特征或候选图像。
    record = dict(question_id=q, profile=profile, sources=sources, exports=exports,
                  controls=values, parent_replay_exact=True, parent_selection_seconds=parent_seconds,
                  historical_control_replays=historical_checks, new_model_calls=0)
    durable(dest, record)
    return record


def offline(root, workers=4, controls_only=False):
    """接口：冻结5400父重放与10800控制选择；开发控制先验收再跑全量，不调用模型。"""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    data = rows(); codes = {p: sha256(ROOT/p) for p in CORE_FILES}
    spec = dict(task='rd_component_ablations', methods=CONTROLS, profiles=list(PROFILES),
                frame_budgets={p: budget(p) for p in PROFILES}, question_ids=[r['question_id'] for r in data],
                data_manifest_sha256=sha256(ROOT/'data/manifests/video_mme_full_2700.json'),
                dev_manifest_sha256=sha256(DEV), code_sha256=codes,
                maximum_new_real_qa=10800, maximum_synthetic_generations=21,
                new_blip_calls=0, new_selection_visual_calls=0,
                budget_control='recompute genuine first pass; reference_quotas never used as algorithm')
    with exclusive(root/'offline.lock'):
        freeze(root/'offline_protocol.json', spec)
        if (root/'offline_summary.json').exists():
            old = read_json(root/'offline_summary.json')
            for rel, expected in old['exports'].items():
                if sha256(root/rel) != expected:
                    raise ValueError('Completed offline asset changed')
            return old
        for path in CORE_FILES:
            target = root/'code_snapshot'/path; target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256(target) != codes[path]:
                raise ValueError('Frozen source snapshot differs')
            if not target.exists():shutil.copyfile(ROOT/path, target)
        dev = {r['question_id'] for r in read_json(DEV)['rows']}
        selected = [r for r in data if r['question_id'] in dev]
        # 步骤1旧开发200题两个控制必须严格重放；不是按成绩放行。
        with ProcessPoolExecutor(max_workers=workers) as executor:
            controls = list(executor.map(offline_one, [(str(root), 'qwen32', r) for r in selected]))
        if not all(r['historical_control_replays']==dict(A=True, B=True) for r in controls):
            raise ValueError('Historical 200 control replay incomplete')
        durable(root/'development_control_acceptance.json', dict(status='passed', questions=200,
                control_selections=400, real_first_pass=True, new_model_calls=0))
        if controls_only:
            return dict(status='controls_passed', questions=200)
        # 步骤2相同冻结规则全量离线；每20题记录进度，不按代理指标停止实验。
        records = []; began = time.perf_counter()
        jobs = [(str(root), p, r) for p in PROFILES for r in data]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for record in executor.map(offline_one, jobs):
                records.append(record)
                if len(records) % 20 == 0:
                    elapsed = time.perf_counter()-began
                    durable(root/'offline_status.json', dict(status='running', completed=len(records), total=5400,
                        elapsed_seconds=elapsed, eta_seconds=elapsed/max(1,len(records))*(5400-len(records)), utc=utc()))
        exports = {p: h for r in records for p, h in r['exports'].items()}
        counts = {p: {c: dict(Counter(str(r['controls'][c]['frame_count']) for r in records if r['profile']==p))
                      for c in CONTROLS} for p in PROFILES}
        summary = dict(status='passed', parent_replays=5400, control_selections=10800,
                       exports=exports, actual_frame_counts=counts, seconds=time.perf_counter()-began,
                       new_model_calls=0, protocol_sha256=sha256(root/'offline_protocol.json'))
        durable(root/'offline_summary.json', summary)
        durable(root/'offline_status.json', dict(status='completed', completed=5400, total=5400, utc=utc()))
        return summary
