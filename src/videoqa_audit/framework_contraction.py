"""RD内部设计的只读审计：预算等价、种子连接、窗口传播及来源身份。"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
import os
import time

import numpy as np
from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, digest, exclusive, rows, utc
from videoqa_methods.budget_contract import budget_config, select_exp07
from videoqa_methods.density_selection import build_regions, source_time, refinement_targets

BASE = ROOT / 'outputs/full_videoqa/uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01'
RUN8 = ROOT / 'outputs/full_videoqa/llava8__five_methods__videomme2700__20261005_r1'
POOL16 = ROOT / 'outputs/full_videoqa/exp12_generalization__videomme2700__20261005_r1'
RUN16 = ROOT / 'outputs/full_videoqa/exp07_budget16__videomme2700__20261005_r1'
RUN32 = ROOT / 'outputs/full_videoqa/rd12_exp07__qwen32__videomme2700__20260930_r1'
METHOD = 'RD-1.2-exp07'
CASES = ('103-1', '383-2', '395-2', '409-3', '035-2', '844-1')
CODE = ('src/videoqa_methods/budget_contract.py', 'src/videoqa_audit/framework_contraction.py',
        'scripts/audit_framework_contraction.py', 'src/videoqa_methods/density_selection.py',
        'src/videoqa_methods/coarse_eligibility.py', 'src/videoqa_methods/weighted_seed_support.py',
        'src/videoqa_methods/prefix_coverage_budget.py', 'src/videoqa_methods/density_pipeline.py',
        'src/videoqa_methods/retarget_seed_pool.py', 'configs/retrieval_density_v1.json')


def merge_intervals(intervals):
    """接口：合并闭区间及共享端点；仅用于等价推导，不改原定位计划。"""
    result = []
    for left, right in sorted(intervals):
        if left > right:
            raise ValueError('Invalid closed interval')
        if result and left <= result[-1][1]:
            result[-1][1] = max(result[-1][1], right)
        else:
            result.append([left, right])
    return result


def inspect_regions(pool, selected, trace, frame_budget):
    """接口：只读取计划与选择，派生连接数、触发差异和补查传播；不生成候选。"""
    candidates, plan, video = pool['candidates'], pool['plan'], pool['video']
    cfg = budget_config(frame_budget)
    radius, end = Fraction(cfg['window_radius_seconds']), Fraction(video['duration_fraction'])
    cores = {r['core_id']: r for r in plan['cores']}
    # 步骤1：独立计算每种子的窗口并集，检查直接并集的适用前提。
    direct = merge_intervals([(max(Fraction(0), source_time(candidates[i], video) - radius),
                               min(end, source_time(candidates[i], video) + radius))
                              for i in plan['seed_indices']])
    original = [[Fraction(r['left_fraction']), Fraction(r['right_fraction'])] for r in plan['regions']]
    max_gap = max((source_time(candidates[b], video) - source_time(candidates[a], video)
                   for c in cores.values() for a, b in zip(c['seed_indices'], c['seed_indices'][1:])),
                  default=Fraction(0))
    records = []
    # 步骤2：核心统计有代数等价式；触发要求连续两条连接，不等于总连接数阈值。
    for r in plan['regions']:
        group = [cores[i] for i in r['core_ids']]
        sizes = [len(c['seed_indices']) for c in group]
        edges = sum(n - 1 for n in sizes)
        if r['weight'] != 1 + edges or r['refine'] != (max(sizes) >= 3):
            raise ValueError('Derived density or trigger identity differs')
        hot = merge_intervals([(Fraction(c['left_fraction']), Fraction(c['right_fraction']))
                               for c in group if c['seed_count'] >= 3])
        inside_hot = lambda t: any(a <= t <= b for a, b in hot)
        targets = refinement_targets(r, video, cfg) if r['refine'] else []
        outside_targets = [str(t) for t in targets if not inside_hot(t)]
        fine = [i for i in range(pool['initial_count'], len(candidates))
                if Fraction(r['left_fraction']) <= source_time(candidates[i], video) <= Fraction(r['right_fraction'])]
        extra_fine = [i for i in fine if not inside_hot(source_time(candidates[i], video))]
        rr = next(x for x in trace['regions'] if x['region_id'] == r['region_id'])
        selected_fine = [i for i in selected if i in set(fine)]
        records.append(dict(region_id=r['region_id'], core_sizes=sizes, seed_count=r['m'],
            core_count=r['c'], connection_count=edges, density_weight=r['weight'], refine=r['refine'],
            density_threshold_would_differ=(r['weight'] >= 3) != r['refine'],
            mixed_refinement=r['refine'] and any(n < 3 for n in sizes),
            targets=len(targets), targets_outside_triggering_cores=len(outside_targets),
            outside_target_fractions=outside_targets, fine_count=len(fine), fine_outside_hot=len(extra_fine),
            selected_fine=len(selected_fine), selected_fine_outside_hot=len(set(selected_fine) & set(extra_fine)),
            selected_outside_hot_indices=sorted(set(selected_fine) & set(extra_fine)),
            quota=trace['region_budgets'][str(r['region_id'])],
            coarse_reference_count=len(rr['reference_coarse_indices']),
            anchor_index=r['anchor_index'], window=[r['left_fraction'], r['right_fraction']]))
    return dict(regions=records, direct_seed_union_equal=direct == original,
                source_gap_condition=max_gap <= 2 * radius, maximum_core_source_gap=str(max_gap))


def locations(q, k):
    """接口：K定位唯一冻结来源；K16跨模型共享，只执行一次选择重放。"""
    feature = BASE / 'features' / (q + '.npy')
    if k == 8:
        return RUN8 / 'inputs/pools' / (q + '.json'), feature, RUN8 / 'inputs/traces' / (q + '.json'), RUN8 / 'selections' / METHOD / (q + '.json')
    if k == 16:
        return POOL16 / 'inputs/pools' / (q + '.json'), feature, RUN16 / 'inputs/traces' / (q + '.json'), RUN16 / 'inputs/selections' / (q + '.json')
    if k == 32:
        return BASE / 'pools' / (q + '.json'), feature, RUN32 / 'traces' / (q + '.json'), RUN32 / 'selections' / METHOD / (q + '.json')
    raise ValueError('Only frozen K8/K16/K32 inputs are audited')


def check_binding(path, checkpoint, root):
    """接口：把新读取字节绑定到旧检查点，而不是只把当前哈希当作验收结论。"""
    saved = read_json(checkpoint)
    expected = saved['files'].get(str(path.relative_to(root)))
    if expected is None or sha256(path) != expected:
        raise ValueError('Historical checkpoint binding differs: ' + str(path))


def replay_one(job):
    """接口：单题单K只读重放，结果原子保存；恢复验证输入和输出指纹后跳过计算。"""
    root, q, k = Path(job[0]), job[1], job[2]
    pp, fp, tp, sp = locations(q, k)
    cp32 = BASE / 'selection_checkpoints' / (q + '.json')
    source_paths = [pp, fp, tp, sp, cp32]
    if k == 8:
        source_paths.append(RUN8 / 'inputs/checkpoints' / (q + '.json'))
    elif k == 16:
        source_paths.extend([POOL16 / 'inputs/checkpoints' / (q + '.json'), RUN16 / 'inputs/checkpoints' / (q + '.json')])
        source_paths.extend(RUN16 / stage / 'selections' / METHOD / (q + '.json') for stage in ('qwen16', 'llava16'))
    sources = {str(p): sha256(p) for p in source_paths}
    dest = root / 'records' / ('K' + str(k)) / (q + '.json')
    if dest.exists():
        old = read_json(dest)
        payload = {key: value for key, value in old.items() if key != 'record_sha256'}
        if old['sources'] != sources or old['record_sha256'] != digest(payload):
            raise ValueError('Completed audit source or output differs')
        return old
    started = time.perf_counter()
    # 步骤1：验证旧资产绑定与特征行映射，禁止补算缺失评分或特征。
    check_binding(fp, cp32, BASE)
    if k == 32:
        check_binding(pp, cp32, BASE)
        if sha256(tp) != read_json(sp)['trace_sha256']:
            raise ValueError('Frozen K32 trace differs')
    elif k == 8:
        cp = RUN8 / 'inputs/checkpoints' / (q + '.json')
        for p in (pp, tp, sp): check_binding(p, cp, RUN8)
    else:
        check_binding(pp, POOL16 / 'inputs/checkpoints' / (q + '.json'), POOL16)
        for p in (tp, sp): check_binding(p, RUN16 / 'inputs/checkpoints' / (q + '.json'), RUN16)
    pool, old_trace, old_selection = read_json(pp), read_json(tp), read_json(sp)
    if k == 8: old_trace = old_trace['exp07']
    f32 = np.load(fp, allow_pickle=False)
    features = f32 if k == 32 else f32[pool['original_candidate_indices']]
    # 步骤2：只有K的适配重建种子/窗口并执行原前缀及DP；数值容差不放宽离散一致性。
    selected, trace = select_exp07(pool, features, k)
    if trace['region_budgets'] != old_trace['region_budgets']:
        raise ValueError('Frozen quota mismatch: ' + q)
    error = 0.0
    for current, old in zip(trace['regions'], old_trace['regions']):
        if any(current[key] != old[key] for key in ('region_id', 'prefix', 'members', 'reference_coarse_indices', 'density_weight')):
            raise ValueError('Frozen prefix identity mismatch: ' + q)
        error = max(error, float(np.max(np.abs(np.asarray(current['coverage_curve']) - old['coverage_curve']))))
    if len(trace['regions']) != len(old_trace['regions']) or error > 1e-6:
        raise ValueError('Frozen coverage mismatch: ' + q)
    frames = [pool['candidates'][i] for i in selected]
    if frames != old_selection['selection']['selected_frames']:
        raise ValueError('Frozen ordered source frame mismatch: ' + q)
    if k == 16:
        for stage in ('qwen16', 'llava16'):
            item = read_json(RUN16 / stage / 'selections' / METHOD / (q + '.json'))
            if item['selection'] != old_selection['selection'] or item['rgb_sha256'] != old_selection['rgb_sha256']:
                raise ValueError('Shared K16 model input differs')
    # 步骤3：派生结构审计，只保存源身份和统计，不复制视频、特征或原轨迹。
    structural = inspect_regions(pool, selected, trace, k)
    record = dict(question_id=q, frame_budget=k, seed_count=len(pool['plan']['seed_indices']),
        actual_frame_count=len(selected), selected_indices=selected,
        selected_source_identity=[dict(source_pts=f['source_pts'], source_frame_index=f['source_frame_index']) for f in frames],
        structural=structural, maximum_curve_error=error, discrete_replay_exact=True,
        shared_k16_source_rgb_exact=k == 16, sources=sources,
        new_model_calls=0, seconds=time.perf_counter() - started)
    record['record_sha256'] = digest(record)
    durable(dest, record)
    return record


def prepare(root):
    """接口：冻结清单、源码及旧运行入口身份；不读取模型权重或生成真实答案。"""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    dataset = rows()
    code = {str(ROOT / p): sha256(ROOT / p) for p in CODE}
    # 步骤1：方法发布清单核对冻结函数，新增适配独立记录，不修改原登记。
    release_path = ROOT / 'configs/releases/rd_1_2_exp07_qwen32_dev200_20260928.json'
    release = read_json(release_path)
    for name, h in release['method_code_sha256'].items():
        if sha256(ROOT / name) != h: raise ValueError('Released method changed: ' + name)
    protocol = root / 'protocol.json'
    spec = dict(method=METHOD, budgets=[8, 16, 32], question_ids=[r['question_id'] for r in dataset],
        code_sha256=code, manifest_sha256=sha256(ROOT / 'data/manifests/video_mme_full_2700.json'),
        release_sha256=sha256(release_path), expected_replays=8100, tolerance=1e-6,
        allowed_model_calls=0, representative_questions=list(CASES))
    # 步骤2：恢复沿用同一协议；源码改变或清单改变必须暂停，不覆盖既有结果。
    if protocol.exists():
        old = read_json(protocol)
        if {k: v for k, v in old.items() if k != 'created_utc'} != spec:
            raise ValueError('Audit protocol changed')
        return old
    spec['created_utc'] = utc(); durable(protocol, spec)
    snapshot = root / 'code_snapshot'; snapshot.mkdir(exist_ok=True)
    import shutil
    for name in CODE:
        p = snapshot / name; p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, p)
    return spec


def summarize(root, records):
    """接口：汇总三预算重放及两个主要工作点的结构，保留全题分母和来源。"""
    by_budget = {}
    for k in (8, 16, 32):
        data = [x for x in records if x['frame_budget'] == k]
        if len(data) != 2700 or len({x['question_id'] for x in data}) != 2700:
            raise ValueError('Incomplete replay population')
        regions = [r for x in data for r in x['structural']['regions']]
        by_budget[str(k)] = dict(questions=len(data), regions=len(regions), refined_regions=sum(r['refine'] for r in regions),
            mixed_refinement_regions=sum(r['mixed_refinement'] for r in regions),
            total_targets=sum(r['targets'] for r in regions),
            targets_outside_triggering_cores=sum(r['targets_outside_triggering_cores'] for r in regions),
            fine_frames=sum(r['fine_count'] for r in regions), fine_outside_hot=sum(r['fine_outside_hot'] for r in regions),
            selected_fine=sum(r['selected_fine'] for r in regions),
            selected_fine_outside_hot=sum(r['selected_fine_outside_hot'] for r in regions),
            questions_with_selected_fine_outside_hot=sum(any(r['selected_fine_outside_hot'] for r in x['structural']['regions']) for x in data),
            weight_threshold_trigger_mismatches=sum(r['density_threshold_would_differ'] for r in regions),
            direct_seed_union_exact=sum(x['structural']['direct_seed_union_equal'] for x in data),
            source_gap_condition_passed=sum(x['structural']['source_gap_condition'] for x in data),
            actual_frame_counts=dict(Counter(str(x['actual_frame_count']) for x in data)),
            maximum_curve_error=max(x['maximum_curve_error'] for x in data))
    # 步骤1：案例全部保留，不依据得失挑选；来源图板由已有审计链接提供。
    cases = {q: {str(k): next(x for x in records if x['question_id'] == q and x['frame_budget'] == k)
                 for k in (8, 32)} for q in CASES}
    durable(root / 'case_structures.json', cases)
    result = dict(status='passed', method=METHOD, replays=len(records), by_budget=by_budget,
                  shared_k16_models=['Qwen2.5-VL-7B-Instruct', 'LLaVA-Video-7B-Qwen2'],
                  calls=dict(blip=0, visual_encoder=0, text_generation=0, final_qa=0),
                  artifacts={str(root / 'records' / ('K' + str(x['frame_budget'])) / (x['question_id'] + '.json')):
                             sha256(root / 'records' / ('K' + str(x['frame_budget'])) / (x['question_id'] + '.json')) for x in records})
    # 步骤2：汇总只描述结构影响，帧被范围排除不自动等于语义证据损失。
    durable(root / 'summary.json', result)
    return result


def execute(root, workers=8):
    """接口：CPU后台8100项重放，逐条恢复并核对来源；无GPU初始化或模型调用。"""
    root = Path(root); spec = prepare(root)
    started = time.perf_counter()
    with exclusive(root / 'run.lock'):
        # 步骤1：完成态只核验协议、旧来源和新记录，禁止再次执行选择。
        if (root / 'summary.json').exists():
            saved = read_json(root / 'summary.json')
            for path, h in saved['artifacts'].items():
                if sha256(path) != h: raise ValueError('Completed output changed')
                for source, sh in read_json(path)['sources'].items():
                    if sha256(source) != sh: raise ValueError('Completed source changed')
            durable(root / 'completion_audit.json', dict(status='passed', records=8100, new_model_calls=0,
                                                        additional_selection_replays=0, utc=utc()))
            return saved
        jobs = [(str(root), q, k) for k in spec['budgets'] for q in spec['question_ids']]
        records = []
        # 步骤2：有界CPU并发，BLAS线程由入口固定，长任务脱离SSH独立推进。
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for record in executor.map(replay_one, jobs, chunksize=1):
                records.append(record)
                if len(records) % 20 == 0:
                    elapsed = time.perf_counter() - started
                    durable(root / 'status.json', dict(completed=len(records), total=8100, pid=os.getpid(),
                        sid=os.getsid(0), elapsed_seconds=elapsed, eta_seconds=elapsed / len(records) * (8100-len(records)), utc=utc()))
        # 步骤3：全量完整才落通过状态；异常交由入口保存，旧产物始终只读。
        result = summarize(root, records)
        durable(root / 'execution.json', dict(seconds=time.perf_counter()-started, workers=workers, pid=os.getpid(),
                                             sid=os.getsid(0), new_model_calls=0, utc=utc()))
        durable(root / 'status.json', dict(status='completed', completed=8100, total=8100, utc=utc()))
        return result
