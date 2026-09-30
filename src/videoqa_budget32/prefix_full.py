"""exp07全量适配：冻结算法不变，200条历史来源与2500次新调用严格隔离。"""
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, rows, exclusive, utc, digest, pilot_ids
from videoqa_runtime.baseline_report import wilson, distribution
from videoqa_full.report import compare
from .frames import FrameProvider, validate_frames
from .run import verify as verify_runtime, validate, recover_call, PYTHON
from .development import worker

METHOD = 'RD-1.2-exp07'
BASE = ROOT/'outputs/full_videoqa/uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01'
DEV = ROOT/'outputs/diagnostics/rd12_exp07__qwen__dev200__k32__20260928_r1'
RELEASE = ROOT/'configs/releases/rd_1_2_exp07_qwen32_dev200_20260928.json'
DEV_MANIFEST = ROOT/'data/manifests/video_mme_development_200_20260920.json'


def partition(data, dev_ids):
    """接口：按冻结题号分开历史导入与新增调用，不依据预测筛题。"""
    ids = [r['question_id'] for r in data]
    if len(ids) != len(set(ids)) or len(dev_ids) != len(set(dev_ids)) or not set(dev_ids) <= set(ids):
        raise ValueError('Invalid dataset partition')
    return [q for q in ids if q in set(dev_ids)], [q for q in ids if q not in set(dev_ids)]


def reference_allocation(counts, budget):
    """接口：只为冻结函数提供可行参照配额；它不参与DP生成，不依赖exp04资产。"""
    if not counts or min(counts) < 1 or not len(counts) <= budget <= sum(counts):
        raise ValueError('Infeasible reference budget')
    values = [1]*len(counts)
    remaining = budget-len(counts)
    for i, cap in enumerate(counts):
        added = min(remaining, cap-1); values[i] += added; remaining -= added
    assert remaining == 0
    return values


def release_check():
    """接口：核对已批准候选及开发来源；不接受算法或名单静默变化。"""
    release = read_json(RELEASE)
    assert sha256(DEV/'protocol.json') == release['protocol_sha256']
    assert sha256(DEV_MANIFEST) == release['dataset_manifest_sha256']
    for name, h in release['method_code_sha256'].items():
        assert sha256(ROOT/name) == h, name
    assert sha256(ROOT/release['frozen_artifacts']) == release['frozen_artifacts_sha256']
    return release


def select_one(job):
    """接口：复用完整粗细池执行冻结选择，验证源帧和历史200题离散一致性。"""
    import numpy as np
    from fractions import Fraction
    from videoqa_methods.coarse_eligibility import coarse_selection_cells
    from videoqa_methods.density_selection import source_time
    from videoqa_methods.prefix_coverage_budget import select_prefix_coverage
    run, row, imported = job; q = row['question_id']; started = time.perf_counter()
    pp, fp, cp = BASE/'pools'/(q+'.json'), BASE/'features'/(q+'.npy'), BASE/'selection_checkpoints'/(q+'.json')
    paths = [pp, fp, cp, RELEASE]
    sources = {str(p):sha256(p) for p in paths}
    dest = run/'selections'/METHOD/(q+'.json')
    if dest.exists():
        saved = read_json(dest)
        assert saved['sources'] == sources and sha256(run/saved['trace_file']) == saved['trace_sha256']
        return saved
    # 步骤1：原运行检查点绑定池和特征，不能把其他运行的细帧混入。
    check = read_json(cp)
    for p in (pp, fp): assert sources[str(p)] == check['files'][str(p.relative_to(BASE))]
    pool = read_json(pp); f = np.load(fp, allow_pickle=False)
    cells = coarse_selection_cells(pool['plan'], pool['video'])
    counts = [sum(Fraction(r['left_fraction']) <= source_time(c,pool['video']) <= Fraction(r['right_fraction']) for c in pool['candidates']) for r in cells['regions']]
    target = min(32,len(pool['candidates']))
    quotas = {str(r['region_id']):n for r,n in zip(cells['regions'],reference_allocation(counts,target))}
    # 步骤2：参照配额仅用于下界检查。最终DP和局部序列仍来自原冻结函数。
    ids, trace = select_prefix_coverage(pool['candidates'],f,pool['plan'],pool['video'],dict(frame_budget=32),quotas,pool['initial_scores'])
    selection = dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in ids])
    validate_frames(selection)
    images, hashes = FrameProvider(selection).decode()
    for im in images: im.close()
    if imported:
        old = read_json(DEV/'selections'/METHOD/(q+'.json'))
        assert old['selection'] == selection and old['rgb_sha256'] == hashes
    tp = run/'traces'/(q+'.json'); durable(tp,trace)
    item = dict(question_id=q,video_id=row['video_id'],method=METHOD,selection=selection,rgb_sha256=hashes,
        sources=sources,selected_indices=ids,trace_file=str(tp.relative_to(run)),trace_sha256=sha256(tp),
        imported_answer=imported,selection_seconds=time.perf_counter()-started,new_model_calls=0)
    durable(dest,item); return item


def selection_all(run):
    """接口：全部2700题零模型重放，冻结200题复用身份，失败时停止而不重建资产。"""
    run = Path(run); run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'selection.lock'):
        release_check(); data = rows(); dev = [r['question_id'] for r in read_json(DEV_MANIFEST)['rows']]
        imported, new = partition(data,dev); assert (len(imported),len(new)) == (200,2500)
        if (run/'selection_summary.json').exists():
            existing = read_json(run/'selection_summary.json')
            assert existing['status'] == 'passed' and existing['release_sha256'] == sha256(RELEASE)
            assert existing['imported_ids'] == imported and existing['new_ids'] == new and len(existing['exports']) == 2700
            for rel,h in existing['exports'].items(): assert sha256(run/rel) == h
            return
        assert shutil.disk_usage(run).free > 30*1024**3
        started = time.perf_counter(); exports = {}
        # 步骤1：有界四线程CPU重放与精确解码；不占用BLIP或问答GPU。
        with ThreadPoolExecutor(max_workers=4) as executor:
            for item in executor.map(select_one,[(run,r,r['question_id'] in set(imported)) for r in data]):
                q = item['question_id']; rel = f'selections/{METHOD}/{q}.json'; exports[rel] = sha256(run/rel)
                durable(run/'selection_status.json',dict(completed=len(exports),total=2700,elapsed_seconds=time.perf_counter()-started,utc=utc()))
        # 步骤2：完整清单落盘后方可进入生成准备阶段。
        durable(run/'selection_summary.json',dict(status='passed',exports=exports,imported_ids=imported,new_ids=new,
            new_model_calls=0,wall_seconds=time.perf_counter()-started,release_sha256=sha256(RELEASE)))


def prepare(run, allowed):
    """接口：冻结新协议；历史记录原字节导入独立目录，不伪装成本轮新生成。"""
    from .run import critical_code
    from videoqa_full.prepare import environment
    from videoqa_lmms_bridge.common import framework_identity
    run = Path(run)
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists(): return verify(run)
        release_check(); base = verify_runtime(BASE); devspec = verify_runtime(DEV)
        sel = read_json(run/'selection_summary.json'); syn = read_json(run/'synthetic_acceptance.json')
        assert sel['status'] == syn['status'] == 'passed'
        assert syn['code_sha256'] == critical_code() and syn['environment'] == environment(PYTHON)
        assert syn['framework'] == framework_identity()
        assert allowed and len(set(allowed)) == len(allowed) and set(allowed) <= set(map(str,range(8)))
        for key in ('model','profile','framework','environment','model_files'):
            assert base[key] == devspec[key], key
        # 步骤1：重新核对媒体与权重字节；耗时不混入QA性能。
        for p,x in base['model_files'].items(): assert sha256(p) == x['sha256'],p
        for v,x in base['video_stats'].items(): assert sha256(x['path']) == base['video_hashes'][v],v
        sources = {}; imported = {}
        for rel,h in sel['exports'].items():
            assert sha256(run/rel) == h
            item = read_json(run/rel)
            for p,ph in item['sources'].items(): assert sha256(p) == ph; sources[p] = ph
            sources[str(run/item['trace_file'])] = item['trace_sha256']
        data = rows(); byid = {r['question_id']:r for r in data}
        frozen_dev = read_json(DEV/'frozen_artifacts.json')
        # 步骤2：使用原协议校验历史调用；导入副本保留原protocol和call ID。
        for q in sel['imported_ids']:
            old = validate(DEV,byid[q],METHOD,devspec)
            current = read_json(run/'selections'/METHOD/(q+'.json'))
            assert old['selection'] == current['selection'] and old['rgb_sha256'] == current['rgb_sha256']
            paths = {}
            for folder in ('results','calls','raw_generations'):
                p = DEV/folder/METHOD/(q+'.json'); dest = run/'imported'/folder/METHOD/(q+'.json')
                assert sha256(p) == frozen_dev[str(p.relative_to(ROOT))], p
                dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
                paths[str(dest.relative_to(run))] = sha256(p); sources[str(p)] = sha256(p)
            imported[q] = dict(origin=str(DEV),files=paths,protocol_sha256=sha256(DEV/'protocol.json'))
        assert sum(read_json(DEV/'results'/METHOD/(q+'.json'))['correct'] for q in sel['imported_ids']) == 136
        for m in ('BASE-Uniform','BASE-BLIP-TopK','RD-1.2'):
            for row in data:
                p = BASE/'results'/m/(row['question_id']+'.json'); sources[str(p)] = sha256(p)
        # 步骤3：新增适配层及固定算法单独快照，不改历史协议。
        code = dict(devspec['code_sha256'])
        for n in ['src/videoqa_budget32/prefix_full.py','scripts/run_prefix_coverage_full.py','scripts/render_prefix_full_cases.py'] + list(read_json(RELEASE)['method_code_sha256']):
            code[n] = sha256(ROOT/n)
        for n,h in code.items():
            assert sha256(ROOT/n) == h
            dest = run/'code_snapshot'/n; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/n,dest)
        spec = dict(base,run_id=run.name,methods=[METHOD],question_count=2700,max_calls=2500,max_calls_per_method=2500,
            question_ids=[r['question_id'] for r in data],imported_ids=sel['imported_ids'],new_ids=sel['new_ids'],
            imports=imported,allowed_gpus=allowed,selection_exports=sel['exports'],code_sha256=code,input_sources=sources,
            source_run=str(BASE),development_run=str(DEV),release_sha256=sha256(RELEASE),
            synthetic_acceptance_sha256=sha256(run/'synthetic_acceptance.json'),pilot_question_ids=pilot_ids(data),
            configuration='RD-1.2-exp07 M32/K32; 200 imported + 2500 new',utc=utc())
        durable(run/'protocol.json',spec); return spec


def verify(run):
    """接口：启动与恢复校验所有选择、来源、代码及导入字节，禁止静默替换。"""
    run = Path(run); spec = verify_runtime(run); release_check()
    assert spec['max_calls'] == 2500 and spec['question_count'] == 2700 and spec['methods'] == [METHOD]
    imp,new = partition(rows(),spec['imported_ids'])
    assert imp == spec['imported_ids'] and new == spec['new_ids'] and len(imp) == 200
    assert sha256(RELEASE) == spec['release_sha256']
    assert sha256(run/'synthetic_acceptance.json') == spec['synthetic_acceptance_sha256']
    for p,h in spec['input_sources'].items(): assert sha256(p) == h,p
    for rel,h in spec['selection_exports'].items(): assert sha256(run/rel) == h,rel
    for item in spec['imports'].values():
        for rel,h in item['files'].items(): assert sha256(run/rel) == h,rel
    return spec


def execute(run):
    """接口：后台协调2500题，一题至多一次生成；导入题不进入工作队列。"""
    from videoqa_runtime.baseline_run import available_gpus
    run = Path(run)
    with exclusive(run/'coordinator.lock'):
        spec = verify(run); byid = {r['question_id']:r for r in rows()}; new = set(spec['new_ids'])
        done = set(); pinned = {}; active = {}; gpus = []; failure = None; phase = 'checking'; start = time.perf_counter()
        def publish():
            elapsed = time.perf_counter()-start
            durable(run/'status.json',dict(status=phase,completed_new=len(done),total_new=2500,imported=200,
                completed_total=200+len(done),pid=os.getpid(),session_id=os.getsid(0),active=active,gpus=gpus,
                elapsed_seconds=elapsed,eta_seconds=(2500-len(done))*elapsed/len(done) if done else None,failure=failure,utc=utc()))
        # 步骤1：安全恢复；已生成但未完成后处理的任务固定原GPU，不允许再次生成。
        calls = list((run/'calls'/METHOD).glob('*.json')); assert len(calls) <= 2500
        for p in calls:
            assert p.stem in new
            recover_call(run,byid[p.stem],METHOD,spec)
            c = read_json(p); assert c['status'] == 'completed'; pinned[p.stem] = c['gpu']
        for p in (run/'results'/METHOD).glob('*.json'):
            assert p.stem in new; validate(run,byid[p.stem],METHOD,spec); done.add(p.stem)
        if done == new:
            if not (run/'execution.json').exists():
                durable(run/'execution.json',dict(status='completed',new_answer_calls=2500,
                    actual_gpus=sorted(set(pinned.values())),wall_seconds=None,
                    recovered_completion=True,limitation='Original coordinator wall time unavailable',utc=utc()))
            return
        required = {g for q,g in pinned.items() if q not in done}
        while True:
            try: gpus,states = available_gpus(spec['allowed_gpus'])
            except RuntimeError: gpus,states = [],{}
            if gpus and required <= set(gpus): break
            phase = 'waiting_resources'; publish(); time.sleep(15)
        for folder in ('logs','workers','failures'): (run/folder).mkdir(exist_ok=True)
        durable(run/'gpu_preflight.json',dict(actual=gpus,states=states,utc=utc()))
        ctx = mp.get_context('spawn'); jobs = {g:ctx.Queue() for g in gpus}; events = ctx.Queue(); stop = ctx.Event()
        children = {}; ready = set(); busy = set(); pending = []
        pilot = [q for q in spec['pilot_question_ids'] if q in new]
        def schedule():
            nonlocal phase,pending
            if any(q not in done for q in pilot): phase = 'pilot'; pending = [q for q in pilot if q not in done]
            else:
                for q in pilot: validate(run,byid[q],METHOD,spec)
                durable(run/'pilot_acceptance.json',dict(status='passed',new_ids=pilot,imported_ids=[q for q in spec['pilot_question_ids'] if q not in new],accuracy_not_gate=True))
                phase = 'running'
                pending = sorted(new-done,key=lambda q:(-read_json(run/'selections'/METHOD/(q+'.json'))['selection']['video']['duration_seconds'],q))
        def refill():
            for gpu in gpus:
                if gpu in busy: continue
                q = next((q for q in pending if q not in pinned or pinned[q] == gpu),None)
                if q is None: continue
                pending.remove(q); busy.add(gpu); jobs[gpu].put(byid[q])
        # 步骤2：复用既有单方法工作进程；阶段放行与完成检查不依赖交互会话。
        try:
            phase = 'loading'
            for gpu in gpus:
                child = ctx.Process(target=worker,args=(str(run),gpu,jobs[gpu],events,stop)); child.start(); children[gpu] = child
            publish()
            while len(done) < 2500:
                try: event = events.get(timeout=5)
                except queue.Empty: event = None
                if event:
                    if event['type'] == 'failed': raise RuntimeError(str(event))
                    if event['type'] == 'ready':
                        ready.add(event['gpu'])
                        if len(ready) == len(gpus): schedule(); refill()
                    elif event['type'] == 'active': active[event['gpu']] = event['question_id']
                    elif event['type'] == 'completed':
                        done.add(event['question_id']); busy.remove(event['gpu']); active.pop(event['gpu'],None)
                        if phase == 'pilot' and not pending and not busy: schedule()
                        refill()
                for gpu,child in children.items():
                    if child.exitcode is not None: raise RuntimeError(f'Worker {gpu} exited {child.exitcode}')
                publish()
        except BaseException as exc:
            failure = repr(exc); stop.set(); phase = 'paused'; publish(); raise
        finally:
            for channel in jobs.values(): channel.put(None)
            while any(p.is_alive() for p in children.values()):
                for child in children.values(): child.join(timeout=1)
                publish()
        phase = 'completed'; publish()
        durable(run/'execution.json',dict(status='completed',new_answer_calls=2500,actual_gpus=gpus,wall_seconds=time.perf_counter()-start,utc=utc()))


def summarize(run):
    """接口：合并有来源的200条与2500条新答案，主计分复算并输出所有配对得失。"""
    from collections import Counter
    run = Path(run); spec = verify(run); data = rows(); records = {}; current = {}
    for folder in ('results','calls','raw_generations'):
        assert {p.stem for p in (run/folder/METHOD).glob('*.json')} == set(spec['new_ids'])
    devspec = read_json(DEV/'protocol.json')
    # 步骤1：使用原协议分别核验导入与新生成，计分用原answer_index再次独立核对。
    for row in data:
        q = row['question_id']
        item = validate(DEV if q in spec['imported_ids'] else run,row,METHOD,devspec if q in spec['imported_ids'] else spec)
        assert item['correct'] == (item['answer']['parsed_answer'] == 'ABCD'[row['answer_index']])
        current[q] = item
    records[METHOD] = current
    for m in ('BASE-Uniform','BASE-BLIP-TopK','RD-1.2'):
        records[m] = {r['question_id']:read_json(BASE/'results'/m/(r['question_id']+'.json')) for r in data}
        for row in data:
            x = records[m][row['question_id']]
            assert x['correct'] == (x['answer']['parsed_answer'] == 'ABCD'[row['answer_index']])
    devq = set(spec['imported_ids']); devv = {r['video_id'] for r in data if r['question_id'] in devq}
    shared = [r for r in data if all(len(g[r['question_id']]['selection']['selected_frames']) == 32 for g in records.values())]
    subsets = dict(all2700=data,development200=[r for r in data if r['question_id'] in devq],other2500=[r for r in data if r['question_id'] not in devq],outside_development_videos2100=[r for r in data if r['video_id'] not in devv],common32=shared)
    assert [len(subsets[k]) for k in list(subsets)[:4]] == [2700,200,2500,2100]
    def metric(items):
        c = sum(v['correct'] for v in items)
        return dict(n=len(items),correct=c,accuracy=c/len(items),wilson_95=wilson(c,len(items)))
    tables = {m:{s:{layer:metric([g[r['question_id']] for r in rs if layer == 'all' or r['stratum'] == layer]) for layer in ('all','short','medium','long')} for s,rs in subsets.items()} for m,g in records.items()}
    pairs = {m:{s:compare(rs,current,g) for s,rs in subsets.items()} for m,g in records.items() if m != METHOD}
    # 步骤2：官方题型只作分组评价，不参与选择；逐题来源与全部答案同时保存。
    task_types = sorted({r['task_type'] for r in data})
    type_stats = {m:{t:metric([g[r['question_id']] for r in data if r['task_type'] == t]) for t in task_types} for m,g in records.items()}
    flat = [dict(question_id=r['question_id'],video_id=r['video_id'],stratum=r['stratum'],task_type=r['task_type'],source='imported_development' if r['question_id'] in devq else 'new_full_batch',
        methods={m:dict(correct=g[r['question_id']]['correct'],answer=g[r['question_id']]['answer']['parsed_answer'],frames=len(g[r['question_id']]['selection']['selected_frames'])) for m,g in records.items()}) for r in data]
    durable(run/'per_question.json',flat)
    cases = []
    for m in ('BASE-Uniform','RD-1.2'):
        for layer in ('short','medium','long'):
            for direction,key in [('gain','left_only'),('loss','right_only')]:
                eligible = set(pairs[m]['all2700']['cells'][key])
                q = next((r['question_id'] for r in data if r['stratum'] == layer and r['question_id'] in eligible),None)
                if q is not None: cases.append(dict(question_id=q,comparison=m,stratum=layer,direction=direction))
    durable(run/'case_manifest.json',dict(selection='first frozen-order gain/loss per stratum and comparator; no backfill',entries=cases,unique_ids=list(dict.fromkeys(c['question_id'] for c in cases))))
    # 步骤3：窗口/补查不变，分别记录粗资格恢复、细帧入选、预算和替换，避免把全部变化称为补查收益。
    stage_rows = []
    from fractions import Fraction
    from videoqa_methods.density_selection import source_time
    for row in data:
        q = row['question_id']; pool = read_json(BASE/'pools'/(q+'.json'))
        item = read_json(run/'selections'/METHOD/(q+'.json')); trace = read_json(run/item['trace_file'])
        ids = set(item['selected_indices']); old = set(pool['selected_indices']); n = pool['initial_count']
        seed_ids = set(i for r in pool['plan']['regions'] for i in r['seed_indices'])
        outside = [i for i in range(n) if not any(Fraction(r['left_fraction']) <= source_time(pool['candidates'][i],pool['video']) <= Fraction(r['right_fraction']) for r in pool['plan']['regions'])]
        assert set(trace['eligible_indices']) == set(range(len(pool['candidates'])))
        assert set(trace['protected_indices']) <= ids
        assert sum(trace['region_budgets'].values()) == len(ids) == min(32,len(pool['candidates']))
        stage_rows.append(dict(question_id=q,initial_count=n,pool_size=len(pool['candidates']),region_count=len(pool['plan']['regions']),
            refinement_region_count=sum(r['refine'] for r in pool['plan']['regions']),new_candidates=len(pool['candidates'])-n,
            selected_fine=len([i for i in ids if i>=n]),old_selected_fine=len([i for i in old if i>=n]),
            selected_seeds=len(ids&seed_ids),eligible_outside_coarse=len(outside),selected_outside_coarse=sorted(ids&set(outside)),
            region_budgets=trace['region_budgets'],old_region_budgets=pool['budget_trace']['region_budgets'],
            removed=sorted(old-ids),added=sorted(ids-old),objective=trace['objective'],normalized_objective=trace['normalized_objective']))
    durable(run/'stage_analysis.json',dict(status='completed',rows=stage_rows,new_model_calls=0,limitation='Structural proxies, not semantic evidence annotation'))
    times = [current[q] for q in spec['new_ids']]
    summary = dict(status='completed',method=METHOD,accuracy=tables,task_types=type_stats,comparisons=pairs,new_calls=2500,imported=200,
        frame_counts={m:dict(Counter(len(x['selection']['selected_frames']) for x in g.values())) for m,g in records.items()},
        parse_failures={m:[q for q,x in g.items() if x['answer']['parsed_answer'] is None] for m,g in records.items()},
        performance={k:distribution([r['timings'][k] for r in times]) for k in times[0]['timings']},
        execution=read_json(run/'execution.json'),selection_wall_seconds=read_json(run/'selection_summary.json')['wall_seconds'],
        synthetic_generations=read_json(run/'synthetic_acceptance.json')['synthetic_calls_started']+sum(read_json(p)['synthetic_generations'] for p in (run/'workers').glob('*.json')),
        failures=[str(p.relative_to(run)) for p in (run/'failures').glob('*.json')],
        limitations=['Repeated development selection; remaining data not entirely unseen','200 imported answers plus 2500 new; no best-of-runs','Cached selection/decoding QA, not raw-video end-to-end'],utc=utc())
    durable(run/'summary.json',summary)
    # 步骤4：自动生成可读统计，不把尚未执行的人工差异审核写成已完成。
    lines = ['# exp07 Qwen32全量结果','',
        '来源：200条冻结开发答案＋2500条新回答。主分母2700；人工案例解释需另行审核。','',
        '| 方法 | 全量 | 短 | 中 | 长 | 开发200 | 其余2500 | 开发视频外2100 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,groups in tables.items():
        a = groups['all2700']; cols = [f"{a['all']['correct']}/2700 ({100*a['all']['accuracy']:.3f}%)"]
        cols += [str(a[t]['correct']) for t in ('short','medium','long')]
        cols += [str(groups[t]['all']['correct']) for t in ('development200','other2500','outside_development_videos2100')]
        lines.append('| '+m+' | '+' | '.join(cols)+' |')
    lines += ['','## 配对差值','']
    for m,pair in pairs.items():
        a = pair['all2700']; lo,hi = a['stratified_video_cluster_bootstrap_95']
        lines.append(f"- 相对{m}：{a['difference_pp']:+.3f}个百分点；新增正确{len(a['cells']['left_only'])}、退化{len(a['cells']['right_only'])}；视频簇95%区间[{lo:+.3f},{hi:+.3f}]。")
    lines += ['','## 验证边界','',
        '开发集已用于多轮版本选择，其他全量案例也曾被观察，不称完全盲测。当前冻结方法不因全量得分自动调整。',
        '完整分组、题型、共同32帧子集及性能见summary.json；全部答案来源见per_question.json；机制指标见stage_analysis.json。',
        '正式恢复验收以completion_audit.json为准；此统计生成本身不证明后续审计已完成。']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def audit(run):
    """接口：实际恢复完成状态，不加载模型；核对新旧账本均字节不变。"""
    run = Path(run)
    def files():
        return {str(p.relative_to(run)):sha256(p) for root in (run,run/'imported') for folder in ('results','calls','raw_generations') for p in (root/folder/METHOD).glob('*.json')}
    before = files(); assert len(before) == 8100
    execute(run)
    assert before == files()
    spec = read_json(run/'protocol.json')
    for p,x in spec['model_files'].items(): assert sha256(p) == x['sha256']
    durable(run/'completion_audit.json',dict(status='passed',unchanged_files=8100,extra_generations=0,utc=utc()))
