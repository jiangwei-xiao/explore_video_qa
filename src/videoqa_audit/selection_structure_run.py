"""同粗池选择组织诊断的来源契约及全量CPU重放；与模型运行层隔离。"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np

from . import selection_structure as method

ROOT = method.ROOT
PROFILES = ('llava8', 'qwen32')
PYTHONS = dict(llava8='/App/conda/envs/explore_video_qa_lmms/bin/python',
               qwen32='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python')
PARENT_RUN = ROOT/'outputs/full_videoqa/rd_components__videomme2700__llava8_qwen32__20261026_r2'
DEV = ROOT/'data/manifests/video_mme_development_200_20260920.json'
FILES = ('src/videoqa_audit/selection_structure.py',
         'src/videoqa_audit/selection_structure_run.py',
         'src/videoqa_audit/selection_structure_runtime.py',
         'src/videoqa_audit/selection_structure_report.py',
         'scripts/run_selection_structure_diagnostic.py')


def read(path):
    """接口：读取JSON记录；原始来源只读使用。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    """接口：有界读取文件摘要，不将完整特征/视频载入内存。"""
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    """接口：稳定结构身份，不依赖字典插入顺序或默认NaN序列化。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def freeze(path, value):
    """接口：首次发布原子fsync记录；恢复仅接受相同内容，不改写既有字节。"""
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen record changed: '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        # 步骤1写入临时文件；步骤2硬链接首次发布，竞争者不能覆盖先发布的内容。
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read(path) != value:
                raise ValueError('Concurrent frozen record differs: '+str(path))
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def utc():
    """接口：运行与审计时间统一使用UTC。"""
    return datetime.now(timezone.utc).isoformat()


def snapshot_code(root,files):
    """接口：冻结实际执行源码/配置字节，摘要和副本都必须匹配，不复制模型或视频。"""
    root=Path(root)
    # 步骤1路径仅限仓库源码清单；步骤2独占发布后校验字节，不改写已存在的快照。
    for name,h in files.items():
        source=ROOT/name; target=root/'code_snapshot'/name
        if not source.resolve().is_relative_to(ROOT.resolve()) or Path(name).is_absolute():
            raise ValueError('Snapshot source is outside repository')
        if sha(source)!=h: raise ValueError('Snapshot source changed before publication')
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:
                stream.write(source.read_bytes());stream.flush();os.fsync(stream.fileno())
        if sha(target)!=h: raise ValueError('Execution snapshot differs')


def budget(profile):
    """接口：两个固定配置只派生K，不开放可搜索的算法参数。"""
    if profile not in PROFILES: raise ValueError('Unknown profile')
    return 8 if profile == 'llava8' else 32


def source_identity(selection):
    """接口：跨池比较用真实源身份及顺序，不直接比较候选编号。"""
    v = selection['video']
    return dict(video={k:v[k] for k in ('path','start_pts','time_base','duration_fraction')},
                frames=[[f['source_frame_index'],f['source_pts']] for f in selection['selected_frames']])


def coarse_view(pool, features):
    """接口：只截取已正确映射的粗前缀，禁止把K32补查行直接用于K8。"""
    n = pool['initial_count']
    if len(features) != len(pool['candidates']): raise ValueError('Mapped pool/feature count differs')
    return dict(pool, candidates=pool['candidates'][:n]), features[:n]


def map_to_parent(selected, trace, seed_indices):
    """接口：选择完成后映回父区域/前缀，纯描述，不反向限制G/W的选择空间。"""
    chosen=set(selected); owner={}; records=[]
    for region in trace['coarse_prefix_trace']['regions']:
        rid=str(region['region_id']); refs=region['reference_coarse_indices']; prefix=region['prefix']
        for i in refs:
            if i in owner: raise ValueError('Parent coarse cells overlap')
            owner[i]=rid
        in_region=[i for i in selected if i in set(refs)]
        quota=trace['region_budgets'][rid]
        records.append(dict(region_id=rid,parent_quota=quota,control_count=len(in_region),
            selected_prefix_positions={str(i):(prefix.index(i)+1 if i in prefix else None) for i in in_region},
            parent_prefix_selected=prefix[:quota],selected_indices=in_region))
    if any(i not in owner for i in selected): raise ValueError('Control frame missing parent ownership')
    return dict(regions=records,seed_retained=sorted(chosen & set(seed_indices)),
        anchor_retained=sorted(chosen & set(trace['protected_indices'])),
        limitation='Post-selection descriptive mapping; no semantic loss or isolated causal claim')


def compare_trace(actual, expected, path='trace'):
    """接口：父轨迹离散零容差、数值1e-6；仅排除运行计时和旧控制外层标签。"""
    ignored = {'selection_seconds','control','quota_origin'}
    if isinstance(expected, dict):
        if not isinstance(actual, dict): raise ValueError('Trace type differs: '+path)
        for key, value in expected.items():
            if key in ignored: continue
            if key not in actual: raise ValueError('Missing trace quantity: '+path+'/'+key)
            compare_trace(actual[key], value, path+'/'+key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected): raise ValueError('Trace capacity differs: '+path)
        for i,(a,b) in enumerate(zip(actual, expected)): compare_trace(a,b,path+'/'+str(i))
    elif isinstance(expected, float):
        if not np.isfinite(expected) or not np.isfinite(actual) or abs(float(actual)-expected)>1e-6:
            raise ValueError('Trace numeric value differs: '+path)
    elif actual != expected:
        raise ValueError('Trace discrete quantity differs: '+path)


def verified_parent():
    """接口：旧完成审计绑定全量离线清单，不能把新读取的摘要当作历史验收证据。"""
    completion = read(PARENT_RUN/'completion_audit.json')
    summary = read(PARENT_RUN/'offline_summary.json')
    if (completion['status'] != 'passed' or summary['status'] != 'passed'
            or completion['offline_summary_sha256'] != sha(PARENT_RUN/'offline_summary.json')
            or summary['protocol_sha256'] != sha(PARENT_RUN/'offline_protocol.json')):
        raise ValueError('Historical component-ablation acceptance binding differs')
    return summary


def development_rows():
    """接口：直接复用已发布开发200题与顺序，不读取答案或重新抽样。"""
    from videoqa_audit.region_scope_runtime import development_rows as existing
    data = existing()
    counts = {s:sum(r['stratum']==s for r in data) for s in ('short','medium','long')}
    if counts != dict(short=67, medium=67, long=66): raise ValueError('Development strata differ')
    return data


def preflight():
    """接口：只读列出执行依赖；本地镜像缺源时返回未就绪，不冒充远端验收。"""
    required = [ROOT/p for p in FILES] + [method.REFERENCE, DEV,
        ROOT/'src/videoqa_runtime/common.py', PARENT_RUN/'offline_summary.json']
    required += list(map(Path, PYTHONS.values()))
    absent = [str(p) for p in required if not p.is_file()]
    result = dict(status='not_ready' if absent else 'files_present_not_yet_accepted',
                  missing=absent, new_model_calls=0, location=str(ROOT))
    if method.REFERENCE.is_file():
        result['reference_sha256'] = sha(method.REFERENCE)
        if result['reference_sha256'] != method.REFERENCE_SHA256:
            raise ValueError('Pinned WFS reference differs')
    return result


def _initialize_cpu():
    """接口：CPU工作进程禁止使用GPU，各数值库最多两线程。"""
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')


def bounded_results(executor,function,jobs,limit):
    """接口：最多limit个在途CPU任务；首个捕获异常后取消未启动项，不预排全部5400题。"""
    iterator=iter(enumerate(jobs)); pending={}
    def submit_one():
        item=next(iterator,None)
        if item is None: return
        ordinal,job=item;pending[executor.submit(function,job)]=ordinal
    for _ in range(limit): submit_one()
    # 步骤1只保持有界在途；步骤2先取返回值再补任务，异常不能自动顺延后续题。
    try:
        while pending:
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in sorted(ready,key=lambda f:pending[f]):
                pending.pop(future)
                yield future.result()
                submit_one()
    finally:
        for future in pending: future.cancel()


def offline_one(job):
    """接口：一题父重放及G/W两次选择；收到的row仅保留身份，不含标准答案。"""
    from videoqa_generalization.component_ablations import load_assets
    from videoqa_methods.coarse_only_ablation import select_coarse_only
    root, profile, row, parent_exports = job
    root = Path(root); q = row['question_id']; k = budget(profile)
    record_path = root/'offline'/profile/(q+'.json')
    pool, features, prefix, _, sources = load_assets(profile,q)
    old_sp = PARENT_RUN/'selections'/profile/method.PARENT/(q+'.json')
    old_tp = PARENT_RUN/'traces'/profile/'A'/(q+'.json')
    # 步骤1所有旧选择/轨迹必须命中旧完成审计绑定的导出摘要。
    for p in (old_sp,old_tp):
        if sha(p) != parent_exports[str(p.relative_to(PARENT_RUN))]:
            raise ValueError('Historical coarse control changed: '+str(p))
        sources[str(p)] = sha(p)
    if record_path.exists():
        saved = read(record_path)
        if saved['question_id'] != q or saved['profile'] != profile or saved['sources'] != sources:
            raise ValueError('Resume question/profile/source identity differs')
        for p,h in saved['exports'].items():
            if sha(root/p) != h: raise ValueError('Completed offline output changed')
        return saved
    coarse, cf = coarse_view(pool,features)
    method.validate_coarse(coarse,cf,k)
    # 步骤2交替先后仅影响计时顺序；两控制只读取完整粗池，不接触历史答题结果。
    started = time.perf_counter(); parent_ids,parent_trace = select_coarse_only(pool,features,prefix,k)
    old = read(old_sp)
    selection = dict(video=coarse['video'], selected_frames=[coarse['candidates'][i] for i in parent_ids])
    if source_identity(selection) != source_identity(old['selection']): raise ValueError('Parent discrete replay differs')
    compare_trace(parent_trace,read(old_tp))
    records = {method.PARENT:(parent_ids,parent_trace,time.perf_counter()-started)}
    order = method.CONTROLS if row['ordinal']%2 == 0 else method.CONTROLS[::-1]
    for control in order:
        began = time.perf_counter()
        ids,trace = method.select_coarse_control(coarse,cf,k,control)
        trace['posthoc_parent_mapping']=map_to_parent(ids,parent_trace,pool['plan']['seed_indices'])
        records[control] = (ids,trace,time.perf_counter()-began)
    # 步骤3导出当前轨迹和最终源帧；检查点最后发布，半成品不计作完成。
    exports = {}; details = {}
    for control,(ids,trace,seconds) in records.items():
        tp = root/'traces'/profile/control/(q+'.json'); freeze(tp,trace)
        selected = dict(video=coarse['video'], selected_frames=[coarse['candidates'][i] for i in ids])
        item = dict(question_id=q,video_id=row['video_id'],profile=profile,method=control,
                    frame_budget=k,coarse_count=pool['initial_count'],actual_frame_count=len(ids),
                    selected_indices=ids,selection=selected,selection_seconds=seconds,
                    trace_file=str(tp.relative_to(root)),trace_sha256=sha(tp))
        sp = root/'selections'/profile/control/(q+'.json')
        if sp.exists():
            # 最后检查点发布前中断时可重算科学输出，已发布计时仍保留首次计算值。
            item['selection_seconds']=read(sp)['selection_seconds']
        freeze(sp,item)
        for p in (tp,sp): exports[str(p.relative_to(root))] = sha(p)
        details[control] = dict(actual_frame_count=len(ids),selection_seconds=seconds)
    saved = dict(question_id=q,profile=profile,sources=sources,exports=exports,
                 parent_replay_exact=True,details=details,new_model_calls=0)
    freeze(record_path,saved)
    return saved


def verify_offline(root):
    """接口：恢复只校验已完成文件及当前源码，不重新调用选择器或改写产物。"""
    root = Path(root); spec = read(root/'offline_protocol.json'); summary = read(root/'offline_summary.json')
    if (summary['protocol_sha256'] != sha(root/'offline_protocol.json')
            or summary['parent_replays'] != 5400 or summary['control_selections'] != 10800):
        raise ValueError('Offline matrix incomplete')
    for p,h in spec['code_sha256'].items():
        if sha(ROOT/p) != h: raise ValueError('Frozen execution code changed: '+p)
        if sha(root/'code_snapshot'/p) != h: raise ValueError('Execution code snapshot changed: '+p)
    for p,h in summary['sources'].items():
        if sha(p) != h: raise ValueError('Input source changed: '+p)
    for p,h in summary['exports'].items():
        if sha(root/p) != h: raise ValueError('Offline export changed: '+p)
    return summary


def offline(root, workers=4):
    """接口：两个工作点2700题同粗池重放，最多4进程；不解码或生成答案。"""
    from videoqa_full.state import exclusive, rows, durable
    from videoqa_generalization.component_ablations import CORE_FILES
    if not 1 <= workers <= 4: raise ValueError('CPU process cap is four')
    root = Path(root); _initialize_cpu()
    with exclusive(root/'offline.lock'):
        if (root/'offline_summary.json').exists(): return verify_offline(root)
        old = verified_parent(); data = rows(); dev = development_rows()
        question_ids = [r['question_id'] for r in data]
        if len(question_ids) != 2700 or len(set(question_ids)) != 2700: raise ValueError('Full manifest differs')
        code = {p:sha(ROOT/p) for p in set((*FILES,*CORE_FILES,'third_party/WFS-SB-reference/core.py'))}
        snapshot_code(root,code)
        spec = dict(task='selection_structure_same_coarse_v1',profiles=list(PROFILES),
            methods=[method.PARENT,*method.CONTROLS],question_ids=question_ids,
            development_ids=[r['question_id'] for r in dev],max_new_real_qa=800,max_synthetic_qa=8,
            code_sha256=code,reference_sha256=method.REFERENCE_SHA256,paper_config=method.PAPER_CONFIG,
            parent_acceptance_sha256=sha(PARENT_RUN/'completion_audit.json'),
            parent_offline_sha256=sha(PARENT_RUN/'offline_summary.json'),dev_manifest_sha256=sha(DEV))
        freeze(root/'offline_protocol.json',spec)
        exports = {}; sources = {str(PARENT_RUN/name):sha(PARENT_RUN/name) for name in
            ('completion_audit.json','offline_summary.json','offline_protocol.json')}
        sources[str(DEV)]=sha(DEV)
        frame_counts = {}; began = time.perf_counter(); done = 0
        jobs = []
        for p in PROFILES:
            for i,r in enumerate(data):
                q=r['question_id']
                keys=[f'selections/{p}/{method.PARENT}/{q}.json',f'traces/{p}/A/{q}.json']
                jobs.append((str(root),p,dict(question_id=q,video_id=r['video_id'],ordinal=i),
                             {key:old['exports'][key] for key in keys}))
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_initialize_cpu) as executor:
            # 步骤1每个进程复用原资产映射；步骤2主进程只汇总小型身份及结果元数据。
            for value in bounded_results(executor,offline_one,jobs,workers):
                exports.update(value['exports']); sources.update(value['sources']); done += 1
                for m,detail in value['details'].items():
                    key = value['profile']+'/'+m+'/'+str(detail['actual_frame_count'])
                    frame_counts[key] = frame_counts.get(key,0)+1
                if done%10 == 0:
                    elapsed=time.perf_counter()-began
                    durable(root/'offline_status.json',dict(status='running',completed=done,total=5400,
                        pid=os.getpid(),sid=os.getsid(0),elapsed_seconds=elapsed,
                        eta_seconds=elapsed/done*(5400-done),utc=utc()))
        summary = dict(status='passed',parent_replays=5400,control_selections=10800,
            actual_frame_counts=frame_counts,exports=exports,sources=sources,
            protocol_sha256=sha(root/'offline_protocol.json'),wall_seconds=time.perf_counter()-began,
            new_model_calls=0)
        freeze(root/'offline_summary.json',summary)
        return summary
