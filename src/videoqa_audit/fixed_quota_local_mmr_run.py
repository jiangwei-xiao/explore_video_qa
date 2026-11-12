"""400题固定配额局部MMR的来源、冻结、CPU选择与安全恢复。"""
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import time

from . import fixed_quota_local_mmr as method
from .selection_structure_run import (ROOT, PROFILES, PYTHONS, DEV, read, sha, digest,
    freeze, utc, snapshot_code, budget, source_identity, coarse_view, development_rows,
    bounded_results, _initialize_cpu)

PREVIOUS = ROOT/'outputs/diagnostics/selection_structure__videomme200__llava8_qwen32__20261112_r1'
FILES = ('src/videoqa_audit/fixed_quota_local_mmr.py',
    'src/videoqa_audit/fixed_quota_local_mmr_run.py',
    'src/videoqa_audit/fixed_quota_local_mmr_runtime.py',
    'src/videoqa_audit/fixed_quota_local_mmr_report.py',
    'scripts/run_fixed_quota_local_mmr.py')
OLD_PANEL = ('871-2','128-1','306-2','893-1','691-3','597-1','383-2','705-1')
DOCUMENT = 'docs/experiments/video_mme_200/2026-11-12/fixed_quota_local_mmr/执行协议.md'


def old_path(profile, kind, method_id, q):
    """接口：旧运行只读定位；路径不依赖答案或旧方法表现。"""
    return PREVIOUS/kind/profile/method_id/(q+'.json')


def verify_exports(paths, manifest):
    """接口：逐项核验实际读取的旧导出，不重新扫描全部视频或全量选择。"""
    for path in paths:
        if sha(path) != manifest['exports'][str(path.relative_to(PREVIOUS))]:
            raise ValueError('Accepted historical export changed: '+str(path))


def offline_one(job):
    """接口：一题粗资产与父轨迹→父输入重构及唯一H选择；不执行父DP。"""
    from videoqa_generalization.component_ablations import load_assets
    root, profile, q = job; root=Path(root); dest=root/'offline'/profile/(q+'.json')
    if dest.exists():
        saved=read(dest)
        for p,h in saved['sources'].items():
            if sha(p)!=h: raise ValueError('Cached offline source changed')
        for p,h in saved['exports'].items():
            if sha(root/p)!=h: raise ValueError('Cached offline output changed')
        return saved
    began=time.perf_counter(); pool,f,_,_,sources=load_assets(profile,q)
    coarse,cf=coarse_view(pool,f); k=budget(profile)
    tp=old_path(profile,'traces',method.PARENT,q); sp=old_path(profile,'selections',method.PARENT,q)
    summary=read(PREVIOUS/'offline_summary.json'); verify_exports([tp,sp],summary)
    parent=read(tp); frozen=read(sp); sources.update({str(p):sha(p) for p in (tp,sp)})
    # 步骤1用旧前缀及最终实际配额重构P，不重新运行完整父选择链。
    _,ids=method.parent_partition(coarse,parent,k)
    reconstruction=dict(video=coarse['video'],selected_frames=[coarse['candidates'][i] for i in ids])
    if source_identity(reconstruction)!=source_identity(frozen['selection']):
        raise ValueError('Frozen parent input reconstruction differs')
    loaded=time.perf_counter(); selected,trace=method.select_region_mmr_fixed_quota(coarse,cf,parent,k)
    elapsed=time.perf_counter()-loaded
    if profile=='llava8' and q=='128-1':
        gp=old_path(profile,'selections',method.GLOBAL,q); verify_exports([gp],summary)
        if selected!=read(gp)['selected_indices']: raise ValueError('Single-region 128-1 differs from G')
        sources[str(gp)]=sha(gp)
    # 步骤2输出H及逐区分量，检查点最后发布；不覆盖任何旧产物。
    tracepath=root/'traces'/profile/method.CONTROL/(q+'.json'); freeze(tracepath,trace)
    selection=dict(video=coarse['video'],selected_frames=[coarse['candidates'][i] for i in selected])
    value=dict(question_id=q,profile=profile,method=method.CONTROL,frame_budget=k,
        selected_indices=selected,selection=selection,selection_seconds=elapsed,
        trace_sha256=sha(tracepath),actual_frame_count=len(selected))
    path=root/'selections'/profile/method.CONTROL/(q+'.json')
    if path.exists(): value['selection_seconds']=read(path)['selection_seconds']
    freeze(path,value)
    saved=dict(question_id=q,profile=profile,sources=sources,
        exports={str(p.relative_to(root)):sha(p) for p in (path,tracepath)},
        parent_reconstruction_exact=True,regions=len(trace['regions']),
        source_read_seconds=loaded-began,selection_seconds=elapsed,new_model_calls=0)
    freeze(dest,saved); return saved


def offline(root,workers=4):
    """接口：仅400项离线选择，最多4进程×2线程，恢复不重复完成项。"""
    from videoqa_full.state import exclusive,durable
    root=Path(root)
    with exclusive(root/'offline.lock'):
        if (root/'offline_summary.json').exists(): return verify_offline(root)
        code={p:sha(ROOT/p) for p in FILES}
        spec=dict(control=method.CONTROL,profiles=list(PROFILES),question_ids=[r['question_id'] for r in development_rows()],
            previous_run=str(PREVIOUS),previous_completion_sha256=sha(PREVIOUS/'completion_audit.json'),
            previous_offline_sha256=sha(PREVIOUS/'offline_summary.json'),code_sha256=code,
            execution_document=DOCUMENT,execution_document_sha256=sha(ROOT/DOCUMENT),
            max_real_qa=400,max_synthetic_qa=8,old_panel=list(OLD_PANEL),new_blip_calls=0,new_visual_calls=0)
        if read(PREVIOUS/'completion_audit.json')['status']!='passed': raise ValueError('Previous run not accepted')
        freeze(root/'offline_protocol.json',spec); snapshot_code(root,code)
        jobs=[(str(root),p,q) for p in PROFILES for q in spec['question_ids']]
        outputs=[]; began=time.perf_counter()
        with ProcessPoolExecutor(max_workers=min(4,workers),initializer=_initialize_cpu) as executor:
            for value in bounded_results(executor,offline_one,jobs,limit=8):
                outputs.append(value)
                durable(root/'offline_status.json',dict(status='running',completed=len(outputs),total=400,
                    elapsed_seconds=time.perf_counter()-began,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        sources={}; exports={}
        for value in outputs: sources.update(value['sources']); exports.update(value['exports'])
        summary=dict(status='passed',parent_reconstructions=400,control_selections=400,
            regions=sum(v['regions'] for v in outputs),sources=sources,exports=exports,
            protocol_sha256=sha(root/'offline_protocol.json'),wall_seconds=time.perf_counter()-began,
            no_full_replay=True,new_model_calls=0)
        freeze(root/'offline_summary.json',summary); return summary


def verify_offline(root):
    """接口：完成恢复只核验摘要，不调用模型、选择器或重新求解DP。"""
    root=Path(root); spec=read(root/'offline_protocol.json'); summary=read(root/'offline_summary.json')
    if (summary['control_selections']!=400 or summary['parent_reconstructions']!=400
            or summary['protocol_sha256']!=sha(root/'offline_protocol.json')):
        raise ValueError('Offline matrix differs')
    for p,h in spec['code_sha256'].items():
        if sha(ROOT/p)!=h or sha(root/'code_snapshot'/p)!=h: raise ValueError('Frozen execution code differs')
    if (sha(PREVIOUS/'completion_audit.json')!=spec['previous_completion_sha256']
            or sha(PREVIOUS/'offline_summary.json')!=spec['previous_offline_sha256']):
        raise ValueError('Prior acceptance fingerprint changed')
    if sha(ROOT/spec['execution_document'])!=spec['execution_document_sha256']:
        raise ValueError('Frozen execution document changed')
    for p,h in summary['sources'].items():
        if sha(p)!=h: raise ValueError('Frozen source changed')
    for p,h in summary['exports'].items():
        if sha(root/p)!=h: raise ValueError('Frozen offline export changed')
    return summary
