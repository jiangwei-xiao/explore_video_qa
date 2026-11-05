"""exp27双配置全量确认：冻结范围收缩、共同选择器重放和1096次调用计划。"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import multiprocessing as mp
import os
import shutil
import time

import numpy as np
from videoqa_runtime.common import ROOT, read_json, sha256, offline_environment
from videoqa_full.state import durable, digest, exclusive, rows, pilot_ids, utc
from videoqa_full.prepare import environment
from videoqa_lmms_bridge.common import framework_identity, load_task
from videoqa_budget32.frames import FrameProvider, validate_frames
from videoqa_methods.region_refinement_scope import restrict_refinement_pool
from videoqa_methods.rd_selection import select_rd
from videoqa_audit.framework_contraction import RUN32, RUN8
from .component_ablation_runtime import cpu_tools, cpu_identity, processed_matches, copy_file_stats, historical_bundle

METHOD='RD-1.2-exp27'
CONTROLS={'S':METHOD}
PROFILES=('qwen32','llava8')
PYTHONS={'qwen32':'/App/conda/envs/explore_video_qa_lmms_qwen/bin/python',
         'llava8':'/App/conda/envs/explore_video_qa_lmms/bin/python'}
PARENTS={'qwen32':RUN32,'llava8':RUN8}
HISTORY=ROOT/'outputs/methods/rd12_exp27_region_scope_20261008_r1'
DEV=ROOT/'data/manifests/video_mme_development_200_20260920.json'
EXPECTED={'qwen32':dict(development=200,parent=1561,new=939),
          'llava8':dict(development=200,parent=2343,new=157)}
CORE_FILES=('src/videoqa_generalization/region_scope_full.py',
 'src/videoqa_generalization/region_scope_full_runtime.py','src/videoqa_generalization/region_scope_full_report.py',
 'scripts/run_region_scope_full.py','src/videoqa_methods/rd_selection.py',
 'src/videoqa_methods/region_refinement_scope.py','src/videoqa_methods/budget_contract.py',
 'src/videoqa_methods/density_selection.py','src/videoqa_methods/coarse_eligibility.py',
 'src/videoqa_methods/weighted_seed_support.py','src/videoqa_methods/prefix_coverage_budget.py',
 'configs/retrieval_density_v1.json')
QA_FILES=('src/videoqa_generalization/llava_observe.py','src/videoqa_generalization/llava_budget8.py',
 'src/videoqa_budget32/observe.py','src/videoqa_budget32/qwen_adapter.py','src/videoqa_budget32/frames.py',
 'src/videoqa_full/qwen_fa2_backend.py','src/videoqa_full/qwen_backend.py','src/videoqa_runtime/llava_backend.py',
 'src/videoqa_lmms_bridge/common.py','src/videoqa_lmms_bridge/llava_adapter.py','src/videoqa_full/state.py',
 'src/videoqa_audit/region_scope_runtime.py','src/videoqa_generalization/component_ablation_runtime.py')
_PREPARE=None


def freeze(path,value):
    """接口：协议和完成产物只创建一次；恢复核对内容，不改写字节或mtime。"""
    path=Path(path)
    if path.exists():
        if read_json(path)!=value:raise ValueError('Frozen record differs: '+str(path))
    else:durable(path,value)


def budget(profile):
    """接口：固定两工作点，不开放独立M或算法参数。"""
    return {'qwen32':32,'llava8':8}[profile]


def source_identity(selection):
    """接口：跨池比较有序源帧与完整时钟，不以重编号后的候选ID比较媒体。"""
    v=selection['video']
    return dict(video={k:v[k] for k in ('path','start_pts','time_base','duration_fraction')},
                frames=[[f['source_frame_index'],f['source_pts']] for f in selection['selected_frames']])


def check_slot(value,profile,q):
    """接口：恢复缓存必须属于指定题号、预算及方法，合法记录错槽也拒绝。"""
    if (value['question_id']!=q or value['profile']!=profile or value['frame_budget']!=budget(profile)
            or value['method']!=METHOD):raise ValueError('Question/profile/budget/method slot differs')


def compare_trace(current,old):
    """接口：离散前缀/配额/成员零容差；共同数值绝对误差最多1e-6。"""
    # 步骤1逐字段核对受限选择空间；步骤2对原浮点路径的曲线和局部获胜值比较。
    for key in ('region_budgets','protected_indices','eligible_indices'):
        if current[key]!=old[key]:raise ValueError('Discrete replay mismatch: '+key)
    if len(current['regions'])!=len(old['regions']):raise ValueError('Region count differs')
    errors=[abs(current['objective']-old['objective'])]
    for a,b in zip(current['regions'],old['regions']):
        for key in ('region_id','members','reference_coarse_indices','prefix','density_weight'):
            if a[key]!=b[key]:raise ValueError('Region replay mismatch: '+key)
        if len(a['coverage_curve'])!=len(b['coverage_curve']) or len(a['local_steps'])!=len(b['local_steps']):
            raise ValueError('Trace length differs')
        errors.extend(np.abs(np.asarray(a['coverage_curve'])-b['coverage_curve']).tolist())
        for x,y in zip(a['local_steps'],b['local_steps']):
            if x['candidate_index']!=y['candidate_index']:raise ValueError('Local winner differs')
            errors.extend(abs(x[k]-y[k]) for k in ('support','redundancy','local_score'))
    maximum=max(errors,default=0.)
    if maximum>1e-6:raise ValueError('Numeric replay tolerance exceeded')
    return maximum


def offline_one(job):
    """接口：只读取冻结候选/特征并重放exp27；不解码、不调用任何模型。"""
    root,profile,q=Path(job[0]),job[1],job[2];k=budget(profile)
    exports=old_exports();path=HISTORY/'offline'/profile/(q+'.json')
    if sha256(path)!=exports[str(path)]:raise ValueError('Historical offline selection changed')
    old=read_json(path);check_slot(old,profile,q);trp=HISTORY/old['trace_file']
    if sha256(trp)!=old['trace_sha256'] or sha256(trp)!=exports[str(trp)]:raise ValueError('Historical trace changed')
    sources={str(path):sha256(path),str(trp):sha256(trp),**old['sources']}
    dest=root/'selections'/profile/METHOD/(q+'.json')
    if dest.exists():
        saved=read_json(dest);check_slot(saved,profile,q)
        if saved['sources']!=sources:raise ValueError('Replay recovery source differs')
        for p,h in sources.items():
            if sha256(p)!=h:raise ValueError('Historical source changed')
        if sha256(root/saved['trace_file'])!=saved['trace_sha256']:raise ValueError('Replay trace changed')
        return saved
    began=time.perf_counter()
    for p,h in sources.items():
        if sha256(p)!=h:raise ValueError('Historical source differs')
    pool=read_json(old['source_pool']);base=np.load(old['source_features'],allow_pickle=False)
    features=base if k==32 else base[pool['original_candidate_indices']]
    loaded=time.perf_counter();new,nf,scope=restrict_refinement_pool(pool,features,k)
    if new['parent_candidate_indices']!=old['parent_candidate_indices'] or scope!=old['scope']:
        raise ValueError('Scope/pool mapping replay differs')
    selected,trace=select_rd(new,nf,k,'RD-1.2-exp07',trace_mode='full')
    error=compare_trace(trace,read_json(trp))
    selection=dict(video=new['video'],selected_frames=[new['candidates'][i] for i in selected])
    if selected!=old['selected_indices'] or selection!=old['selection']:raise ValueError('Frozen exp27 frames differ')
    validate_frames(selection,k);trace_path=root/'traces'/profile/(q+'.json');freeze(trace_path,trace)
    result=dict(question_id=q,profile=profile,method=METHOD,frame_budget=k,selection=selection,
        sources=sources,trace_file=str(trace_path.relative_to(root)),trace_sha256=sha256(trace_path),
        scope=scope,implementation_id=trace['implementation_id'],coverage_method='RD-1.2-exp07',
        maximum_numeric_error=error,actual_frame_count=len(selected),
        timings=dict(asset_read_seconds=loaded-began,selection_seconds=time.perf_counter()-loaded),
        new_model_calls=0)
    freeze(dest,result);return result


@lru_cache(maxsize=1)
def old_exports():
    """接口：旧完成摘要是源导出清单；不覆盖其协议或将当前文件自身当作旧指纹。"""
    return read_json(HISTORY/'offline_summary.json')['files']


def offline(root,workers=8):
    """接口：5400项共同入口控制及来源冻结，出错立即停止，不重新取证。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);data=rows()
    fixed=[HISTORY/'offline_protocol.json',HISTORY/'offline_summary.json',HISTORY/'independent_completion_audit.json',
           DEV,ROOT/'outputs/analysis/rd_equivalence__S1__20261027_r2/completion.json']
    spec=dict(task='exp27_region_scope_full',method=METHOD,coverage_parent='RD-1.2-exp07',
        profiles=list(PROFILES),question_ids=[r['question_id'] for r in data],
        sources={str(p):sha256(p) for p in fixed},code_sha256={p:sha256(ROOT/p) for p in (*CORE_FILES,*QA_FILES)},
        max_new_real_qa=1096,max_synthetic=16,expected_partition=EXPECTED)
    with exclusive(root/'offline.lock'):
        freeze(root/'offline_protocol.json',spec)
        if (root/'offline_summary.json').exists():
            v=read_json(root/'offline_summary.json')
            for p,h in v['exports'].items():
                if sha256(root/p)!=h:raise ValueError('Completed replay changed')
            return v
        for n in spec['code_sha256']:
            p=root/'code_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
        records=[];began=time.perf_counter()
        # 步骤1只启动CPU进程；步骤2按冻结题序落盘，每项同时核验旧池/轨迹来源。
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as ex:
            for item in ex.map(offline_one,[(str(root),p,r['question_id']) for r in data for p in PROFILES]):
                records.append(item)
                if len(records)%20==0:durable(root/'offline_status.json',dict(status='running',completed=len(records),total=5400,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        exports={}
        for v in records:
            for p in [root/'selections'/v['profile']/METHOD/(v['question_id']+'.json'),root/v['trace_file']]:
                exports[str(p.relative_to(root))]=sha256(p)
        metrics={p:dict(questions=2700,parent_fine_count=sum(v['scope']['parent_fine_count'] for v in records if v['profile']==p),
            fine_count=sum(v['scope']['fine_count'] for v in records if v['profile']==p),
            actual_frame_counts=dict(Counter(str(v['actual_frame_count']) for v in records if v['profile']==p))) for p in PROFILES}
        value=dict(status='passed',records=len(records),exports=exports,metrics=metrics,
            maximum_numeric_error=max(v['maximum_numeric_error'] for v in records),seconds=time.perf_counter()-began,new_model_calls=0)
        freeze(root/'offline_summary.json',value);return value


def historical_reference(profile,q,development):
    """接口：固定开发exp27优先，其余只引用exp07；正确性不参与选源。"""
    if development:return HISTORY/'qa'/profile/'results'/METHOD/(q+'.json')
    from videoqa_audit.region_scope_runtime import parent_result
    return parent_result(profile,q)


def provenance(path,old):
    """接口：开发引用递归追溯真实主调用，保留中间结果和协议，不伪造新调用。"""
    path=Path(path);q=old['question_id'];sources={str(path):sha256(path)}
    # 步骤1旧开发导入可能没有自己的call，必须沿输入导出解析原父调用。
    if path.is_relative_to(HISTORY) and old.get('imported'):
        run=HISTORY/'qa'/('qwen32' if old['model'].startswith('Qwen') else 'llava8')
        item_path=run/'selections'/METHOD/(q+'.json');item=read_json(item_path)
        for p,h in item['parent_sources'].items():
            if sha256(p)!=h:raise ValueError('Development import ancestry changed')
        parent_path=Path(item['parent_result']);parent=read_json(parent_path)
        if parent['answer']!=old['answer']:raise ValueError('Development import answer differs')
        refs,gpu=provenance(parent_path,parent);sources.update(refs)
        sources.update({str(item_path):sha256(item_path),str(run/'protocol.json'):sha256(run/'protocol.json')})
        return sources,gpu
    # 步骤2父版共享结果和imported目录由原通用解析器定位实际代表方法。
    refs=historical_bundle(path,old);sources.update(refs)
    calls=[p for p in refs if '/calls/' in p]
    if len(calls)!=1:raise ValueError('Physical provenance is ambiguous')
    return sources,read_json(calls[0])['gpu']


def initialize_prepare(root,profile,binding_sha):
    """接口：CPU工作进程独立处理器，避免GIL串行；特征与模型生成不在此加载。"""
    global _PREPARE
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    offline_environment();root=Path(root)
    _PREPARE=dict(root=root,profile=profile,binding_sha=binding_sha,tools=cpu_tools(profile),
        source_spec=read_json(PARENTS[profile]/'protocol.json'),offline=read_json(root/'offline_summary.json'),
        dev={r['question_id'] for r in read_json(DEV)['rows']})
    load_task()


def prepare_one(row):
    """接口：逐题冻结RGB/官方像素/提示Token，再判定是否可以引用历史返回。"""
    z=_PREPARE;root,profile=z['root'],z['profile'];q=row['question_id'];run=root/'qa'/profile;dest=run/'inputs'/(q+'.json')
    if dest.exists():
        v=read_json(dest)
        if v['question_id']!=q or v['profile']!=profile or v['prepare_binding_sha256']!=z['binding_sha']:raise ValueError('Input cache slot differs')
        for p,h in v['sources'].items():
            if sha256(p)!=h:raise ValueError('Input preparation source changed')
        return q,sha256(dest),v['groups'],v['reference_kind']
    path=root/'selections'/profile/METHOD/(q+'.json')
    if sha256(path)!=z['offline']['exports'][str(path.relative_to(root))]:raise ValueError('Replay export changed')
    item=read_json(path);check_slot(item,profile,q);selection=item['selection'];began=time.perf_counter()
    images,rgb=FrameProvider(selection).decode()
    try:processed=cpu_identity(profile,z['tools'],images,row)
    finally:
        for im in images:im.close()
    decoded=time.perf_counter();history=historical_reference(profile,q,q in z['dev']);old=read_json(history)
    for key in ('question_id','video_id','question','options','stratum'):
        if old[key]!=row[key]:raise ValueError('Historical task identity differs')
    same=source_identity(selection)==source_identity(old['selection']);reference=None
    if q in z['dev'] and not same:raise ValueError('Frozen development input differs')
    reference_kind='new';sources={str(path):sha256(path),str(history):sha256(history)}
    if same:
        if rgb!=old['rgb_sha256'] or not processed_matches(profile,processed,old['answer']['observed']):raise ValueError('Historical complete input differs')
        refs,gpu=provenance(history,old);sources.update(refs)
        reference_kind='development' if q in z['dev'] else 'parent'
        reference=dict(path=str(history),sources=refs,gpu=gpu,kind=reference_kind)
    value=dict(question_id=q,profile=profile,method=METHOD,frame_budget=budget(profile),
        items={'S':dict(selection=selection,rgb_sha256=rgb,processed=processed)},
        groups=[dict(representative='S',controls=['S'],reference=reference)],sources=sources,
        prepare_binding_sha256=z['binding_sha'],reference_kind=reference_kind,
        timings=dict(decode_and_identity_seconds=decoded-began,provenance_seconds=time.perf_counter()-decoded),
        duration_seconds=float(Fraction(selection['video']['duration_fraction'])))
    freeze(dest,value);return q,sha256(dest),value['groups'],reference_kind


def validate_partition(profile,counts):
    """接口：冻结400开发/3904父引用/1096新调用；身份异常不自动追加问答。"""
    if dict(counts)!=EXPECTED[profile]:raise ValueError('Expected import/new partition differs: '+str(dict(counts)))


def prepare(root,profile,allowed_gpus):
    """接口：完整输入在任何真实生成前冻结，原环境/权重/推理默认值逐项核验。"""
    from . import region_scope_full_runtime as runtime
    root=Path(root);run=root/'qa'/profile;run.mkdir(parents=True,exist_ok=True);data=rows()
    if not allowed_gpus or len(set(allowed_gpus))!=len(allowed_gpus) or not set(allowed_gpus)<=set(map(str,range(8))):raise ValueError('Invalid GPUs')
    offline_environment()
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():return verify(run)
        old=read_json(PARENTS[profile]/'protocol.json');devspec=read_json(HISTORY/'qa'/profile/'protocol.json')
        if environment(PYTHONS[profile])!=old['environment'] or old['environment']!=devspec['environment'] or framework_identity()!=old['framework']:
            raise ValueError('Historical environment/framework changed')
        codes={p:sha256(ROOT/p) for p in (*CORE_FILES,*QA_FILES)}
        for p in QA_FILES:
            for source in (old,devspec):
                if p in source['code_sha256'] and codes[p]!=source['code_sha256'][p]:raise ValueError('Frozen inference code changed: '+p)
        # 步骤1只读核验模型与900视频身份；旧摘要可复用，状态变化时重新核对字节。
        models=copy_file_stats(old['model_files']);stats={}
        for vid,st in old['video_stats'].items():
            p=Path(st['path']);now=p.stat()
            if (now.st_size!=st['bytes'] or now.st_mtime_ns!=st['mtime_ns']) and sha256(p)!=old['video_hashes'][vid]:raise ValueError('Video changed')
            stats[vid]=dict(st,bytes=now.st_size,mtime_ns=now.st_mtime_ns)
        binding=dict(profile=profile,code_sha256=codes,offline_protocol_sha256=sha256(root/'offline_protocol.json'),
            offline_summary_sha256=sha256(root/'offline_summary.json'),parent_protocol_sha256=sha256(PARENTS[profile]/'protocol.json'),
            development_protocol_sha256=sha256(HISTORY/'qa'/profile/'protocol.json'),model_files=models,video_stats=stats)
        freeze(run/'prepare_binding.json',binding);bsha=sha256(run/'prepare_binding.json');began=time.perf_counter();exports={};groups={};counts=Counter()
        # 步骤2八个CPU进程各两线程；旧答案只作完整输入身份比较，不影响源选择。
        with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=initialize_prepare,initargs=(str(root),profile,bsha)) as ex:
            for i,(q,h,gs,kind) in enumerate(ex.map(prepare_one,data),1):
                exports['inputs/'+q+'.json']=h;groups[q]=gs;counts[kind]+=1
                if i%10==0:durable(run/'prepare_status.json',dict(status='running',completed=i,total=2700,seconds=time.perf_counter()-began,utc=utc()))
        validate_partition(profile,counts);new={q:['S'] if gs[0]['reference'] is None else [] for q,gs in groups.items()}
        pilot=list(dict.fromkeys(pilot_ids(data)+[r['question_id'] for r in data if new[r['question_id']]][:5]))
        first=read_json(historical_reference(profile,data[0]['question_id'],False))
        spec=dict(profile=profile,model=old['model'],method=METHOD,frame_budget=budget(profile),methods=CONTROLS,
            question_ids=[r['question_id'] for r in data],max_results=2700,max_new_calls=EXPECTED[profile]['new'],new_groups=new,
            reference_counts=dict(counts),input_exports=exports,allowed_gpus=allowed_gpus,model_files=models,video_stats=stats,
            environment=old['environment'],framework=old['framework'],code_sha256=codes,
            parent_protocol=str(PARENTS[profile]/'protocol.json'),parent_protocol_sha256=binding['parent_protocol_sha256'],
            development_protocol=str(HISTORY/'qa'/profile/'protocol.json'),development_protocol_sha256=binding['development_protocol_sha256'],
            offline_protocol_sha256=binding['offline_protocol_sha256'],offline_summary_sha256=binding['offline_summary_sha256'],
            historical_generation_config=first['answer']['observed']['model_generation_config'],
            pilot_question_ids=pilot,max_new_tokens=16,seed=2027,utc=utc())
        # 步骤3清单发布后仅导入引用，所有新问答等待两个配置的输入都完成冻结。
        freeze(run/'protocol.json',spec);runtime.materialize_imports(run,spec)
        freeze(run/'prepare_completed.json',dict(status='completed',reference_counts=dict(counts),seconds=time.perf_counter()-began))
        return spec


def verify(run):
    """接口：核验题号/预算/来源绑定，恢复不加载模型或执行选择。"""
    run=Path(run);spec=read_json(run/'protocol.json');root=run.parent.parent;profile=spec['profile']
    if spec['method']!=METHOD or spec['frame_budget']!=budget(profile) or spec['methods']!=CONTROLS or spec['question_ids']!=[r['question_id'] for r in rows()]:raise ValueError('Protocol matrix differs')
    validate_partition(profile,spec['reference_counts'])
    if set(spec['new_groups'])!=set(spec['question_ids']) or any(v not in ([],['S']) for v in spec['new_groups'].values()):raise ValueError('Invalid call assignment')
    if spec['max_new_calls']!=EXPECTED[profile]['new'] or sum(map(len,spec['new_groups'].values()))!=spec['max_new_calls']:raise ValueError('Call budget differs')
    if framework_identity()!=spec['framework'] or environment(PYTHONS[profile])!=spec['environment']:raise ValueError('Frozen environment differs')
    for p,h in spec['code_sha256'].items():
        if sha256(ROOT/p)!=h:raise ValueError('Frozen implementation changed: '+p)
    for label in ('parent','development'):
        if sha256(spec[label+'_protocol'])!=spec[label+'_protocol_sha256']:raise ValueError('Historical protocol changed')
    for name in ('offline_protocol','offline_summary'):
        if sha256(root/(name+'.json'))!=spec[name+'_sha256']:raise ValueError('Offline input changed')
    for p,v in spec['model_files'].items():
        st=Path(p).stat()
        if st.st_size!=v['bytes'] or st.st_mtime_ns!=v['mtime_ns']:raise ValueError('Model state changed')
    for rel,h in spec['input_exports'].items():
        p=run/rel
        if sha256(p)!=h:raise ValueError('Frozen input changed')
        v=read_json(p);check_slot(v,profile,p.stem)
        if (v['groups'][0]['reference'] is None)!=(spec['new_groups'][p.stem]==['S']):raise ValueError('Call group differs from frozen input')
        for source,expected in v['sources'].items():
            if sha256(source)!=expected:raise ValueError('Input ancestry changed')
    return spec
