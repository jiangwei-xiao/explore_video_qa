"""双数据集等价重放：来源绑定、独立旧链、一次DP计数及零计算恢复。"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from functools import lru_cache
import copy
import multiprocessing as mp
import os
import resource
import sys
import time
import traceback

import numpy as np

from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, digest, exclusive, rows, utc
from videoqa_runtime.baseline_report import distribution
from videoqa_methods import rd_selection as production
from videoqa_methods import prefix_coverage_budget as legacy_prefix
from videoqa_methods import relevance_coverage_budget as legacy_weighted
from videoqa_methods.budget_contract import select_exp07
from videoqa_audit.framework_contraction import BASE, RUN8, POOL16, RUN16, RUN32

METHODS = production.METHODS
SCOPES = {'videomme':{'questions':2700,'budgets':[8,16,32]},'mlvu':{'questions':2174,'budgets':[8,32]}}
WEIGHTED32 = ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
MLVU = ROOT/'outputs/full_videoqa/mlvu_dev__S1__llava8_qwen32__four_methods__20261014_r1'
MANIFEST = ROOT/'data/manifests/video_mme_full_2700.json'
FILES = ('src/videoqa_methods/rd_selection.py', 'src/videoqa_audit/rd_equivalence.py',
    'scripts/replay_rd_equivalence.py', 'src/videoqa_methods/budget_contract.py',
    'src/videoqa_methods/density_selection.py', 'src/videoqa_methods/coarse_eligibility.py',
    'src/videoqa_methods/weighted_seed_support.py', 'src/videoqa_methods/prefix_coverage_budget.py',
    'src/videoqa_methods/relevance_coverage_budget.py', 'src/videoqa_audit/framework_contraction.py',
    'src/videoqa_full/state.py', 'src/videoqa_runtime/common.py',
    'configs/retrieval_density_v1.json', 'configs/method_versions.json',
    'tests/test_rd_selection.py', 'tests/test_rd_equivalence.py',
    'docs/experiments/cross_dataset/2026-10-27/rd_equivalence/执行协议.md',
    'docs/experiments/cross_dataset/2026-10-27/rd_equivalence/执行修订.md')
_CONTEXT = None


def identity(selection):
    """接口：最终源帧身份由真实时钟和有序源编号/PTS决定，不混用跨池候选编号。"""
    return dict(video={k:selection['video'][k] for k in ('path','start_pts','time_base','duration_fraction')},
                frames=[[f['source_frame_index'],f['source_pts']] for f in selection['selected_frames']])


def file_hash(path):
    """接口：只缓存同一大小/mtime的已读文件摘要；来源变化不得命中旧摘要。"""
    path=Path(path);stat=path.stat()
    return _cached_hash(str(path),stat.st_size,stat.st_mtime_ns)


@lru_cache(maxsize=1024)
def _cached_hash(path,size,mtime):
    return sha256(path)


def freeze(path,value):
    """接口：已有协议/结果只接受字节语义一致，不静默覆写完成产物。"""
    path=Path(path)
    if path.exists():
        if read_json(path)!=value:raise ValueError('Frozen record changed: '+str(path))
    else:durable(path,value)


def checkpoint_paths(dataset,q):
    """接口：按既有完成运行定位权威检查点，不按历史对错选择来源。"""
    if dataset=='videomme':
        return [BASE/'selection_checkpoints'/(q+'.json'),RUN8/'inputs/checkpoints'/(q+'.json'),
                POOL16/'inputs/checkpoints'/(q+'.json'),RUN16/'inputs/checkpoints'/(q+'.json')]
    return [MLVU/'checkpoints'/p/(q+'.json') for p in ('llava8','qwen32')]


def jobs():
    """接口：仅投影冻结清单的题号/视频号/序号，不将标签传给选择器。"""
    data=rows();ml=read_json(MLVU/'manifest.json')['rows']
    if len(data)!=2700 or len(ml)!=2174:raise ValueError('Frozen population differs')
    out=[]
    for dataset,records in [('videomme',data),('mlvu',ml)]:
        if len({r['question_id'] for r in records})!=len(records):raise ValueError('Duplicate question ID')
        for ordinal,row in enumerate(records):
            q=row['question_id']
            if not q or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in q):
                raise ValueError('Unsafe question ID')
            out.append(dict(dataset=dataset,question_id=q,video_id=row['video_id'],ordinal=ordinal))
    return out


def prepare(root):
    """接口：冻结两数据集清单、检查点、代码和CPU环境，第一项选择前发布。"""
    # 步骤1固定问题身份及旧检查点；步骤2把代码/环境绑定到独立新运行。
    root=Path(root);root.mkdir(parents=True,exist_ok=True);data=jobs()
    checkpoint_hashes={str(p):file_hash(p) for j in data for p in checkpoint_paths(j['dataset'],j['question_id'])}
    spec=dict(implementation_id=production.IMPLEMENTATION_ID,methods=list(METHODS),jobs=data,
        expected_comparisons=24896,scopes=copy.deepcopy(SCOPES),
        manifests={str(p):file_hash(p) for p in (MANIFEST,MLVU/'manifest.json')},checkpoints=checkpoint_hashes,
        code_sha256={p:sha256(ROOT/p) for p in FILES},
        environment=dict(python=sys.version,numpy=np.__version__,executable=sys.executable,
            threads=2,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
        calls=dict(blip=0,visual_encoder=0,auxiliary_generation=0,final_qa=0,training=0,decode=0))
    path=root/'protocol.json'
    if path.exists():
        old=read_json(path)
        if {k:v for k,v in old.items() if k!='created_utc'}!=spec:raise ValueError('Protocol/code/source fingerprint changed')
        return old
    spec['created_utc']=utc();freeze(path,spec)
    import shutil
    for name in FILES:
        target=root/'code_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    return spec


def _bound(path,checkpoint,source_root,sources):
    """接口：实际读取字节必须匹配旧完成检查点，而非仅接受当前文件自身哈希。"""
    expected=read_json(checkpoint)['files'].get(str(Path(path).relative_to(source_root)))
    actual=file_hash(path)
    if expected is None or actual!=expected:raise ValueError('Historical asset binding differs: '+str(path))
    sources[str(path)]=actual


def _selection(path,sources):
    sources[str(path)]=file_hash(path)
    return read_json(path)


def mapped_features(pool,base_pool,features):
    """接口：K8/K16只按原映射取行，并独立核对每行对应的源身份。"""
    mapping=pool['original_candidate_indices']
    if len(mapping)!=len(pool['candidates']):raise ValueError('Feature mapping size differs')
    for i,j in enumerate(mapping):
        if not isinstance(j,int) or not 0<=j<len(base_pool['candidates']):raise ValueError('Invalid feature row mapping')
        a,b=pool['candidates'][i],base_pool['candidates'][j]
        if any(a[k]!=b[k] for k in ('source_frame_index','source_pts')):raise ValueError('Feature row source identity differs')
    return features[mapping]


def load_scope(job,k,base_pool=None,base_features=None):
    """接口：返回合法池/特征/两方法冻结轨迹与源帧；不读取任何答案文件。"""
    # 步骤1按数据集和预算定位旧资产；步骤2核验绑定/行映射；步骤3读取冻结科学量。
    dataset,q=job['dataset'],job['question_id'];sources={};extra=[]
    if dataset=='mlvu':
        profile='llava8' if k==8 else 'qwen32';cp=MLVU/'checkpoints'/profile/(q+'.json')
        pp=MLVU/'pools'/profile/(q+'.json');fp=MLVU/'features'/profile/(q+'.npy');tp=MLVU/'traces'/profile/(q+'.json')
        for p in (pp,fp,tp):_bound(p,cp,MLVU,sources)
        pool=read_json(pp);f=np.load(fp,allow_pickle=False);traces=read_json(tp);gold={}
        for method,key in zip(METHODS,('exp07','exp12')):
            path=MLVU/'selections'/profile/method/(q+'.json');_bound(path,cp,MLVU,sources)
            gold[method]=dict(trace=traces[key],selection=read_json(path)['selection'])
        return pool,f,gold,sources,extra
    fp=BASE/'features'/(q+'.npy');cp=BASE/'selection_checkpoints'/(q+'.json')
    _bound(fp,cp,BASE,sources)
    if k==8:
        cp=RUN8/'inputs/checkpoints'/(q+'.json');pp=RUN8/'inputs/pools'/(q+'.json');tp=RUN8/'inputs/traces'/(q+'.json')
        for p in (pp,tp):_bound(p,cp,RUN8,sources)
        pool=read_json(pp);f=mapped_features(pool,base_pool,base_features);traces=read_json(tp);gold={}
        for method,key in zip(METHODS,('exp07','exp12')):
            path=RUN8/'selections'/method/(q+'.json');_bound(path,cp,RUN8,sources)
            gold[method]=dict(trace=traces[key],selection=read_json(path)['selection'])
    elif k==16:
        cp=POOL16/'inputs/checkpoints'/(q+'.json');pp=POOL16/'inputs/pools'/(q+'.json');tp=POOL16/'inputs/traces'/(q+'.json');sp=POOL16/'inputs/m16'/(q+'.json')
        for p in (pp,tp,sp):_bound(p,cp,POOL16,sources)
        pool=read_json(pp);f=mapped_features(pool,base_pool,base_features)
        cp7=RUN16/'inputs/checkpoints'/(q+'.json');t7=RUN16/'inputs/traces'/(q+'.json');s7=RUN16/'inputs/selections'/(q+'.json')
        for p in (t7,s7):_bound(p,cp7,RUN16,sources)
        gold={METHODS[0]:dict(trace=read_json(t7),selection=read_json(s7)['selection']),
              METHODS[1]:dict(trace=read_json(tp),selection=read_json(sp)['selection'])}
        for method,origin in [(METHODS[0],RUN16),(METHODS[1],POOL16)]:
            common_rgb=None
            for profile in ('qwen16','llava16'):
                path=origin/profile/'selections'/method/(q+'.json');value=_selection(path,sources)
                if identity(value['selection'])!=identity(gold[method]['selection']):raise ValueError('Cross-model K16 source differs')
                if common_rgb is not None and common_rgb!=value['rgb_sha256']:raise ValueError('Cross-model K16 RGB differs')
                common_rgb=value['rgb_sha256'];extra.append(str(path))
    else:
        pp=BASE/'pools'/(q+'.json');_bound(pp,cp,BASE,sources);pool=base_pool;f=base_features;gold={}
        for method,origin in [(METHODS[0],RUN32),(METHODS[1],WEIGHTED32)]:
            sp=origin/'selections'/method/(q+'.json');value=_selection(sp,sources);tp=origin/value['trace_file']
            if file_hash(tp)!=value['trace_sha256']:raise ValueError('Frozen trace binding differs')
            sources[str(tp)]=file_hash(tp);gold[method]=dict(trace=read_json(tp),selection=value['selection'])
    return pool,f,gold,sources,extra


@contextmanager
def count_dp():
    """接口：记录实际旧/新求解器调用；审计包装只在当前CPU进程内生效。"""
    modules=(legacy_prefix,legacy_weighted,production);original=[m.allocate_prefix_budget for m in modules];counter=[0]
    def wrap(fn):
        def counted(*args,**kwargs):counter[0]+=1;return fn(*args,**kwargs)
        return counted
    for module,fn in zip(modules,original):module.allocate_prefix_budget=wrap(fn)
    try:yield counter
    finally:
        for module,fn in zip(modules,original):module.allocate_prefix_budget=fn


def _legacy(pool,f,k,method):
    """接口：旧链仅在审计控制中执行，exp12控制包含原等权前置链。"""
    selected,prefix=select_exp07(pool,f,k)
    if method==METHODS[0]:return selected,prefix,prefix
    selected,trace=legacy_weighted.select_relevance_coverage(pool,f,prefix)
    return selected,trace,prefix


def compare_outputs(pool,old,new,gold,prefix):
    """接口：严格比较全部离散量与源帧；数值共同科学量允许绝对误差1e-6。"""
    # 步骤1核对预算/源身份；步骤2逐区域核对前缀、边界与数值递推。
    old_ids,old_trace=old;new_ids,new_trace=new;error=0.
    if old_ids!=new_ids or old_trace['region_budgets']!=new_trace['region_budgets']:raise ValueError('Discrete selection/quota differs')
    actual=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in new_ids])
    if identity(actual)!=identity(gold['selection']):raise ValueError('Frozen final source/order differs')
    if new_trace['seed_indices']!=pool['plan']['seed_indices']:raise ValueError('Seed identity differs')
    if new_trace['protected_indices']!=[r['anchor_index'] for r in pool['plan']['regions']]:raise ValueError('Anchor identity differs')
    def numeric(a,b):
        nonlocal error
        a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Numeric shape/nonfinite mismatch')
        distance=float(np.max(np.abs(a-b))) if a.size else 0.;error=max(error,distance)
        if distance>1e-6:raise ValueError('Numerical difference exceeds 1e-6: '+str(distance))
    if old_trace['region_budgets']!=gold['trace']['region_budgets']:raise ValueError('Legacy/frozen quota differs')
    numeric(old_trace['objective'],new_trace['objective']);numeric(old_trace['objective'],gold['trace']['objective'])
    if len(new_trace['regions'])!=len(prefix['regions']) or len(new_trace['regions'])!=len(gold['trace']['regions']):raise ValueError('Region count differs')
    for p,a,b,frozen,expected in zip(pool['plan']['regions'],new_trace['regions'],prefix['regions'],gold['trace']['regions'],old_trace['regions']):
        for key in ('region_id','members','reference_coarse_indices','prefix','density_weight'):
            if a[key]!=b[key]:raise ValueError('Prefix or membership differs: '+key)
        if a['original_window']!=[p['left_fraction'],p['right_fraction']]:raise ValueError('Original window differs')
        if a['seed_indices']!=p['seed_indices']:raise ValueError('Region seeds differ')
        if a['prefix']!=frozen['prefix'] or a['reference_coarse_indices']!=frozen['reference_coarse_indices']:raise ValueError('Frozen prefix/reference differs')
        numeric(a['coverage_curve'],expected['coverage_curve']);numeric(a['coverage_curve'],frozen['coverage_curve'])
        if 'reference_weights' in expected:numeric(a['reference_weights'],expected['reference_weights'])
        if len(a['local_steps'])!=len(b['local_steps']):raise ValueError('Step count differs')
        for x,y in zip(a['local_steps'],b['local_steps']):
            if x['candidate_index']!=y['candidate_index']:raise ValueError('Local winner differs')
            for key in ('support','redundancy','local_score'):numeric(x[key],y[key])
    return error


def initialize(root,stop):
    """接口：每个CPU工作进程固定线程/GPU禁用，缓存本轮协议。"""
    global _CONTEXT
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    _CONTEXT=dict(root=Path(root),stop=stop,spec=read_json(Path(root)/'protocol.json'))


def verify_record(path):
    """接口：完成项只核验来源与记录摘要；不执行选择或重写结果。"""
    path=Path(path);value=read_json(path);payload={k:v for k,v in value.items() if k!='record_sha256'}
    # 步骤1身份必须与结果槽位一致，防止另一题的合法记录被误当作缓存命中。
    if path.parent.parent.parent.name=='records':
        if (value.get('dataset')!=path.parent.parent.name or value.get('question_id')!=path.stem
                or path.parent.name!='K'+str(value.get('frame_budget'))):
            raise ValueError('Completed record question/budget identity differs from slot')
        if set(value.get('methods',{}))!=set(METHODS):raise ValueError('Completed method scope differs')
    # 步骤2先核对记录自身，再核对原输入字节；两者均成立才允许跳过选择。
    if digest(payload)!=value['record_sha256']:raise ValueError('Completed record changed')
    for p,h in value['sources'].items():
        if file_hash(p)!=h:raise ValueError('Completed source changed: '+p)
    return value


def run_question(job):
    """接口：按题共享特征，两个方法及追踪模式逐项对照，失败即共享停止。"""
    root,stop,spec=(_CONTEXT[k] for k in ('root','stop','spec'));q=job['question_id'];dataset=job['dataset']
    budgets=spec['scopes'][dataset]['budgets'];new_count=reused=0
    try:
        # 步骤1先验证完成槽位和旧来源，整题命中时不加载特征数组或调用选择器。
        for cp in checkpoint_paths(dataset,q):
            if file_hash(cp)!=spec['checkpoints'][str(cp)]:raise ValueError('Source checkpoint changed')
        paths=[root/'records'/dataset/('K'+str(k))/(q+'.json') for k in budgets]
        if all(p.exists() for p in paths):
            for p in paths:verify_record(p)
            return dict(new=0,reused=2*len(paths))
        base_pool=base_features=None;shared_started=time.perf_counter()
        if dataset=='videomme':
            cp=BASE/'selection_checkpoints'/(q+'.json');bindings={}
            _bound(BASE/'pools'/(q+'.json'),cp,BASE,bindings);_bound(BASE/'features'/(q+'.npy'),cp,BASE,bindings)
            base_pool=read_json(BASE/'pools'/(q+'.json'));base_features=np.load(BASE/'features'/(q+'.npy'),allow_pickle=False)
        shared_read_seconds=time.perf_counter()-shared_started
        # 步骤2按预算加载一次合法池，分别执行旧链/新链并实际统计DP调用。
        for k,path in zip(budgets,paths):
            if stop.is_set():return dict(new=new_count,reused=reused)
            if path.exists():verify_record(path);reused+=2;continue
            began=time.perf_counter();pool,f,gold,sources,extra=load_scope(job,k,base_pool,base_features)
            for cp in checkpoint_paths(dataset,q):sources[str(cp)]=file_hash(cp)
            loaded=time.perf_counter();methods={}
            for method in METHODS:
                if stop.is_set():return dict(new=new_count,reused=reused)
                values={};timing={};counts={}
                order=('old','new') if job['ordinal']%2==0 else ('new','old')
                for kind in order:
                    with count_dp() as counter:
                        t=time.perf_counter()
                        if kind=='old':values[kind]=_legacy(pool,f,k,method)
                        else:values[kind]=production.select_rd(pool,f,k,method,'full')
                        timing[kind]=time.perf_counter()-t;counts[kind]=counter[0]
                old_ids,old_trace,prefix=values['old'];new_ids,new_trace=values['new']
                error=compare_outputs(pool,(old_ids,old_trace),(new_ids,new_trace),gold[method],prefix)
                # 步骤3独立核对摘要模式，只有两模式及历史均一致才原子发布预算记录。
                with count_dp() as counter:
                    t=time.perf_counter();summary_ids,summary=production.select_rd(pool,f,k,method);summary_seconds=time.perf_counter()-t
                if counts['new']!=1 or counter[0]!=1 or counts['old']!=(1 if method==METHODS[0] else 2):raise ValueError('Unexpected DP call count')
                if summary_ids!=new_ids or summary!={key:value for key,value in new_trace.items() if key not in ('regions','seed_indices','eligible_indices')}:
                    raise ValueError('Trace mode changed selection or objective')
                methods[method]=dict(selected_indices=new_ids,trace=new_trace,maximum_numeric_error=error,
                    legacy_seconds=timing['old'],new_full_seconds=timing['new'],new_summary_seconds=summary_seconds,
                    legacy_dp_calls=counts['old'],new_full_dp_calls=counts['new'],new_summary_dp_calls=counter[0])
            value=dict(dataset=dataset,question_id=q,video_id=job['video_id'],frame_budget=k,sources=sources,
                cross_model_k16_sources=extra,methods=methods,asset_read_seconds=loaded-began,
                shared_question_asset_read_seconds=shared_read_seconds,
                wall_seconds=time.perf_counter()-began,worker_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                new_model_calls=0,new_decode_calls=0,utc=utc())
            value['record_sha256']=digest(value);freeze(path,value);new_count+=2
        return dict(new=new_count,reused=reused)
    except BaseException as error:
        # 步骤4反例只首次落盘，共享停止信号禁止继续提交新问题，不调整规则回退。
        stop.set()
        with exclusive(root/'first_failure.lock'):
            if not (root/'failure.json').exists():
                durable(root/'failure.json',dict(job=job,error=repr(error),traceback=traceback.format_exc(),
                    current_budget=locals().get('k'),current_method=locals().get('method'),
                    sources=locals().get('sources',{}),old_new=locals().get('values',{}),utc=utc()))
        raise


def summarize(root):
    """接口：完整24896项才汇总，不把部分结果或模式检查计成更多问题。"""
    # 步骤1逐槽位和来源核验；步骤2按固定范围汇总，不以文件总数替代身份检查。
    root=Path(root);spec=read_json(root/'protocol.json');groups=[];files={}
    for job in spec['jobs']:
        for k in spec['scopes'][job['dataset']]['budgets']:
            path=root/'records'/job['dataset']/('K'+str(k))/(job['question_id']+'.json')
            value=verify_record(path);groups.append(value);files[str(path.relative_to(root))]=file_hash(path)
    if sum(len(v['methods']) for v in groups)!=24896:raise ValueError('Incomplete comparison matrix')
    metrics={}
    for dataset,scope in spec['scopes'].items():
        for k in scope['budgets']:
            values=[v for v in groups if v['dataset']==dataset and v['frame_budget']==k]
            if len(values)!=scope['questions']:raise ValueError('Incomplete dataset/budget population')
            for method in METHODS:
                data=[v['methods'][method] for v in values]
                metrics[dataset+'_K'+str(k)+'_'+method]=dict(n=len(data),maximum_numeric_error=max(v['maximum_numeric_error'] for v in data),
                    legacy_seconds=distribution([v['legacy_seconds'] for v in data]),
                    new_full_seconds=distribution([v['new_full_seconds'] for v in data]),new_summary_seconds=distribution([v['new_summary_seconds'] for v in data]),
                    legacy_dp_calls=sum(v['legacy_dp_calls'] for v in data),new_full_dp_calls=sum(v['new_full_dp_calls'] for v in data),
                    new_summary_dp_calls=sum(v['new_summary_dp_calls'] for v in data),
                    worker_peak_rss_kib_max=max(v['worker_peak_rss_kib'] for v in values))
    result=dict(status='passed',comparisons=24896,group_records=len(groups),metrics=metrics,files=files,
        maximum_numeric_error=max(v['maximum_numeric_error'] for g in groups for v in g['methods'].values()),
        source_bindings=digest({p:h for g in groups for p,h in g['sources'].items()}),
        source_code_sha256=spec['code_sha256'],calls=spec['calls'],new_method_version=False,utc=utc())
    if (root/'summary.json').exists():result['utc']=read_json(root/'summary.json')['utc']
    freeze(root/'summary.json',result)
    lines=['# RD等价执行链重放结果','','| 范围 | 方法 | n | 最大数值误差 | 旧DP次数 | 新单次DP次数 | 旧均值秒 | 新摘要均值秒 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for name,value in metrics.items():
        lines.append(f"| {name.split('_RD-')[0]} | RD-{name.split('_RD-')[1]} | {value['n']} | {value['maximum_numeric_error']:.3g} | {value['legacy_dp_calls']} | {value['new_summary_dp_calls']} | {value['legacy_seconds']['mean']:.6f} | {value['new_summary_seconds']['mean']:.6f} |")
    lines+=['','所有离散量严格一致；数值容差1e-6。完整模式与摘要模式各自一次DP，两个模式检查不增加问题分母。',
        'CPU选择计时排除来源读取，旧/新先后交替。RSS是工作进程累计高水位，不能解释为单函数增量内存。',
        '旧输入对应的答案和计分继续引用历史记录，本轮没有新QA准确率或原片E2E结果。','']
    report='\n'.join(lines)
    if (root/'report.md').exists():
        if (root/'report.md').read_text()!=report:raise ValueError('Report content changed')
    else:(root/'report.md').write_text(report)
    return result


def verify_protocol(root):
    """接口：恢复前核对代码、清单和检查点；任何漂移都暂停，不隐式迁移。"""
    spec=read_json(Path(root)/'protocol.json')
    # 步骤1派生范围仍须与固定任务一致；步骤2核对环境、代码及原检查点指纹。
    if (spec['methods']!=list(METHODS) or spec['scopes']!=SCOPES or spec['expected_comparisons']!=24896
            or spec['implementation_id']!=production.IMPLEMENTATION_ID):raise ValueError('Frozen method/matrix scope changed')
    if spec['jobs']!=jobs():raise ValueError('Frozen question identities changed')
    expected=dict(python=sys.version,numpy=np.__version__,executable=sys.executable,
        threads=2,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if spec['environment']!=expected:raise ValueError('CPU environment changed')
    for rel,expected in spec['code_sha256'].items():
        if sha256(ROOT/rel)!=expected:raise ValueError('Code changed: '+rel)
    for path,expected in {**spec['manifests'],**spec['checkpoints']}.items():
        if file_hash(path)!=expected:raise ValueError('Source fingerprint changed: '+path)
    return spec


def audit_completed(root):
    """接口：完成态只验证现存文件/来源，无进程池、无选择、无产物重写。"""
    # 步骤1完成标记绑定原协议与所有收尾产物；步骤2逐槽位验证，不进入计算路径。
    root=Path(root);spec=verify_protocol(root);done=read_json(root/'completion.json');summary=read_json(root/'summary.json')
    if (done['comparisons']!=24896 or done['summary_sha256']!=file_hash(root/'summary.json')
            or done['protocol_sha256']!=file_hash(root/'protocol.json')):raise ValueError('Completion binding differs')
    for name,h in done['final_artifacts'].items():
        if file_hash(root/name)!=h:raise ValueError('Finalization artifact changed')
    expected={str(Path('records')/j['dataset']/('K'+str(k))/(j['question_id']+'.json'))
              for j in spec['jobs'] for k in spec['scopes'][j['dataset']]['budgets']}
    if set(summary['files'])!=expected:raise ValueError('Completed result slots differ')
    for relative,expected in summary['files'].items():
        if file_hash(root/relative)!=expected:raise ValueError('Result bytes changed')
        verify_record(root/relative)
    if len(summary['files'])!=12448 or summary['calls']!=spec['calls']:raise ValueError('Completion counts differ')
    return dict(status='already_completed',comparisons=24896,new_comparisons=0,new_selection_calls=0,new_model_calls=0)


def execute(root,workers=8):
    """接口：有界提交CPU任务，首个反例停止新提交，持续落盘心跳和恢复检查点。"""
    root=Path(root)
    with exclusive(root/'run.lock'):
        # 步骤1完整恢复仅做审计；未完成任务才初始化有界CPU池和工作账本。
        if (root/'completion.json').exists():return audit_completed(root)
        spec=prepare(root);verify_protocol(root)
        if (root/'failure.json').exists():raise ValueError('Prior failure preserved; investigate before creating a corrected run')
        durable(root/'launcher.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        context=mp.get_context('spawn');stop=context.Event();began=time.perf_counter();new=reused=finished=0
        iterator=iter(spec['jobs']);pending={}
        with ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=initialize,initargs=(str(root),stop)) as executor:
            def submit():
                job=next(iterator,None)
                if job is not None:pending[executor.submit(run_question,job)]=job
            for _ in range(workers):submit()
            # 步骤2最多保留workers个在途任务，首个异常停止后续提交并保留反例。
            while pending:
                complete,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                try:
                    for future in complete:
                        pending.pop(future);value=future.result();new+=value['new'];reused+=value['reused'];finished+=1
                    if stop.is_set():raise RuntimeError('Worker signalled first mismatch')
                except BaseException:
                    stop.set()
                    for future in pending:future.cancel()
                    durable(root/'status.json',dict(status='paused',new_comparisons=new,reused_comparisons=reused,utc=utc()));raise
                for _ in complete:submit()
                elapsed=time.perf_counter()-began
                durable(root/'status.json',dict(status='running',pid=os.getpid(),sid=os.getsid(0),
                    completed_questions=finished,total_questions=len(spec['jobs']),completed_comparisons=new+reused,total_comparisons=24896,
                    active=[v['dataset']+':'+v['question_id'] for v in pending.values()],seconds=elapsed,
                    eta_seconds=elapsed/max(finished,1)*(len(spec['jobs'])-finished),utc=utc()))
        # 步骤3先发布摘要/计时/终态；步骤4最后发布绑定全部收尾文件的完成标记。
        result=summarize(root)
        if not (root/'execution.json').exists():
            durable(root/'execution.json',dict(status='completed',wall_seconds=time.perf_counter()-began,
                new_comparisons=new,reused_comparisons=reused,new_selector_calls=2*new,legacy_chain_calls=new,
                workers=workers,new_model_calls=0,utc=utc()))
        durable(root/'status.json',dict(status='completed',completed_comparisons=24896,utc=utc()))
        freeze(root/'completion.json',dict(status='passed',comparisons=24896,record_files=len(result['files']),
            summary_sha256=file_hash(root/'summary.json'),protocol_sha256=file_hash(root/'protocol.json'),
            final_artifacts={p:file_hash(root/p) for p in ('report.md','execution.json','status.json')},calls=spec['calls']))
        return result
