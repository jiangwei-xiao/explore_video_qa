"""exp33固定选择的四单元问答：允许已知证据风险，严格冻结输入与一次生成。"""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import copy
import fcntl
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import numpy as np

from videoqa_runtime.common import ROOT,read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,digest,invoke_once,utc,rows
from videoqa_full.prepare import environment
from videoqa_budget32.frames import FrameProvider,validate_frames
from videoqa_audit.frame_group_diagnostic import (
    cpu_input,observed_input,canonical_option,history_sources,task_backend,observe_answer,restore_answer,synthetic_input,
)
from videoqa_audit.region_scope_runtime import PYTHONS,QA_FILES
from videoqa_audit.framework_contraction import RUN8,RUN32
from videoqa_audit.adjacent_context_experiment import REVIEW,DIAGNOSTIC,PROFILES,PARENT,METHOD

SOURCE=ROOT/'outputs/methods/rd12_exp33_adjacent_context_20261015_r2'
ML=ROOT/'outputs/full_videoqa/mlvu_dev__S1__llava8_qwen32__four_methods__20261014_r1'
CELLS=(('videomme','llava8'),('videomme','qwen32'),('mlvu','llava8'),('mlvu','qwen32'))
EXPECTED={('videomme','llava8'):(200,84,1,115),('videomme','qwen32'):(200,4,0,196),
          ('mlvu','llava8'):(210,102,0,108),('mlvu','qwen32'):(210,4,0,206)}
FILES=tuple(dict.fromkeys(('src/videoqa_audit/adjacent_context_development.py','scripts/run_adjacent_context_development.py',
    'src/videoqa_audit/frame_group_diagnostic.py','src/videoqa_mlvu/backends.py','src/videoqa_mlvu/data.py',
    'src/videoqa_scorer_comparison/runtime.py','src/videoqa_budget32/frames.py','src/videoqa_audit/cross_dataset_review.py')+QA_FILES))
_TOOLS=None


@contextmanager
def cache_lock(path):
    """接口：共享身份缓存等待已有计算完成；与防重复启动的非阻塞排他锁职责不同。"""
    # 步骤1：不同单元可以请求同一视频/模型缓存，不能把正常等待当成运行失败。
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        yield


def unit_path(root,dataset,profile):
    """接口：数据集与模型预算共同隔离账本，历史离线目录永不作为写入目标。"""
    if (dataset,profile) not in CELLS:raise ValueError('Unknown fixed evaluation cell')
    return Path(root)/'units'/dataset/profile


def data_rows(dataset):
    """接口：只投影已冻结200/210题身份，不按历史对错选择任务。"""
    ids=read_json(SOURCE/'development_ids.json')[dataset]
    if dataset=='mlvu':
        from videoqa_mlvu.data import rows as mlrows
        bank={r['question_id']:r for r in mlrows(ML)}
    else:bank={r['question_id']:r for r in rows()}
    if len(ids)!=len(set(ids)) or len(ids)!=(200 if dataset=='videomme' else 210):raise ValueError('Frozen subset differs')
    return [bank[q] for q in ids]


def media_key(selection):
    """接口：源视频时钟及有序源身份一致才进入历史输入核验，不以答案选择来源。"""
    v=selection['video']
    return dict(video={k:v[k] for k in ('path','start_pts','time_base','duration_fraction')},
                frames=[[f['source_pts'],f['source_frame_index']] for f in selection['selected_frames']])


def check_partition(dataset,profile,items):
    """接口：事前195引用/625新调用必须精确闭合，异常不能自动转成更多生成。"""
    counts=tuple(sum(x['origin']==kind for x in items) for kind in ('parent_import','diagnostic_import','new_generation'))
    n,parent,diagnostic,new=EXPECTED[(dataset,profile)]
    if len(items)!=n or counts!=(parent,diagnostic,new):raise ValueError('Frozen input/import/call partition differs')
    return dict(results=n,parent_imports=parent,diagnostic_imports=diagnostic,new_calls=new)


def preflight(root):
    """接口：记录本轮带风险小测授权并验证旧离线产物；不改旧失败门槛。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);offline_environment()
    with cache_lock(root/'preflight.lock'):
        path=root/'preflight.json'
        if path.exists():
            spec=read_json(path)
            for n,h in spec['code'].items():
                if sha256(ROOT/n)!=h:raise ValueError('Development code changed')
            if sha256(SOURCE/'offline_summary.json')!=spec['offline_summary_sha256']:raise ValueError('Frozen selections changed')
            return spec
        from videoqa_audit.adjacent_context_experiment import verify_offline
        # 步骤1：旧门槛保留false，新协议只调整是否进入小测，不修订任何选择。
        verify_offline(SOURCE);oldgate=read_json(SOURCE/'offline_gate.json')
        if oldgate['passed'] is not False:raise ValueError('Historical offline decision changed')
        for ds in ('videomme','mlvu'):data_rows(ds)
        spec=dict(method=METHOD,parent=PARENT,source_run=str(SOURCE),offline_summary_sha256=sha256(SOURCE/'offline_summary.json'),
            original_gate_sha256=sha256(SOURCE/'offline_gate.json'),allow_known_evidence_risk=True,
            risk_case='videomme/llava8/012-3',expected_new_calls=625,max_synthetic_calls=8,
            cells=[list(c) for c in CELLS],code={p:sha256(ROOT/p) for p in FILES},utc=utc())
        # 步骤2：快照仅保存执行代码；后续文档/登记状态更新不改变推理指纹。
        for name in FILES:
            dest=root/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        durable(path,spec);return spec


def model_identity(root,profile):
    """接口：每模型只校验一次完整权重/环境，两个数据集共享只读身份检查点。"""
    from videoqa_lmms_bridge.common import framework_identity
    root=Path(root);path=root/'model_identity'/(profile+'.json')
    with cache_lock(path.with_suffix('.lock')):
        if path.exists():
            s=read_json(path)
            if environment(PYTHONS[profile])!=s['environment']:raise ValueError('QA environment changed')
            for p,v in s['model_files'].items():
                st=Path(p).stat()
                if (st.st_size,st.st_mtime_ns)!=(v['bytes'],v['mtime_ns']):raise ValueError('Model stat changed')
            return s
        parent=(RUN32 if profile=='qwen32' else RUN8)/'protocol.json';old=read_json(parent)
        if environment(PYTHONS[profile])!=old['environment'] or framework_identity()!=old['framework']:
            raise ValueError('Historical backend identity differs')
        # 步骤1：共享模型只读；检查当前完整字节，不升级依赖或注意力实现。
        for p,v in old['model_files'].items():
            if sha256(p)!=v['sha256']:raise ValueError('Frozen model bytes changed')
        for n in QA_FILES:
            if n in old['code_sha256'] and sha256(ROOT/n)!=old['code_sha256'][n]:raise ValueError('Frozen inference source changed')
        s={k:old[k] for k in ('model','model_files','framework','environment')}
        s.update(parent_protocol=str(parent),parent_protocol_sha256=sha256(parent),utc=utc())
        durable(path,s);return s


def video_identity(root,dataset,row,selection):
    """接口：同一源视频跨预算只哈希一次；后续逐题核验stat与实际源RGB。"""
    root=Path(root);video=Path(selection['video']['path']);path=root/'video_identity'/(digest(str(video))+'.json')
    with cache_lock(path.with_suffix('.lock')):
        if path.exists():
            value=read_json(path);st=video.stat()
            if (st.st_size,st.st_mtime_ns)!=(value['bytes'],value['mtime_ns']):raise ValueError('Video stat changed')
            return path
        if dataset=='mlvu':expected=read_json(ML/'manifest.json')['video_hashes'][str(video)]
        else:expected=read_json(RUN32/'protocol.json')['video_hashes'][row['video_id']]
        st=video.stat();actual=sha256(video)
        if actual!=expected:raise ValueError('Historical source video changed')
        durable(path,dict(path=str(video),sha256=actual,bytes=st.st_size,mtime_ns=st.st_mtime_ns));return path


def initialize_cpu(profile):
    """接口：spawn进程仅加载官方处理器/Tokenizer，明确无GPU，不加载模型权重。"""
    global _TOOLS
    offline_environment()
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from videoqa_scorer_comparison.runtime import make_tools
    _TOOLS=make_tools(profile)


def prepare_question(job):
    """接口：精确解码与官方CPU处理后，固定父版优先、诊断T次之、否则新生成。"""
    root,dataset,profile,row=Path(job[0]),job[1],job[2],job[3];run=unit_path(root,dataset,profile);q=row['question_id'];dest=run/'inputs'/(q+'.json')
    if dest.exists():
        old=read_json(dest)
        for p,h in old['sources'].items():
            if sha256(p)!=h:raise ValueError('Input checkpoint source changed')
        return old
    source=SOURCE/'offline'/dataset/profile/(q+'.json');candidate=read_json(source)
    expected=read_json(SOURCE/'offline_summary.json')['exports']['offline/'+dataset+'/'+profile+'/'+q+'.json']
    if sha256(source)!=expected:raise ValueError('Frozen exp33 selection differs')
    selection=candidate['selection'];validate_frames(selection,PROFILES[profile]);parentpath=Path(candidate['parent_result']);parent=read_json(parentpath)
    vpath=video_identity(root,dataset,row,selection);case=dict(dataset=dataset,profile=profile,question_id=q)
    # 步骤1：每个新输入都真正通过官方CPU处理，不依据预测进行引用。
    began=time.perf_counter();images,rgb=FrameProvider(selection).decode()
    try:identity=cpu_input(case,_TOOLS,images,row)
    finally:
        for im in images:im.close()
    item=dict(question_id=q,row=row,selection=selection,rgb_sha256=rgb,input_identity=identity,video_identity=str(vpath),
        origin='new_generation',import_answer=None,import_physical_gpu=None,
        changed=candidate['trace']['changed'],trace=candidate['trace'],qwen_pairs_before=candidate['qwen_pairs_before'],qwen_pairs_after=candidate['qwen_pairs_after'],
        sources={str(source):sha256(source),str(parentpath):sha256(parentpath),str(vpath):sha256(vpath)},prepare_seconds=time.perf_counter()-began)
    unchanged=[i for i in range(len(rgb)) if i not in candidate['trace']['changed_slots']]
    if any(rgb[i]!=parent['rgb_sha256'][i] for i in unchanged):raise ValueError('Unchanged source RGB differs')
    # 步骤2：同帧父版必须连实际像素/Token及完整物理调用一起吻合。
    if media_key(selection)==media_key(parent['selection']):
        if rgb!=parent['rgb_sha256'] or identity!=observed_input(profile,parent['answer']):raise ValueError('Parent same-source actual input differs')
        source_case=dict(case,method=PARENT,gpu=str(parent['gpu']))
        item.update(origin='parent_import',import_answer=parent['answer'],import_physical_gpu=str(parent['gpu']))
        item['sources'].update(history_sources(source_case,parentpath,parent))
    else:
        from videoqa_audit.frame_group_diagnostic import cases
        for diagnostic in cases():
            if (diagnostic['dataset'],diagnostic['profile'],diagnostic['question_id'])!=(dataset,profile,q):continue
            ip=DIAGNOSTIC/diagnostic['id']/'inputs/T.json';oldinput=read_json(ip)
            if media_key(selection)!=media_key(oldinput['selection']):continue
            if rgb!=oldinput['rgb_sha256'] or identity!=oldinput['input_identity']:raise ValueError('Diagnostic same-source actual input differs')
            base=ip.parent.parent;oldspec=read_json(base/'protocol.json');model=read_json(root/'model_identity'/(profile+'.json'))
            for k in ('model_files','framework','environment'):
                if oldspec[k]!=model[k]:raise ValueError('Diagnostic model/protocol identity differs')
            result=read_json(base/'results/T.json');cp=base/'calls/T'/(q+'.json');rp=base/'raw_generations/T'/(q+'.json')
            call,raw=read_json(cp),read_json(rp)
            if call['status']!='completed' or call['answer']!=result['answer'] or raw['generation_state']!={'started':True,'returned':True}:raise ValueError('Diagnostic return incomplete')
            item.update(origin='diagnostic_import',import_answer=result['answer'],import_physical_gpu=diagnostic['gpu'])
            for p in (ip,base/'protocol.json',base/'results/T.json',cp,rp):item['sources'][str(p)]=sha256(p)
            break
    durable(dest,item);return item


def prepare_unit(root,dataset,profile):
    """接口：在正确后端环境冻结一个单元的实际输入、引用来源和有限新调用清单。"""
    root=Path(root);run=unit_path(root,dataset,profile);run.mkdir(parents=True,exist_ok=True);began=time.perf_counter()
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():return verify_unit(run)
        preflight(root);model=model_identity(root,profile);data=data_rows(dataset)
        if dataset=='mlvu':
            ml=read_json(ML/'protocol.json')
            if ml['environments'][profile]!=model['environment']:raise ValueError('MLVU historical environment differs')
            for p,h in ml['task_files'].items():
                if sha256(p)!=h:raise ValueError('Native MLVU task changed')
        # 步骤1：两个CPU进程有界解码，全部820输入冻结后才允许首个真实调用。
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize_cpu,initargs=(profile,)) as executor:
            items=[]
            for index,item in enumerate(executor.map(prepare_question,[(str(root),dataset,profile,r) for r in data]),1):
                items.append(item);durable(run/'prepare_status.json',dict(completed=index,total=len(data),seconds=time.perf_counter()-began,utc=utc()))
        partition=check_partition(dataset,profile,items)
        spec=dict(method=METHOD,dataset=dataset,profile=profile,question_ids=[r['question_id'] for r in data],
            new_ids=[x['question_id'] for x in items if x['origin']=='new_generation'],partition=partition,
            inputs={r['question_id']:sha256(run/'inputs'/(r['question_id']+'.json')) for r in data},
            model_identity=model,root_preflight_sha256=sha256(root/'preflight.json'),utc=utc())
        durable(run/'protocol.json',spec)
        # 步骤2：引用只构造本轮结果，不调用生成或给引用伪造本轮QA耗时。
        for item in items:
            if item['origin']!='new_generation':persist_result(run,item,item['import_answer'],item['import_physical_gpu'])
        durable(run/'prepare_summary.json',dict(partition,seconds=time.perf_counter()-began,utc=utc()));return spec


def verify_unit(run):
    """接口：工作进程/恢复入口核验代码、环境和820条中的对应冻结输入。"""
    run=Path(run);root=run.parents[2];spec=read_json(run/'protocol.json');preflight(root)
    if sha256(root/'preflight.json')!=spec['root_preflight_sha256']:raise ValueError('Root protocol changed')
    if environment(PYTHONS[spec['profile']])!=spec['model_identity']['environment']:raise ValueError('Backend environment changed')
    for q,h in spec['inputs'].items():
        if sha256(run/'inputs'/(q+'.json'))!=h:raise ValueError('Frozen input file changed')
    for p,v in spec['model_identity']['model_files'].items():
        st=Path(p).stat()
        if (st.st_size,st.st_mtime_ns)!=(v['bytes'],v['mtime_ns']):raise ValueError('Model file stat changed')
    return spec


def validate_answer(item,answer,dataset,profile):
    """接口：模型真实像素/Token与CPU冻结相同，原生任务计分独立复算。"""
    if observed_input(profile,answer)!=item['input_identity']:raise ValueError('Actual model input differs')
    o=answer['observed'];n=len(item['selection']['selected_frames'])
    if o['source_frames']!=n or o['generate_calls']!=1 or o['prefill_calls']!=1 or not 1<=len(o['generated_token_ids'])<=16:raise ValueError('Actual generation observations differ')
    if profile=='qwen32':
        g=o['generation_kwargs']
        if g['max_new_tokens']!=16 or g['do_sample'] is not False or g['num_beams']!=1 or o['encoding_fps']!=2:raise ValueError('Qwen generation parameters differ')
        if o['encoded_frames']!=n+n%2 or o['internal_padding_frames']!=n%2:raise ValueError('Qwen temporal padding differs')
    elif o['visual_tokens']!=210*n or o['max_new_tokens']!=16:raise ValueError('LLaVA generation/visual budget differs')
    if dataset=='mlvu':
        from videoqa_mlvu.backends import metric
        score=metric(item['row'],[answer['raw_output']])
    else:
        from videoqa_lmms_bridge.common import load_task,task_doc
        score=load_task().videomme_process_results(task_doc(item['row']),[answer['raw_output']])['videomme_perception_score']
    if score!=answer['framework_metric'] or (score['pred_answer'] or None)!=(answer['parsed_answer'] or None):raise ValueError('Native scoring differs')


def persist_result(run,item,answer,gpu,timings=None):
    """接口：新生成与引用同口径计分，原生和MLVU标点归一化同时保留。"""
    run=Path(run);spec=read_json(run/'protocol.json');q=item['question_id'];dest=run/'results'/(q+'.json')
    validate_answer(item,answer,spec['dataset'],spec['profile'])
    option=canonical_option(spec,answer);gold='ABCD'[item['row']['answer_index']]
    value=dict(question_id=q,method=METHOD,dataset=spec['dataset'],profile=spec['profile'],selection=item['selection'],
        rgb_sha256=item['rgb_sha256'],answer=answer,option=option,native_correct=answer['parsed_answer']==gold,correct=option==gold,
        origin=item['origin'],changed=item['changed'],gpu=gpu,input_sha256=sha256(run/'inputs'/(q+'.json')),
        protocol_sha256=sha256(run/'protocol.json'),timings=timings or {},utc=utc())
    if not dest.exists():durable(dest,value)


def recover_call(run,q):
    """接口：只恢复完整原始返回后的计分，不确定调用永不自动重发。"""
    run=Path(run);cp=run/'calls'/METHOD/(q+'.json')
    if not cp.exists():return
    call=read_json(cp);rawpath=run/'raw_generations'/METHOD/(q+'.json');spec=read_json(run/'protocol.json');item=read_json(run/'inputs'/(q+'.json'))
    if call['protocol_sha256']!=sha256(run/'protocol.json'):raise ValueError('Call fingerprint changed')
    if call['status']=='completed':answer=call['answer']
    else:
        if not rawpath.exists():raise RuntimeError('Uncertain generation; no retry')
        raw=read_json(rawpath)
        if raw['identity']!=dict(question_id=q,method=METHOD,protocol_sha256=call['protocol_sha256'],gpu=call['gpu']) or raw['generation_state']!={'started':True,'returned':True}:raise RuntimeError('Raw return uncertain')
        answer=restore_answer(spec,raw,item['row']);call.update(status='completed',answer=answer,generation_state=raw['generation_state'],recovered_postprocessing=True);durable(cp,call)
    persist_result(run,item,answer,call['gpu'])


def process_question(run,q,backend,adapter,observer,gpu,spec):
    """接口：单题固定GPU、准确取帧、一次生成；任何未知状态停止而不换调用ID。"""
    run=Path(run);root=run.parents[2];item=read_json(run/'inputs'/(q+'.json'))
    with exclusive(run/'locks'/(q+'.lock')):
        recover_call(run,q)
        if (run/'results'/(q+'.json')).exists():return
        if q not in spec['new_ids'] or item['origin']!='new_generation':raise ValueError('Question not in frozen new-call plan')
        if (root/'PAUSE.json').exists():raise RuntimeError('Pipeline paused before new generation')
        # 步骤1：复核冻结媒体stat、实际RGB、官方预处理与完整提示。
        v=read_json(item['video_identity']);st=Path(v['path']).stat()
        if (st.st_size,st.st_mtime_ns)!=(v['bytes'],v['mtime_ns']):raise ValueError('Runtime source video stat changed')
        began=time.perf_counter();images,rgb=FrameProvider(item['selection']).decode()
        try:
            tools=(backend.processor,backend.tokenizer if spec['profile']=='llava8' else None)
            if rgb!=item['rgb_sha256'] or cpu_input(spec,tools,images,item['row'])!=item['input_identity']:raise ValueError('Runtime actual input differs')
            decoded=time.perf_counter();fp=sha256(run/'protocol.json')
            rawpath=run/'raw_generations'/METHOD/(q+'.json');identity=dict(question_id=q,method=METHOD,protocol_sha256=fp,gpu=gpu)
            # 步骤2：原文由冻结观测器在生成返回边界立即落盘，先于本轮后处理。
            def invoke(state):
                return observe_answer(spec,observer,backend,adapter,images,item['selection'],item['row'],state,rawpath,identity)
            answer,_,checkpoint=invoke_once(run,q,METHOD,fp,gpu,invoke)
        finally:
            for im in images:im.close()
        persist_result(run,item,answer,gpu,dict(final_decode_and_identity_seconds=decoded-began,
            cached_input_to_answer_seconds=time.perf_counter()-began,call_checkpoint_seconds=checkpoint))


def validate_result(run,q):
    """接口：结果、实际输入、引用来源或新物理账本闭合，独立重算正确性。"""
    run=Path(run);spec=read_json(run/'protocol.json');item=read_json(run/'inputs'/(q+'.json'));result=read_json(run/'results'/(q+'.json'))
    if result['input_sha256']!=sha256(run/'inputs'/(q+'.json')) or result['protocol_sha256']!=sha256(run/'protocol.json'):
        raise ValueError('Result/input protocol identity differs')
    if result['selection']!=item['selection'] or result['rgb_sha256']!=item['rgb_sha256'] or result['origin']!=item['origin']:
        raise ValueError('Result source identity differs')
    answer=result['answer'];validate_answer(item,answer,spec['dataset'],spec['profile']);gold='ABCD'[item['row']['answer_index']]
    if result['option']!=canonical_option(spec,answer) or result['correct']!=(result['option']==gold) or result['native_correct']!=(answer['parsed_answer']==gold):
        raise ValueError('Result correctness differs')
    if item['origin']=='new_generation':
        call=read_json(run/'calls'/METHOD/(q+'.json'));raw=read_json(run/'raw_generations'/METHOD/(q+'.json'))
        identity=dict(question_id=q,method=METHOD,protocol_sha256=sha256(run/'protocol.json'),gpu=result['gpu'])
        if call['status']!='completed' or call['answer']!=answer or raw['identity']!=identity or raw['generation_state']!={'started':True,'returned':True}:
            raise ValueError('Physical call/raw return differs')
    else:
        if answer!=item['import_answer']:raise ValueError('Imported answer changed')
        for path,h in item['sources'].items():
            if sha256(path)!=h:raise ValueError('Imported source changed')
    return result


def audit_unit(run):
    """接口：单元200/210结果和固定调用预算完整；完成恢复不加载模型。"""
    run=Path(run);spec=verify_unit(run)
    if {p.stem for p in (run/'results').glob('*.json')}!=set(spec['question_ids']):raise ValueError('Cell results incomplete')
    for q in spec['question_ids']:validate_result(run,q)
    calls=list((run/'calls'/METHOD).glob('*.json'))
    if {p.stem for p in calls}!=set(spec['new_ids']):raise ValueError('New-call budget/ID set differs')
    ledgers=[read_json(p) for p in (run/'calls').glob('*/*.json')]
    if any(c['status']!='completed' for c in ledgers):raise RuntimeError('Uncertain call remains')
    actual={str(p.relative_to(run)):sha256(p) for folder in ('calls','raw_generations','results') for p in (run/folder).rglob('*.json')}
    previous=run/'completion_audit.json'
    if previous.exists() and read_json(previous)['artifacts']!=actual:raise ValueError('Completion recovery changed generated artifacts')
    out=dict(status='passed',results=len(spec['question_ids']),new_calls=len(calls),imports=len(spec['question_ids'])-len(calls),
        synthetic_calls=sum(c['method'].startswith('synthetic_') for c in ledgers),artifacts=actual,utc=utc())
    durable(previous,out);durable(run/'status.json',dict(status='completed',results=len(spec['question_ids']),utc=utc()));return out


def assign_work(spec,items,gpus):
    """接口：固定首5题先行，其余按时长降序LPT分配，不读取正确性或新答案。"""
    plans={g:dict(pilot=[],body=[]) for g in gpus};load={g:0. for g in gpus};pilot=set(spec['question_ids'][:5])
    pending=[x for x in items if x['origin']=='new_generation']
    for phase,group in [('pilot',[x for x in pending if x['question_id'] in pilot]),
                        ('body',sorted([x for x in pending if x['question_id'] not in pilot],key=lambda x:(-x['selection']['video']['duration_seconds'],spec['question_ids'].index(x['question_id']))) )]:
        for item in group:
            g=min(gpus,key=lambda g:(load[g],int(g)));plans[g][phase].append(item['question_id']);load[g]+=item['selection']['video']['duration_seconds']
    flat=[q for p in plans.values() for phase in ('pilot','body') for q in p[phase]]
    if len(flat)!=len(set(flat)) or set(flat)!=set(spec['new_ids']):raise ValueError('Worker assignment does not close')
    return plans


def worker(root,dataset,profile,gpu):
    """接口：每张分配卡一个长期任务进程，合成一次后执行先行/主体，任务类型隔离。"""
    root=Path(root);run=unit_path(root,dataset,profile);spec=verify_unit(run);plan=read_json(run/'gpu_assignment.json')['plans'][gpu]
    session=run/'workers'/(gpu+'.json')
    with exclusive(run/'worker_locks'/(gpu+'.lock')):
        assigned=plan['pilot']+plan['body']
        if session.exists():
            for q in assigned:recover_call(run,q)
            if all((run/'results'/(q+'.json')).exists() for q in assigned):return
            raise RuntimeError('Previous worker process ended; inspect before replacement process or synthetic budget')
        import torch
        offline_environment();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(2027)
        durable(session,dict(status='loading',gpu=gpu,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        began=time.perf_counter()
        if profile=='qwen32':
            from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
            backend=QwenFA2VideoBackend()
        else:
            from videoqa_runtime.llava_backend import LlavaBackend
            backend=LlavaBackend()
        adapter,observer=task_backend(spec,backend)
        durable(run/'load_reports'/(gpu+'.json'),dict(gpu=gpu,pid=os.getpid(),seconds=time.perf_counter()-began,report=backend.load_report))
        # 步骤1：每个初始工作进程一次合成，ID包含数据集与物理GPU，防止全局碰撞。
        images,selection,data=synthetic_input(spec);method='synthetic_'+dataset+'_'+gpu;qid='warmup'
        fp=sha256(run/'protocol.json');rawpath=run/'raw_generations'/method/(qid+'.json')
        def invoke(state):
            return observe_answer(spec,observer,backend,adapter,images,selection,data,state,rawpath,
                dict(question_id=qid,method=method,protocol_sha256=fp,gpu=gpu))
        try:invoke_once(run,qid,method,fp,gpu,invoke)
        finally:
            for im in images:im.close()
        # 步骤2：先行5题均经过验收屏障，是否答对不参与放行。
        for q in plan['pilot']:
            if (root/'PAUSE.json').exists():return
            durable(session,dict(status='pilot',question_id=q,gpu=gpu,pid=os.getpid(),utc=utc()));process_question(run,q,backend,adapter,observer,gpu,spec)
        durable(run/'pilot_workers'/(gpu+'.json'),dict(status='ready',gpu=gpu,pid=os.getpid(),utc=utc()))
        while not (run/'pilot_acceptance.json').exists():
            if (root/'PAUSE.json').exists():return
            time.sleep(2)
        # 步骤3：完成时长降序分配的主体；任何全局暂停只允许在途生成返回落盘。
        for q in plan['body']:
            if (root/'PAUSE.json').exists():return
            durable(session,dict(status='running',question_id=q,gpu=gpu,pid=os.getpid(),utc=utc()));process_question(run,q,backend,adapter,observer,gpu,spec)
        durable(session,dict(status='completed',gpu=gpu,pid=os.getpid(),questions=len(assigned),utc=utc()))


def run_unit(root,dataset,profile,gpus):
    """接口：管理最多两张固定物理卡，先行验收后同进程接续；失败全局暂停。"""
    root=Path(root);run=unit_path(root,dataset,profile);spec=verify_unit(run)
    with exclusive(run/'manager.lock'):
        for q in spec['new_ids']:recover_call(run,q)
        if all((run/'results'/(q+'.json')).exists() for q in spec['question_ids']):return audit_unit(run)
        items=[read_json(run/'inputs'/(q+'.json')) for q in spec['question_ids']]
        allocation=run/'gpu_assignment.json'
        if allocation.exists():
            assignment=read_json(allocation)
            if list(assignment['plans'])!=gpus:raise ValueError('Resume moved physical GPU')
        else:
            assignment=dict(plans=assign_work(spec,items,gpus),uuids={g:subprocess.check_output(['nvidia-smi','-i',g,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip() for g in gpus})
            durable(allocation,assignment)
        children=[]
        try:
            # 步骤1：每个子进程只看一张物理卡，正确模型环境下不加载BLIP。
            for g in gpus:
                uuid=subprocess.check_output(['nvidia-smi','-i',g,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
                if uuid!=assignment['uuids'][g]:raise ValueError('Physical GPU UUID changed')
                log=(run/('worker_'+g+'.log')).open('a',buffering=1)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=g,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
                child=subprocess.Popen([PYTHONS[profile],str(ROOT/'scripts/run_adjacent_context_development.py'),'--run-id',root.name,
                    '--stage','worker','--dataset',dataset,'--profile',profile,'--gpu',g],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                children.append((g,child,log))
            # 步骤2：只按工程验收放行；每5秒持久化可观察的进度和真实PID。
            while any(p.poll() is None for _,p,_ in children):
                for g,p,_ in children:
                    if p.poll() not in (None,0):raise RuntimeError('QA worker failed on '+g)
                if all((run/'pilot_workers'/(g+'.json')).exists() for g in gpus) and not (run/'pilot_acceptance.json').exists():
                    for q in spec['question_ids'][:5]:validate_result(run,q)
                    durable(run/'pilot_acceptance.json',dict(status='passed_input_protocol',question_ids=spec['question_ids'][:5],utc=utc()))
                durable(run/'heartbeat.json',dict(pid=os.getpid(),workers={g:p.pid for g,p,_ in children},
                    completed=len(list((run/'results').glob('*.json'))),total=len(spec['question_ids']),utc=utc()))
                time.sleep(5)
            if any(p.returncode for _,p,_ in children):raise RuntimeError('QA child exited abnormally')
            if (root/'PAUSE.json').exists():raise RuntimeError('Another cell paused the experiment')
            return audit_unit(run)
        except BaseException as error:
            durable(root/'PAUSE.json',dict(dataset=dataset,profile=profile,error=repr(error),traceback=traceback.format_exc(),utc=utc()));raise
        finally:
            for _,p,log in children:
                if p.poll() is None:p.wait()
                log.close()


def prepare_all(root):
    """接口：四单元CPU准备可以并行，但625次真实生成必须等全部输入发布。"""
    root=Path(root);preflight(root);children=[]
    for dataset,profile in CELLS:
        run=unit_path(root,dataset,profile);run.mkdir(parents=True,exist_ok=True);log=(run/'prepare.log').open('a',buffering=1)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        p=subprocess.Popen([PYTHONS[profile],str(ROOT/'scripts/run_adjacent_context_development.py'),'--run-id',root.name,
            '--stage','prepare-unit','--dataset',dataset,'--profile',profile],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        children.append((dataset,profile,p,log))
    while any(p.poll() is None for _,_,p,_ in children):
        durable(root/'prepare_heartbeat.json',dict(pid=os.getpid(),sid=os.getsid(0),workers={d+'/'+m:p.pid for d,m,p,_ in children},utc=utc()));time.sleep(5)
    for d,m,p,log in children:
        log.close()
        if p.returncode:raise RuntimeError('CPU input freeze failed: '+d+'/'+m)
    protocols={d+'/'+p:sha256(unit_path(root,d,p)/'protocol.json') for d,p in CELLS}
    specs=[read_json(unit_path(root,d,p)/'protocol.json') for d,p in CELLS]
    if sum(len(s['new_ids']) for s in specs)!=625 or sum(s['partition']['parent_imports']+s['partition']['diagnostic_imports'] for s in specs)!=195:
        raise ValueError('Global call/import budget differs')
    durable(root/'input_plan.json',dict(protocols=protocols,new_calls=625,imports=195,method_results=820,max_synthetic_calls=8,utc=utc()))


def idle_worker_gpus():
    """接口：没有空闲卡是正常等待状态，驱动或查询异常仍然抛出，不能静默忽略。"""
    from videoqa_runtime.baseline_run import available_gpus
    # 步骤1：只转换资源不足这一已知状态，保留其他异常供调用方停止核查。
    try:return available_gpus(list(map(str,range(8))))[0]
    except RuntimeError as error:
        if str(error)=='No requested GPU is currently idle':return []
        raise


def execute_all(root):
    """接口：后台自动分配空闲卡，最多四单元各2卡；完成恢复只审计不重新加载。"""
    root=Path(root);preflight(root)
    if (root/'PAUSE.json').exists():raise RuntimeError('Paused experiment requires raw-state review before any resume')
    frozen=read_json(root/'input_plan.json')
    if frozen['protocols']!={d+'/'+p:sha256(unit_path(root,d,p)/'protocol.json') for d,p in CELLS}:raise ValueError('Cell protocol changed')
    with exclusive(root/'coordinator.lock'):
        active={};pending=list(CELLS)
        durable(root/'launch.json',dict(pid=os.getpid(),sid=os.getsid(0),new_call_budget=625,max_gpus_per_cell=2,utc=utc()))
        # 步骤1：已完成单元只核验；剩余单元分配时重新检查真实GPU占用。
        for cell in list(pending):
            run=unit_path(root,*cell)
            if (run/'completion_audit.json').exists():audit_unit(run);pending.remove(cell)
        while pending or active:
            for cell,(p,log,gpus) in list(active.items()):
                if p.poll() is None:continue
                log.close();del active[cell]
                if p.returncode:
                    if not (root/'PAUSE.json').exists():durable(root/'PAUSE.json',dict(cell=list(cell),returncode=p.returncode,utc=utc()))
                    for child,handle,_ in active.values():child.wait();handle.close()
                    raise RuntimeError('Cell manager failed; no automatic retry')
            if (root/'PAUSE.json').exists():raise RuntimeError('Experiment paused')
            free=idle_worker_gpus() if pending else [];used={g for _,_,gs in active.values() for g in gs};free=[g for g in free if g not in used]
            for cell in list(pending):
                if not free:break
                run=unit_path(root,*cell);previous=run/'gpu_assignment.json'
                if previous.exists():
                    assigned=list(read_json(previous)['plans'])
                    if not set(assigned)<=set(free):continue
                else:assigned=free[:2]
                for g in assigned:free.remove(g)
                log=(run/'manager.log').open('a',buffering=1)
                p=subprocess.Popen([PYTHONS[cell[1]],str(ROOT/'scripts/run_adjacent_context_development.py'),'--run-id',root.name,
                    '--stage','run-unit','--dataset',cell[0],'--profile',cell[1],'--gpus',','.join(assigned)],
                    cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                active[cell]=(p,log,assigned);pending.remove(cell)
            durable(root/'heartbeat.json',dict(pid=os.getpid(),sid=os.getsid(0),active={d+'/'+m:dict(pid=p.pid,gpus=g) for (d,m),(p,_,g) in active.items()},
                waiting=[list(c) for c in pending],completed=sum(len(list(unit_path(root,d,p).glob('results/*.json'))) for d,p in CELLS),total=820,utc=utc()))
            if pending or active:time.sleep(5)
        return report(root)


def audit_all(root):
    """接口：820结果/625新调用/195引用及最多8合成独立闭合，原始生成不改写。"""
    root=Path(root);summaries={};ids=[];files={}
    for dataset,profile in CELLS:
        run=unit_path(root,dataset,profile);value=audit_unit(run);summaries[dataset+'/'+profile]=value
        for p in (run/'calls').glob('*/*.json'):ids.append(read_json(p)['call_id'])
        for rel,h in value['artifacts'].items():files[str((run/rel).relative_to(root))]=h
    if sum(x['results'] for x in summaries.values())!=820 or sum(x['new_calls'] for x in summaries.values())!=625 or sum(x['imports'] for x in summaries.values())!=195:
        raise ValueError('Global result/call/reference counts differ')
    if len(ids)!=len(set(ids)) or sum(x['synthetic_calls'] for x in summaries.values())>8:raise ValueError('Global call IDs or synthetic budget differ')
    previous=root/'completion_audit.json'
    if previous.exists() and read_json(previous)['artifacts']!=files:raise ValueError('Completed recovery altered generation artifacts')
    out=dict(status='passed',method_results=820,new_real_calls=625,imports=195,synthetic_calls=sum(x['synthetic_calls'] for x in summaries.values()),
        unique_physical_call_ids=len(ids),artifacts=files,completed_resume_new_generations=0,utc=utc())
    durable(previous,out);return out


def report(root):
    """接口：固定四单元及191题补充、全部配对与七类结果，报告阶段不生成答案。"""
    from videoqa_audit.cross_dataset_review import cluster_report
    root=Path(root);audit=audit_all(root);reports={};panels=[];all_expected={
        ('videomme','llava8'):[114,110,126,123],('videomme','qwen32'):[130,126,136,138],
        ('mlvu','llava8'):[128,140,154,148],('mlvu','qwen32'):[131,154,146,147]}
    references=('BASE-Uniform','BASE-BLIP-TopK','RD-1.2-exp07',PARENT)
    oldpanel={(d,q) for d,q in read_json(SOURCE/'review/manifest.json')['questions']}
    lines=['# exp33固定版双数据集小测','',
        '本阶段允许带已知证据风险进行固定版本小测；旧离线门槛及原选择不改写。结果不是全量或独立盲测。','',
        '| 数据／配置 | n | 均匀 | Top-K | exp07 | exp12 | exp33 | 对exp12新增／退化 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for dataset,profile in CELLS:
        run=unit_path(root,dataset,profile);data=data_rows(dataset);spec=read_json(run/'protocol.json');banks={METHOD:{},**{m:{} for m in references}}
        results={q:read_json(run/'results'/(q+'.json')) for q in spec['question_ids']};native={};counts={};groups={}
        for row in data:
            q=row['question_id'];parent=read_json(REVIEW/'records'/dataset/profile/(q+'.json'))
            banks[METHOD][q]=dict(correct=results[q]['correct'])
            for m in references:banks[m][q]=dict(correct=parent['answers'][m]['correct'])
        if [sum(v['correct'] for v in banks[m].values()) for m in references]!=all_expected[(dataset,profile)]:raise ValueError('Historical comparison scores differ')
        stats=cluster_report(data,banks,macro=dataset=='mlvu',repetitions=10000,seed=2027)
        supplemental=[r for r in data if (dataset,r['question_id']) not in oldpanel]
        supplement=cluster_report(supplemental,banks,macro=dataset=='mlvu',repetitions=10000,seed=2027)
        if len(supplemental)!=(191 if dataset=='videomme' else 210):raise ValueError('Review-excluded denominator differs')
        pair=stats['pairs'][METHOD+' minus '+PARENT];scores=stats['scores']
        for label in ('changed','unchanged','parent_import','diagnostic_import','new_generation','same_temporal_patch','across_temporal_patches'):
            ids=[]
            for q,r in results.items():
                item=read_json(run/'inputs'/(q+'.json'));take=(label=='changed' and r['changed']) or (label=='unchanged' and not r['changed']) or r['origin']==label
                if label in ('same_temporal_patch','across_temporal_patches') and profile=='qwen32' and r['changed']:
                    slot=item['trace']['winner']['replaced_slot'];same=(slot-1)//2==slot//2;take=same==(label=='same_temporal_patch')
                if take:ids.append(q)
            groups[label]=dict(n=len(ids),parent_correct=sum(banks[PARENT][q]['correct'] for q in ids),candidate_correct=sum(banks[METHOD][q]['correct'] for q in ids),
                gains=[q for q in ids if banks[METHOD][q]['correct'] and not banks[PARENT][q]['correct']],
                losses=[q for q in ids if not banks[METHOD][q]['correct'] and banks[PARENT][q]['correct']])
        for kind in ('gain','loss'):
            for row in data:
                q=row['question_id']
                if (dataset,q) in oldpanel:continue
                good,before=banks[METHOD][q]['correct'],banks[PARENT][q]['correct']
                if (kind=='gain' and good and not before) or (kind=='loss' and before and not good):
                    panels.append(dict(dataset=dataset,profile=profile,question_id=q,kind=kind));break
        physical=[r for r in results.values() if r['origin']=='new_generation'];performance={}
        for metric in ('qa_seconds','generation_seconds','peak_allocated_gib','peak_reserved_gib'):
            values=[r['answer']['observed'].get(metric,r['answer']['observed'].get('qa_through_generation_seconds',0)) for r in physical]
            performance[metric]=dict(n=len(values),mean=float(np.mean(values)),p50=float(np.percentile(values,50)),p90=float(np.percentile(values,90)),p95=float(np.percentile(values,95)))
        for m in (METHOD,*references):
            if m==METHOD:native[m]=sum(r['native_correct'] for r in results.values())
            else:native[m]=sum(read_json(REVIEW/'records'/dataset/profile/(r['question_id']+'.json'))['answers'][m]['native_correct'] for r in data)
        item=dict(dataset=dataset,profile=profile,statistics=stats,review_excluded=supplement,groups=groups,native_correct=native,
                  performance_new_calls_only=performance,partition=spec['partition'])
        reports[dataset+'/'+profile]=item;durable(run/'comparison.json',item)
        n=len(data);numbers=[scores[m]['correct'] for m in references]+[scores[METHOD]['correct']]
        lines.append('| '+dataset+'/'+profile+' | '+str(n)+' | '+' | '.join(map(str,numbers))+' | '+str(len(pair['cells']['left_only']))+'/'+str(len(pair['cells']['right_only']))+' |')
    differences={k:v['statistics']['pairs'][METHOD+' minus '+PARENT]['difference_pp'] for k,v in reports.items()}
    gains_by_data={d:any(v['statistics']['scores'][METHOD]['correct']-v['statistics']['scores'][PARENT]['correct']>=3 for k,v in reports.items() if k.startswith(d+'/')) for d in ('videomme','mlvu')}
    outside_evidence={d:any(v['review_excluded']['pairs'][METHOD+' minus '+PARENT]['cells']['left_only'] for k,v in reports.items() if k.startswith(d+'/')) for d in ('videomme','mlvu')}
    gate=all(x>=0 for x in differences.values()) and all(gains_by_data.values()) and all(outside_evidence.values())
    summary=dict(status='completed_review_pending',method=METHOD,results=820,new_real_calls=625,imports=195,synthetic_calls=audit['synthetic_calls'],
        differences_pp=differences,full_test_recommendation_gate=gate,full_test_started=False,automatic_new_variant=False,reports=reports,utc=utc())
    durable(root/'comparison.json',summary);durable(root/'new_outcome_panel.json',dict(slots=panels,maximum=8,review_minutes_limit=45,rule='first non-existing-panel gain/loss in frozen order per cell'))
    special={}
    for q,ds,p in [('012-3','videomme','llava8'),('386-1','videomme','llava8')]:
        run=unit_path(root,ds,p);r=read_json(run/'results'/(q+'.json'));special[q]=dict(option=r['option'],correct=r['correct'],origin=r['origin'],raw_output=r['answer']['raw_output'])
    durable(root/'special_cases.json',special)
    lines+=['',f"820条结果：625新问答＋195引用；合成{audit['synthetic_calls']}次。",'',
        '012-3：'+str(special['012-3']), 'D3诊断引用：'+str(special['386-1']), '',
        '主要差值（百分点）：'+str(differences),'建议讨论全量的数值门槛：'+str(gate),
        '补充结果排除既有复核面板：Video-MME每配置191题、MLVU每配置210题。',
        '已知证据损失与新得失须结合有界复核解释，不把答错直接归为选帧失败；本阶段不自动全量或追加版本。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    durable(root/'status.json',dict(status='completed_review_pending',results=820,new_real_calls=625,imports=195,synthetic_calls=audit['synthetic_calls'],utc=utc()))
    return summary
