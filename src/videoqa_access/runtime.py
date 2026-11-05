"""受控访问试点调度、完整输入冻结、严格一次问答及链路重放。"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from urllib.request import Request,urlopen

import numpy as np
from PIL import Image
from videoqa_runtime.common import ROOT,read_json,sha256,offline_environment
from videoqa_full.state import durable,digest,exclusive,invoke_once,utc
from .trial import METHODS,freeze,prepare_manifest,build_preview,prepare_question,scorer_factory
from .transport import AccessService,start_service,PROFILES,rgb_hash

PY='/App/conda/envs/explore_video_qa/bin/python'
QPY='/App/conda/envs/explore_video_qa_lmms/bin/python'
ENTRY=ROOT/'scripts/run_video_access_pilot.py'


def data_rows(root,dataset):
    """接口：公开清单按固定顺序投影，不读取标签或旧实验对错。"""
    return [r for r in read_json(Path(root)/'manifest.json')['rows'] if r['dataset']==dataset]


def verify(root):
    """接口：恢复核对当前执行源码、清单和模型登记，不隐式接受代码或资产漂移。"""
    root=Path(root);p=read_json(root/'protocol.json')
    for rel,h in p['code'].items():
        if sha256(ROOT/rel)!=h:raise ValueError('Pilot code changed: '+rel)
    for rel,h in p['inputs'].items():
        if sha256(root/rel)!=h:raise ValueError('Pilot frozen input changed')
    for path,stat in p['models'].items():
        s=Path(path).stat()
        if (s.st_size,s.st_mtime_ns)!=(stat['bytes'],stat['mtime_ns']):raise ValueError('Shared model changed')
    for value in read_json(root/'manifest.json')['indices'].values():
        s=Path(value['video']['path']).stat()
        if dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)!=value['video_stat']:raise ValueError('Frozen original stat changed')
    return p


def protocol(root):
    """接口：任何GPU前向前冻结代码/协议及模型身份；完整结束恢复无需GPU。"""
    from videoqa_lmms_bridge.common import framework_identity
    root=Path(root)
    if (root/'protocol.json').exists():return verify(root)
    prepare_manifest(root)
    paths=[*ROOT.glob('src/videoqa_access/*.py'),*ROOT.glob('src/videoqa_methods/*.py'),
        *ROOT.glob('src/videoqa_runtime/*.py'),ENTRY,
        ROOT/'src/videoqa_audit/frame_group_diagnostic.py',ROOT/'src/videoqa_lmms_bridge/common.py',
        ROOT/'src/videoqa_lmms_bridge/llava_adapter.py',ROOT/'src/videoqa_generalization/llava_observe.py',
        ROOT/'src/videoqa_generalization/llava_budget8.py',ROOT/'src/videoqa_mlvu/backends.py',
        ROOT/'src/videoqa_mlvu/data.py',ROOT/'src/videoqa_scorer_comparison/runtime.py',
        ROOT/'src/videoqa_full/state.py',ROOT/'src/videoqa_budget32/frames.py',
        ROOT/'third_party/WFS-SB-reference/core.py',ROOT/'configs/retrieval_density_v1.json',
        ROOT/'configs/local_model_inventory.json',ROOT/'configs/llava_source_manifest.json']
    code={str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(paths))}
    inventory=read_json(ROOT/'configs/local_model_inventory.json');models={}
    # 步骤1共享权重只读核验；步骤2保存独立快照，不修改历史发布与模型目录。
    for model in inventory['models'].values():
        if model['path'] not in ('/home/models/lmms-lab/LLaVA-Video-7B-Qwen2','/home/models/Salesforce/blip-itm-base-coco'):continue
        for name,old in model['files'].items():
            p=Path(model['path'])/name;s=p.stat()
            if sha256(p)!=old['sha256']:raise ValueError('Model SHA256 differs')
            models[str(p)]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=old['sha256'])
    for rel in code:
        dest=root/'code_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    p=dict(protocol_id='budgeted-video-access-pilot-v1',code=code,framework=framework_identity(),models=models,
        inputs={n:sha256(root/n) for n in ('manifest.json','labels.json')},methods=list(METHODS),
        source_frame_budget=8,max_new_tokens=16,seed=2027,maximum_real_qa=780,maximum_synthetic_qa=8,
        preview=dict(fps=1,phase='1/4',long_edge=320,jpeg_quality=80,subsampling=2),
        profiles=PROFILES,raw_payload_budget_mib=[8,32],
        performance_boundary='Measured shared components plus actual payload network replay; not independent remote E2E',
        gate=dict(net_gain=2,comparable_cost_ratio=1.10,cost_reduction_ratio=.80),utc=utc())
    freeze(root/'protocol.json',p);return p


def heartbeat(root,stage,**kwargs):
    """接口：心跳是实时观察记录，不作为完成证明或重新启动依据。"""
    durable(Path(root)/'status.json',dict(stage=stage,pid=os.getpid(),sid=os.getsid(0),utc=utc(),**kwargs))


def select_worker(root,args):
    """接口：单GPU长期BLIP进程，按固定槽位处理题目；原片只从独立服务获得。"""
    import torch
    root=Path(root);verify(root);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    data=data_rows(root,args.dataset)[args.slot::args.workers]
    pending=[r for r in data if not (root/'prepared'/args.dataset/(r['question_id']+'.json')).exists()]
    if not pending:return dict(status='already_prepared',worker=args.slot)
    began=time.perf_counter();scorer=scorer_factory(root/'scorer_cache'/args.dataset/str(args.slot))
    counts_before=dict(scorer.counts);warm=scorer.synthetic_acceptance()
    durable(root/'selection_workers'/args.dataset/(str(args.slot)+'.json'),dict(gpu=args.gpu,
        model_load_seconds=time.perf_counter()-began,synthetic=warm,counts_before=counts_before,counts_after=dict(scorer.counts)))
    indices={r['video_id']:read_json(root/'indices'/(r['video_id']+'.json')) for r in data}
    service=AccessService(root,indices);server,thread=start_service(service);url='http://127.0.0.1:'+str(server.server_port)
    try:
        for ordinal,row in enumerate(data):
            prepare_question(root,row,scorer,url)
            durable(root/'selection_status'/args.dataset/(str(args.slot)+'.json'),dict(status='running',completed=ordinal+1,total=len(data),active_question=row['question_id'],gpu=args.gpu,pid=os.getpid(),utc=utc()))
    finally:server.shutdown();server.server_close();thread.join()
    durable(root/'selection_status'/args.dataset/(str(args.slot)+'.json'),dict(status='completed',completed=len(data),total=len(data),gpu=args.gpu,counts=dict(scorer.counts),utc=utc()))
    return dict(status='completed',questions=len(data))


def load_images(root,item):
    """接口：仅载入本轮已保存输入PNG，逐张核对像素，原片路径不参与问答解码。"""
    images=[]
    for f in item['frames']:
        with Image.open(Path(root)/f['image']) as im:value=im.convert('RGB')
        if rgb_hash(value)!=f['rgb_sha256']:raise ValueError('Prepared RGB changed')
        images.append(value)
    return images


def freeze_inputs(root,args):
    """接口：生成前冻结官方像素/完整提示Token和同题共享；不读取计分标签。"""
    from videoqa_scorer_comparison.runtime import make_tools
    from videoqa_audit.frame_group_diagnostic import cpu_input
    root=Path(root);verify(root);data=data_rows(root,args.dataset);run=root/'qa'/args.dataset;dest=run/'plan.json'
    if dest.exists():
        old=read_json(dest)
        for rel,h in old['exports'].items():
            if sha256(root/rel)!=h:raise ValueError('Frozen model input changed')
        return old
    tools=make_tools('llava8');exports={};calls=[]
    for row in data:
        q=row['question_id'];prepared=read_json(root/'prepared'/args.dataset/(q+'.json'));groups=[];items={}
        # 步骤1框架计分对象仅给占位字母；真实标签在生成返回之后才由计分侧读取。
        public=dict(row,answer_index=0);case=dict(dataset=args.dataset,profile='llava8')
        for method in METHODS:
            item=prepared['items'][method];images=load_images(root,item)
            try:processed=cpu_input(case,tools,images,public)
            finally:
                for im in images:im.close()
            identity=dict(processed,source=[[f['source_pts'],f['source_frame_index']] for f in item['frames']],
                rgb_sha256=[f['rgb_sha256'] for f in item['frames']],protocol_sha256=sha256(root/'protocol.json'))
            items[method]=dict(item,processed=processed,identity=identity)
            match=next((g for g in groups if g['identity']==identity),None)
            if match is None:
                match=dict(representative=method,methods=[],identity=identity);groups.append(match);calls.append([q,method])
            match['methods'].append(method)
        rel=f'qa/{args.dataset}/inputs/{q}.json';freeze(root/rel,dict(question_id=q,dataset=args.dataset,groups=groups,items=items,prepared_sha256=sha256(root/'prepared'/args.dataset/(q+'.json'))));exports[rel]=sha256(root/rel)
        heartbeat(root,'freezing_inputs',dataset=args.dataset,completed=len(exports),total=len(data))
    value=dict(dataset=args.dataset,questions=len(data),method_results=len(data)*6,new_calls=len(calls),shared_results=len(data)*6-len(calls),calls=calls,exports=exports)
    if len(calls)>len(data)*6:raise ValueError('QA budget exceeded')
    freeze(dest,value);return value


def qa_worker(root,args):
    """接口：已完成/原始返回先恢复，仍有未开始调用才加载一次冻结LLaVA。"""
    from videoqa_audit.frame_group_diagnostic import (task_backend,observe_answer,restore_answer,
        observed_input,canonical_option,synthetic_input)
    root=Path(root);verify(root);run=root/'qa'/args.dataset;plan=read_json(run/'plan.json');fingerprint=sha256(root/'protocol.json')
    data=data_rows(root,args.dataset)[args.slot::args.workers];case=dict(dataset=args.dataset,profile='llava8');backend=adapter=observer=None
    def get_backend():
        """步骤内懒加载：恢复和完整输入复用不产生模型加载或合成调用。"""
        nonlocal backend,adapter,observer
        if backend is None:
            import torch
            from videoqa_runtime.llava_backend import LlavaBackend
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            backend=LlavaBackend();adapter,observer=task_backend(case,backend)
            durable(run/'workers'/(str(args.slot)+'.json'),dict(gpu=args.gpu,load_report=backend.load_report,pid=os.getpid(),utc=utc()))
            images,selection,row=synthetic_input(case);rawpath=run/'synthetic/raw'/(str(args.slot)+'.json')
            try:
                invoke_once(run/'synthetic',str(args.slot),'synthetic',fingerprint,args.gpu,
                    lambda state:observe_answer(case,observer,backend,adapter,images,selection,row,state,rawpath,
                        dict(dataset=args.dataset,worker=args.slot,synthetic=True,gpu=args.gpu,protocol_sha256=fingerprint)))
            finally:
                for im in images:im.close()
    for ordinal,row in enumerate(data):
        q=row['question_id'];inp=read_json(run/'inputs'/(q+'.json'));public=dict(row,answer_index=0)
        if sha256(root/'prepared'/args.dataset/(q+'.json'))!=inp['prepared_sha256']:raise ValueError('Prepared source receipt changed')
        groups=inp['groups'];shift=ordinal%len(groups);groups=groups[shift:]+groups[:shift]
        for group in groups:
            rep=group['representative'];item=inp['items'][rep];rawpath=run/'raw_generations'/rep/(q+'.json');callpath=run/'calls'/rep/(q+'.json')
            if [q,rep] not in plan['calls']:raise ValueError('Call not in frozen generation plan')
            identity=dict(dataset=args.dataset,question_id=q,method=rep,protocol_sha256=fingerprint,gpu=args.gpu)
            if callpath.exists() and read_json(callpath)['gpu']!=args.gpu:raise ValueError('Partial question resumed on different physical GPU')
            # 步骤2完整原始返回优先恢复后处理；状态不确定时拒绝第二次生成。
            if rawpath.exists():
                if not callpath.exists():raise ValueError('Raw return has no pre-generation ledger')
                raw=read_json(rawpath)
                if raw['identity']!=identity:raise ValueError('Raw generation identity differs')
                answer=restore_answer(case,raw,public)
                if callpath.exists() and read_json(callpath)['status']!='completed':
                    old=read_json(callpath);old.update(status='completed',answer=answer,generation_state=dict(started=True,returned=True),restored_postprocessing=True);durable(callpath,old)
            else:
                if callpath.exists():raise RuntimeError('Generation ledger exists without confirmed raw return; repetition forbidden')
                get_backend();images=load_images(root,item)
                def operation(state):
                    """接口：只观测官方预处理时间，审计的首次预处理不计作生产第二次调用。"""
                    times=[];original=backend.processor.preprocess
                    def preprocess(*aa,**kk):
                        began=time.perf_counter();value=original(*aa,**kk);times.append(time.perf_counter()-began);return value
                    backend.processor.preprocess=preprocess
                    try:
                        return observe_answer(case,observer,backend,adapter,images,item['selection'],public,state,rawpath,identity)
                    finally:
                        backend.processor.preprocess=original
                        durable(run/'preprocess_times'/rep/(q+'.json'),dict(all_calls_seconds=times,
                            charged_actual_call_seconds=times[-1] if times else None,
                            interpretation='last call is official adapter preprocessing; prior validation calls excluded'))
                try:
                    answer,_,_=invoke_once(run,q,rep,fingerprint,args.gpu,
                        operation)
                finally:
                    for im in images:im.close()
            if observed_input('llava8',answer)!=item['processed']:raise ValueError('Actual model pixels/prompt differ')
            if answer['observed']['visual_tokens']!=210*len(item['frames']) or answer['observed']['max_new_tokens']!=16:raise ValueError('Generation budget differs')
            option=canonical_option(case,answer)
            labels=read_json(root/'labels.json');label='ABCD'[labels[args.dataset+'/'+q]]
            for method in group['methods']:
                record=dict(dataset=args.dataset,question_id=q,method=method,representative=rep,
                    call_id=args.dataset+':'+rep+':'+q,raw_generation_file=str(rawpath.relative_to(root)),
                    raw_generation_sha256=sha256(rawpath),answer=answer,native_prediction=answer['parsed_answer'],
                    prediction=option,correct=option==label,native_correct=answer['parsed_answer']==label,
                    input_identity=inp['items'][method]['identity'],gpu=args.gpu,protocol_sha256=fingerprint)
                freeze(run/'results'/method/(q+'.json'),record)
        durable(run/'worker_status'/(str(args.slot)+'.json'),dict(status='running',completed=ordinal+1,total=len(data),gpu=args.gpu,pid=os.getpid(),active_question=q,pilot_passed=ordinal>=4,utc=utc()))
    durable(run/'worker_status'/(str(args.slot)+'.json'),dict(status='completed',completed=len(data),total=len(data),gpu=args.gpu,utc=utc()))
    return dict(status='completed',questions=len(data))


def replay(root,args):
    """接口：真实已保存响应体只重放网络；不重复QA、不伪装再次解码原片。"""
    root=Path(root);verify(root);dest=root/'network_replay'/args.dataset/'summary.json'
    if dest.exists():return read_json(dest)
    service=AccessService(root,{});server,thread=start_service(service);url='http://127.0.0.1:'+str(server.server_port)
    records={};data=data_rows(root,args.dataset)
    try:
        for row in data:
            q=row['question_id'];prepared=read_json(root/'prepared'/args.dataset/(q+'.json'));index=read_json(root/'indices'/(row['video_id']+'.json'))
            keys=[index['preview_payload_sha256']]+[r['payload_sha256'] for item in prepared['items'].values() for r in item['requests']]
            for key in dict.fromkeys(keys):
                for profile in PROFILES:
                    path=root/'network_replay'/'payloads'/profile/(key+'.json')
                    if path.exists():value=read_json(path)
                    else:
                        body=json.dumps(dict(payload_sha256=key,link_profile=profile)).encode();began=time.perf_counter()
                        with urlopen(Request(url+'/replay',body,{'Content-Type':'application/json'}),timeout=900) as response:payload=response.read()
                        elapsed=time.perf_counter()-began
                        if hashlib.sha256(payload).hexdigest()!=key:raise ValueError('Replayed response differs')
                        value=dict(payload_sha256=key,profile=profile,seconds=elapsed,bytes=len(payload),decode_repeated=False);freeze(path,value)
                    records[profile+'/'+key]=value
            heartbeat(root,'network_replay',dataset=args.dataset,completed=data.index(row)+1,total=len(data),payload_profiles=len(records))
    finally:server.shutdown();server.server_close();thread.join()
    value=dict(status='completed',dataset=args.dataset,records=records,new_qa=0);freeze(dest,value);return value


def free_gpus(allowed):
    """接口：仅查询显存/利用率均空闲的候选卡，不抢占且不自动扩卡。"""
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return [i.strip() for line in out.splitlines() for i,m,u in [line.split(',')] if i.strip() in allowed and int(m)<1000 and int(u)<5]


def run_stage(root,stage,dataset,gpus):
    """接口：子进程继承协调会话、关闭stdin、独立日志；观察超时不重复启动。"""
    processes=[]
    for slot,gpu in enumerate(gpus):
        log=Path(root)/'logs'/(dataset+'_'+stage+'_'+str(slot)+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        stream=log.open('ab',buffering=0);python=QPY if stage=='qa' else PY
        command=[python,'-u',str(ENTRY),stage,'--root',str(root),'--dataset',dataset,'--gpu',gpu,'--slot',str(slot),'--workers',str(len(gpus))]
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT);stream.close();processes.append(process)
    durable(Path(root)/'active_children.json',dict(stage=stage,dataset=dataset,pids=[p.pid for p in processes],gpus=gpus,utc=utc()))
    while any(p.poll() is None for p in processes):
        heartbeat(root,stage,dataset=dataset,children=[dict(pid=p.pid,returncode=p.poll()) for p in processes]);time.sleep(5)
    if any(p.returncode!=0 for p in processes):raise RuntimeError('Stage failed; inspect existing workers, do not retry generation')


def coordinate(root,args):
    """接口：12视频验收→VME六组→判定→条件MLVU→独立审计，脱离交互会话。"""
    from .acceptance import engineering
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'run.lock'):
        if (root/'completion.json').exists():return audit(root,args)
        try:
            protocol(root);engineering(root)
            allowed=args.gpus.split(',')
            for dataset in ('videomme','mlvu'):
                if dataset=='mlvu' and not read_json(root/'reports/videomme.json')['gate']['passed']:
                    freeze(root/'mlvu_not_started.json',dict(reason='Video-MME prespecified resource gate not passed',new_qa=0));break
                data=data_rows(root,dataset);manifest=read_json(root/'manifest.json')
                # 步骤1准备只用CPU；步骤2每GPU阶段重新检查，固定阶段内并发。
                with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as executor:
                    for n,index in enumerate(executor.map(build_preview,[(str(root),r['video_id'],manifest['indices'][r['video_id']]) for r in data]),1):
                        heartbeat(root,'preview_build',dataset=dataset,completed=n,total=len(data))
                for stage in ('select','qa'):
                    assignment=root/'assignments'/(dataset+'_'+stage+'.json')
                    if assignment.exists():gpus=read_json(assignment)['gpus']
                    else:
                        while not (available:=free_gpus(allowed)):
                            heartbeat(root,'waiting_gpus',dataset=dataset,next_stage=stage);time.sleep(15)
                        gpus=available[:2];freeze(assignment,dict(gpus=gpus))
                    while not set(gpus)<=set(free_gpus(allowed)):
                        heartbeat(root,'waiting_original_gpus',dataset=dataset,gpus=gpus,next_stage=stage);time.sleep(15)
                    if stage=='qa':
                        cp=subprocess.run([QPY,str(ENTRY),'inputs','--root',str(root),'--dataset',dataset],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=(root/'logs'/(dataset+'_inputs.log')).open('ab'),stderr=subprocess.STDOUT)
                        if cp.returncode:raise RuntimeError('CPU actual-input freeze failed')
                    run_stage(root,stage,dataset,gpus)
                replay(root,type('Args',(),dict(dataset=dataset))());report(root,type('Args',(),dict(dataset=dataset))())
            value=audit(root,args);freeze(root/'completion.json',value);heartbeat(root,'completed',result=value);return value
        except BaseException as exc:
            durable(root/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),pid=os.getpid(),utc=utc()));heartbeat(root,'paused_error',error=repr(exc));raise


def report(root,args):
    """接口：计分与共同组件/链路成本汇总，不加载任何模型。"""
    from .report import summarize
    return summarize(Path(root),args.dataset)


def audit(root,args):
    """接口：独立重读每个结果/原文/调用与源像素；完整恢复不执行选择或模型。"""
    from .report import completion_audit
    return completion_audit(Path(root))
