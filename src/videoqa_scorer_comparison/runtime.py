"""评分器对照的两后端运行：同题输入共享、历史核验导入与严格一次生成。"""
import os
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, rows, utc, digest, invoke_once, exclusive, pilot_ids
from videoqa_budget32.frames import FrameProvider,validate_frames
from videoqa_audit.region_scope_runtime import cpu_identity,source_order,parent_result
from videoqa_lmms_bridge.common import framework_identity,load_task,task_doc
from .preparation import METHODS,PROFILES

PYTHONS = dict(qwen32='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python',
               llava8='/App/conda/envs/explore_video_qa_lmms/bin/python')
_TOOLS=None


def make_tools(profile):
    """接口：官方CPU预处理器/Tokenizer；不加载视觉语言权重或执行前向。"""
    import torch
    torch.set_num_threads(2)
    if profile=='qwen32':
        from transformers import AutoProcessor
        p=AutoProcessor.from_pretrained('/home/models/Qwen/Qwen2.5-VL-7B-Instruct',
            local_files_only=True,use_fast=False,min_pixels=100352,max_pixels=602112,
            size=dict(shortest_edge=100352,longest_edge=602112))
        p.apply_chat_template([dict(role='user',content='synthetic')],tokenize=False,add_generation_prompt=True)
        return p,None
    from transformers import AutoTokenizer
    from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
    return SigLipImageProcessor(),AutoTokenizer.from_pretrained(
        '/home/models/lmms-lab/LLaVA-Video-7B-Qwen2',local_files_only=True)


def initialize_cpu(profile):
    """接口：spawn子进程预加载懒导入，避免并行首次导入/模板编译竞争。"""
    global _TOOLS
    _TOOLS=make_tools(profile)


def observed_identity(profile,answer):
    """接口：历史实际模型输入观测与本轮CPU输入具有相同字段定义。"""
    o=answer['observed'];v=dict(prompt=o['prompt'],input_token_ids=o['input_token_ids'],
        pixel_sha256=o['processor_pixel_sha256'] if profile=='qwen32' else o['pixel_sha256'])
    if profile=='qwen32':v['video_grid_thw']=o['video_grid_thw']
    return v


def historical_sources(path,old):
    """接口：追溯物理主调用，兼容历史导入与同帧引用；不根据正确性挑来源。"""
    base=next(p for p in path.parents if (p/'protocol.json').exists())
    primary=old.get('representative',old['method']);prefix='imported/' if 'imported' in path.relative_to(base).parts else ''
    paths=[path,base/(prefix+'calls')/primary/(old['question_id']+'.json'),
           base/(prefix+'raw_generations')/primary/(old['question_id']+'.json'),base/'protocol.json']
    call,raw=read_json(paths[1]),read_json(paths[2])
    if call['status']!='completed' or call['answer']!=old['answer'] or raw['generation_state']!={'started':True,'returned':True}:
        raise ValueError('Historical physical generation incomplete')
    o=old['answer']['observed']
    if 'generation_kwargs' in o:
        if o['generation_kwargs']['max_new_tokens']!=16 or o['generation_kwargs']['do_sample'] is not False:
            raise ValueError('Historical generation settings differ')
    elif o['max_new_tokens']!=16:raise ValueError('Historical generation limit differs')
    return {str(p):sha256(p) for p in paths}


def group_inputs(items):
    """接口：同题完整身份分组，按固定方法顺序确定主调用；不读取标准答案。"""
    groups=[]
    for method in METHODS:
        item=items[method];match=next((g for g in groups if g['identity']==item['input_identity']),None)
        if match is None:
            match=dict(representative=method,methods=[],identity=item['input_identity'],
                       historical_result=None,historical_sources={});groups.append(match)
        match['methods'].append(method)
        if item.get('historical_result') and match['historical_result'] is None:
            match['historical_result']=item['historical_result'];match['historical_sources']=item['historical_sources']
    return groups


def prepare_question(job):
    """接口：逐题三方法真实RGB/像素/完整提示核验后冻结导入及共享，任何不符暂停。"""
    root,profile,row=Path(job[0]),job[1],job[2];run=root/'qa'/profile;q=row['question_id'];items={};bank={};sources={}
    for method in METHODS:
        source=root/'selections'/profile/method/(q+'.json');sources[str(source)]=sha256(source)
        selection=read_json(source)['selection'];validate_frames(selection,PROFILES[profile])
        key=digest(source_order(selection))
        if key not in bank:
            images,rgb=FrameProvider(selection).decode()
            try:identity=cpu_identity(profile,_TOOLS,images,row)
            finally:
                for image in images:image.close()
            bank[key]=(rgb,identity)
        rgb,identity=bank[key]
        item=dict(question_id=q,method=method,selection=selection,rgb_sha256=rgb,
                  input_identity=dict(identity,source=source_order(selection)))
        oldpath=parent_result(profile,q,method);old=read_json(oldpath);sources[str(oldpath)]=sha256(oldpath)
        if source_order(selection)==source_order(old['selection']):
            if rgb!=old['rgb_sha256'] or identity!=observed_identity(profile,old['answer']):
                raise ValueError('Historical same-source input is not equivalent: '+profile+'/'+method+'/'+q)
            hs=historical_sources(oldpath,old)
            origin=read_json(next(p for p in oldpath.parents if (p/'protocol.json').exists())/'protocol.json')
            current=read_json(run/'parent_protocol.json')
            for field in ('model','model_files','framework','environment'):
                if origin[field]!=current[field]:raise ValueError('Historical QA protocol identity differs: '+field)
            item.update(historical_result=str(oldpath),historical_sources=hs)
        items[method]=item
    groups=group_inputs(items);paths=[]
    for method,item in items.items():
        dest=run/'selections'/method/(q+'.json');durable(dest,item);paths.append(dest)
    gp=run/'groups'/(q+'.json');durable(gp,groups);paths.append(gp)
    return dict(question_id=q,groups=groups,sources=sources,
                exports={str(p.relative_to(run)):sha256(p) for p in paths})


def prepare_inputs(root,profile,gpus):
    """接口：全2700题输入分区在正式生成前冻结；两配置独立环境与调用账本。"""
    from videoqa_audit.framework_contraction import RUN32,RUN8
    from videoqa_full.prepare import environment
    root=Path(root);run=root/'qa'/profile;run.mkdir(parents=True,exist_ok=True)
    if (run/'protocol.json').exists():return verify(run)
    parent=RUN32 if profile=='qwen32' else RUN8;old=read_json(parent/'protocol.json')
    if framework_identity()!=old['framework'] or environment(PYTHONS[profile])!=old['environment']:
        raise ValueError('QA framework or environment changed')
    durable(run/'parent_protocol.json',old)
    # 步骤1：本轮CPU输入预处理与历史模型字节/生成协议一致，才允许历史回答引用。
    for path,value in old['model_files'].items():
        if sha256(path)!=value['sha256']:raise ValueError('Frozen QA model changed')
    data=rows();exports={};sources={};groups={};start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize_cpu,initargs=(profile,)) as executor:
        for i,item in enumerate(executor.map(prepare_question,[(str(root),profile,r) for r in data]),1):
            exports.update(item['exports']);sources.update(item['sources']);groups[item['question_id']]=item['groups']
            durable(run/'prepare_status.json',dict(completed=i,total=2700,elapsed_seconds=time.perf_counter()-start,utc=utc()))
    new=[(q,g['representative']) for q,gs in groups.items() for g in gs if not g['historical_result']]
    imports=sum(len(g['methods']) for gs in groups.values() for g in gs if g['historical_result'])
    maximum=len(new)
    if maximum>8100:raise ValueError('QA budget exceeded')
    code=read_json(root/'preflight.json')['code_sha256'];sources[str(root/'preflight.json')]=sha256(root/'preflight.json')
    spec=dict(profile=profile,model=old['model'],frame_budget=PROFILES[profile],methods=list(METHODS),
        question_ids=[r['question_id'] for r in data],question_ordinals={r['question_id']:i for i,r in enumerate(data)},
        maximum_new_real_generations=maximum,new_calls=new,allowed_gpus=gpus,model_files=old['model_files'],
        environment=old['environment'],framework=old['framework'],video_stats=old['video_stats'],
        video_hashes=old['video_hashes'],selection_exports=exports,sources=sources,code_sha256=code,
        pilot_question_ids=pilot_ids(data),scorer_id='S2-BLIP-2-ITM',max_new_tokens=16,seed=2027,utc=utc())
    durable(run/'input_summary.json',dict(method_results=8100,new_calls=maximum,historical_result_references=imports,
        within_run_shared_results=8100-imports-maximum,preparation_wall_seconds=time.perf_counter()-start))
    durable(run/'protocol.json',spec);verify(run);import_answers(run,spec);return spec


def verify(run):
    """接口：代码/环境/输入/模型指纹改变立即暂停，恢复不改变事前调用计划。"""
    from videoqa_full.prepare import environment
    run=Path(run);s=read_json(run/'protocol.json')
    if len(s['question_ids'])!=2700 or len(set(s['question_ids']))!=2700 or len(s['new_calls'])!=s['maximum_new_real_generations']:
        raise ValueError('Frozen population/call budget mismatch')
    if environment(PYTHONS[s['profile']])!=s['environment'] or framework_identity()!=s['framework']:
        raise ValueError('QA runtime fingerprint changed')
    for name,h in s['code_sha256'].items():
        if sha256(ROOT/name)!=h:raise ValueError('Code changed: '+name)
    for name,h in s['sources'].items():
        if sha256(name)!=h:raise ValueError('Input origin changed: '+name)
    for name,h in s['selection_exports'].items():
        if sha256(run/name)!=h:raise ValueError('Frozen QA input changed')
    for name,value in s['model_files'].items():
        stat=Path(name).stat()
        if stat.st_size!=value['bytes'] or stat.st_mtime_ns!=value['mtime_ns']:raise ValueError('Model file changed')
    return s


def persist_group(run,row,spec,group,answer,gpu,timings,imported=False):
    """接口：一次物理回答映射到同题等价方法，引用保留来源、成本不重复计算。"""
    run=Path(run);q=row['question_id'];fp=sha256(run/'protocol.json')
    for method in group['methods']:
        p=run/'results'/method/(q+'.json')
        if p.exists():continue
        item=read_json(run/'selections'/method/(q+'.json'))
        durable(p,dict(status='completed',model=spec['model'],method=method,representative=group['representative'],
            protocol_sha256=fp,**{k:row[k] for k in ('question_id','video_id','question','options','stratum')},
            selection=item['selection'],rgb_sha256=item['rgb_sha256'],answer=answer,gpu=gpu,
            imported=imported,historical_sources=group['historical_sources'],
            correct=answer['parsed_answer']=='ABCD'[row['answer_index']],
            timings=timings if method==group['representative'] and not imported else {},
            timing_kind='historical_verified_reference' if imported else 'physical_primary' if method==group['representative'] else 'same_input_reference',utc=utc()))


def import_answers(run,spec):
    """接口：引用仅来自事前完整输入匹配，不建立虚假的本轮生成账本。"""
    for row in rows():
        for g in read_json(Path(run)/'groups'/(row['question_id']+'.json')):
            if g['historical_result']:
                for p,h in g['historical_sources'].items():
                    if sha256(p)!=h:raise ValueError('Historical source changed')
                old=read_json(g['historical_result']);persist_group(run,row,spec,g,old['answer'],old.get('gpu'),{},True)


def recover_group(run,row,spec,group):
    """接口：仅完整原始返回恢复后处理，开始状态不明时绝不再次生成。"""
    run=Path(run);q=row['question_id'];m=group['representative'];p=run/'calls'/m/(q+'.json')
    if not p.exists():return
    call=read_json(p);fp=sha256(run/'protocol.json')
    if call['protocol_sha256']!=fp:raise ValueError('Call fingerprint differs')
    if call['status']!='completed':
        raw=read_json(run/'raw_generations'/m/(q+'.json'))
        if raw['identity']!=dict(question_id=q,method=m,protocol_sha256=fp,gpu=call['gpu']) or raw['generation_state']!={'started':True,'returned':True}:
            raise ValueError('Uncertain generation cannot be repeated')
        if spec['profile']=='qwen32':
            from videoqa_budget32.observe import result_from_raw
            answer=result_from_raw(raw,row)
        else:
            from videoqa_generalization.llava_observe import result_from_raw
            a=result_from_raw(raw,row);answer=dict(a['answer'],observed=a['observed'])
        call.update(status='completed',answer=answer,recovered_postprocessing=True);durable(p,call)
    persist_group(run,row,spec,group,call['answer'],call['gpu'],dict(recovered_postprocessing=True))


def validate_result(run,row,method,spec):
    """接口：实际RGB/官方像素/完整提示/视觉Token/账本和参考计分共同闭合。"""
    run=Path(run);q=row['question_id'];r=read_json(run/'results'/method/(q+'.json'))
    item=read_json(run/'selections'/method/(q+'.json'));a=r['answer'];o=a['observed'];expected=item['input_identity']
    if r['model']!=spec['model'] or r['method']!=method or r['protocol_sha256']!=sha256(run/'protocol.json') or any(r[k]!=row[k] for k in ('question_id','video_id','question','options','stratum')):
        raise ValueError('Result/question identity differs')
    if r['selection']!=item['selection'] or r['rgb_sha256']!=item['rgb_sha256'] or dict(observed_identity(spec['profile'],a),source=source_order(r['selection']))!=expected:
        raise ValueError('Actual input differs from frozen CPU identity')
    if o['generate_calls']!=1 or o['prefill_calls']!=1 or not 1<=len(o['generated_token_ids'])<=16:raise ValueError('Generation observations invalid')
    if spec['profile']=='llava8' and (o['visual_tokens']!=210*len(item['selection']['selected_frames']) or o['max_new_tokens']!=16):raise ValueError('LLaVA protocol differs')
    if spec['profile']=='qwen32' and (o['generation_kwargs']['max_new_tokens']!=16 or o['generation_kwargs']['do_sample'] is not False):raise ValueError('Qwen generation differs')
    metric=load_task().videomme_process_results(task_doc(row),[a['raw_output']])['videomme_perception_score']
    if metric!=a['framework_metric'] or r['correct']!=(metric['answer']==metric['pred_answer']):raise ValueError('Framework score differs')
    group=next(g for g in read_json(run/'groups'/(q+'.json')) if method in g['methods'])
    if r['representative']!=group['representative'] or r['imported']!=bool(group['historical_result']):raise ValueError('Sharing plan differs')
    if not r['imported']:
        m=group['representative'];call=read_json(run/'calls'/m/(q+'.json'));raw=read_json(run/'raw_generations'/m/(q+'.json'))
        if call['status']!='completed' or call['answer']!=a or raw['generation_state']!={'started':True,'returned':True} or raw['identity']!=dict(question_id=q,method=m,protocol_sha256=r['protocol_sha256'],gpu=r['gpu']):
            raise ValueError('Physical raw generation attribution differs')
    return r


def process_question(run,row,backend,adapter,gpu,spec):
    """接口：同题各唯一输入按轮换顺序一次生成；开始失败占额度且不重试。"""
    run=Path(run);q=row['question_id'];groups=read_json(run/'groups'/(q+'.json'));shift=spec['question_ordinals'][q]%len(groups)
    for g in groups[shift:]+groups[:shift]:
        if g['historical_result']:continue
        m=g['representative']
        if all((run/'results'/method/(q+'.json')).exists() for method in g['methods']):continue
        with exclusive(run/'locks'/(q+'.lock')):
            recover_group(run,row,spec,g)
            if all((run/'results'/method/(q+'.json')).exists() for method in g['methods']):continue
            item=read_json(run/'selections'/m/(q+'.json'));start=time.perf_counter();images,rgb=FrameProvider(item['selection']).decode()
            try:
                if rgb!=item['rgb_sha256']:raise ValueError('Runtime RGB changed')
                tools=(backend.processor,backend.tokenizer if spec['profile']=='llava8' else None)
                actual=dict(cpu_identity(spec['profile'],tools,images,row),source=source_order(item['selection']))
                if actual!=item['input_identity']:raise ValueError('Runtime preprocessing changed before generation')
                decoded=time.perf_counter();fp=sha256(run/'protocol.json');identity=dict(question_id=q,method=m,protocol_sha256=fp,gpu=gpu)
                def invoke(state):
                    """步骤2：原生框架生成返回即保存raw，后处理失败仅恢复计分。"""
                    raw=run/'raw_generations'/m/(q+'.json')
                    if spec['profile']=='qwen32':
                        from videoqa_budget32.observe import observe
                        return observe(backend,adapter,images,row,state,raw,identity)
                    from videoqa_generalization.llava_observe import observe
                    value=observe(backend,adapter,images,item['selection'],row,state,raw,identity)
                    return dict(value['answer'],observed=value['observed'])
                answer,_,checkpoint=invoke_once(run,q,m,fp,gpu,invoke)
                persist_group(run,row,spec,g,answer,gpu,dict(final_decode_and_identity_seconds=decoded-start,
                    cached_input_to_answer_seconds=time.perf_counter()-start,call_checkpoint_seconds=checkpoint))
            finally:
                for image in images:image.close()
    for method in METHODS:validate_result(run,row,method,spec)


def qa_initialize(run,gpu):
    """接口：每卡一个固定后端，先合成覆盖边界；没有真实题输入或标准答案。"""
    import torch
    from PIL import Image
    from videoqa_lmms_bridge.bridge import synthetic_row
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(2027)
    run=Path(run);spec=read_json(run/'protocol.json');start=time.perf_counter()
    if spec['profile']=='qwen32':
        from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
        from videoqa_budget32.qwen_adapter import FrozenInputQwen
        from videoqa_budget32.observe import observe
        backend=QwenFA2VideoBackend();adapter=FrozenInputQwen(backend);counts=(1,7,8,11,31,32)
    else:
        from videoqa_runtime.llava_backend import LlavaBackend
        from videoqa_generalization.llava_budget8 import adapter_for
        from videoqa_generalization.llava_observe import observe
        backend=LlavaBackend();adapter=adapter_for(backend);counts=(1,7,8)
    loading=time.perf_counter()-start;records=[];start=time.perf_counter()
    for j,n in enumerate(counts):
        size=(896,672) if j%2==0 else (672,896)
        images=[Image.new('RGB',size,(20+i*3,70,120)) for i in range(n)]
        s=dict(video=dict(path='synthetic.mp4',duration_seconds=float(n)),selected_frames=[dict(timestamp_seconds=float(i)) for i in range(n)])
        try:
            if spec['profile']=='qwen32':v=observe(backend,adapter,images,synthetic_row(n),{})
            else:v=observe(backend,adapter,images,s,synthetic_row(n),{})
            records.append(dict(source_frames=n,result=v))
        finally:
            for im in images:im.close()
    durable(run/'workers'/('qa_'+gpu+'_'+str(os.getpid())+'.json'),dict(status='passed',gpu=gpu,pid=os.getpid(),sid=os.getsid(0),
        load_seconds=loading,warmup_seconds=time.perf_counter()-start,synthetic_generations=len(counts),
        records=records,load_report=backend.load_report,utc=utc()))
    return backend,adapter


def audit(run):
    """接口：三组2700结果、物理调用预算与共享来源闭合，完成恢复不新增生成。"""
    run=Path(run);spec=verify(run);data=rows();ids=set(spec['question_ids'])
    for method in METHODS:
        if {p.stem for p in (run/'results'/method).glob('*.json')}!=ids:raise ValueError('Incomplete method results')
        for row in data:validate_result(run,row,method,spec)
    calls={(p.stem,p.parent.name) for p in (run/'calls').glob('*/*.json')}
    expected={tuple(v) for v in spec['new_calls']}
    if calls!=expected:raise ValueError('Physical generation ledger differs from frozen U')
    raw={(p.stem,p.parent.name) for p in (run/'raw_generations').glob('*/*.json')}
    if raw!=expected:raise ValueError('Raw returns incomplete')
    files={str(p.relative_to(run)):sha256(p) for folder in ('results','calls','raw_generations') for p in (run/folder).glob('*/*.json')}
    path=run/'completion_audit.json'
    if path.exists() and read_json(path)['files']!=files:raise ValueError('Completion recovery changed model artifacts')
    durable(path,dict(status='passed',method_results=8100,new_generations=len(calls),files=files,utc=utc()))
    return dict(status='completed',new_generations=len(calls),method_results=8100)
