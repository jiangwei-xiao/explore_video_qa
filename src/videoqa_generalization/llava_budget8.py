"""LLaVA八帧五策略：冻结输入分组、一次生成共享及后台安全恢复。"""
import contextlib
import csv
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from videoqa_runtime.common import ROOT, read_json, sha256, offline_environment
from videoqa_full.state import durable, exclusive, rows, pilot_ids, utc, digest, invoke_once
from videoqa_full.prepare import environment
from videoqa_budget32.frames import FrameProvider, validate_frames, native_frames
from videoqa_budget32.selection import config, two_pass, INDEX, load_old
from videoqa_methods.retarget_seed_pool import retarget_pool
from videoqa_methods.seed_budget_selection import select_seed_budget
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_runtime.baseline_selection import select_topk
from videoqa_lmms_bridge.common import framework_identity, load_task, task_doc, POST_PROMPT
from .inputs import BASE, LLAVA_UNIFORM, LLAVA_BASE, DEV, frozen
from .exp07_budget16 import SOURCE, HISTORY, selection_from_prefix

METHODS = ('BASE-Uniform','BASE-BLIP-TopK','RD-1.2','RD-1.2-exp07','RD-1.2-exp12')
LPY = '/App/conda/envs/explore_video_qa_lmms/bin/python'
EXP07 = ROOT/'outputs/full_videoqa/exp07_budget16__videomme2700__20261005_r1'
_CPU_TOOLS = None


def initialize_cpu_tools(root):
    """接口：主线程预加载懒导入组件，避免Transformers首次并行导入竞态。"""
    global _CPU_TOOLS
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from transformers import AutoTokenizer
    from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
    from videoqa_lmms_bridge.llava_adapter import tensor_hash
    # 步骤1：仅加载Tokenizer/CPU预处理器，不加载模型或执行前向。
    pre=read_json(Path(root)/'preflight.json')
    tokenizer=AutoTokenizer.from_pretrained(pre['model_path'],local_files_only=True)
    _CPU_TOOLS=(torch,tokenizer,SigLipImageProcessor(),tensor_hash)
    prepared_prompt(rows()[0],tokenizer)


def prepare_job(job):
    """接口：可序列化的CPU任务，独立进程避免PyAV/解释器在线程池中的争用。"""
    return prepare_one(*job)


def input_groups(items):
    """接口：仅同题内按完整输入身份分组，主方法由固定顺序决定，不读取答案。"""
    groups=[]
    # 步骤1：源身份、RGB、预处理和提示Token共同匹配，分数与候选编号不参与等价判断。
    for method in METHODS:
        item=items[method];identity=item['input_identity']
        match=next((g for g in groups if g['identity']==identity),None)
        if match is None:
            match=dict(representative=method,methods=[],identity=identity,input_sha256=digest(identity))
            groups.append(match)
        match['methods'].append(method)
    return groups


def source_key(selection):
    """接口：候选编号可变化，源身份按视频、编号、PTS及实际时间比较。"""
    return [selection['video']['path'],[(f['source_frame_index'],f['source_pts'],f['timestamp_seconds'])
            for f in selection['selected_frames']]]


def prepared_prompt(row,tokenizer):
    """接口：只用原题和选项构造正式模板，标准答案不进入模型输入。"""
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    context=load_task().videomme_doc_to_text(task_doc(row),{'post_prompt':POST_PROMPT})
    conv=conv_templates['qwen_1_5'].copy();conv.append_message(conv.roles[0],'<image>\n'+context)
    conv.append_message(conv.roles[1],None);prompt=conv.get_prompt()
    return prompt,tokenizer_image_token(prompt,tokenizer,-200)


def prepare_one(root,row):
    """接口：独立重放三组M16控制，再按M8合法池生成五输入；CPU核验真实像素。"""
    root=Path(root);q=row['question_id'];started=time.perf_counter()
    paths=dict(pool32=BASE/'pools'/(q+'.json'),features=BASE/'features'/(q+'.npy'),
        source_check=BASE/'selection_checkpoints'/(q+'.json'),
        pool16=SOURCE/'inputs/pools'/(q+'.json'),trace16=SOURCE/'inputs/traces'/(q+'.json'),
        selection16=SOURCE/'inputs/m16'/(q+'.json'),
        exp07=EXP07/'inputs/selections'/(q+'.json'))
    sources={str(p):sha256(p) for p in paths.values()};check=root/'inputs/checkpoints'/(q+'.json')
    if check.exists():
        record=read_json(check)
        if any(record['sources'].get(p)!=h for p,h in sources.items()):raise ValueError('Preparation source changed')
        for p,h in record['sources'].items():
            if sha256(p)!=h:raise ValueError('Prepared source changed')
        for rel,h in record['files'].items():
            if sha256(root/rel)!=h:raise ValueError('Prepared input changed')
        return record
    p32=read_json(paths['pool32']);f32=np.load(paths['features'],allow_pickle=False)
    source_check=read_json(paths['source_check'])
    for name in ['pool32','features']:
        if sources[str(paths[name])]!=source_check['files'][str(paths[name].relative_to(BASE))]:
            raise ValueError('Frozen pool/feature mismatch')
    # 步骤1：原RD控制用其独立旧池；exp07/12控制用冻结M16全粗资格，避免混淆方法定义。
    old,of,old_sources=load_old(q);sources.update(old_sources);control,_,_=two_pass(old,of,config(16,16))
    if control!=old['selected_indices']:raise ValueError('RD M16 control mismatch: '+q)
    p16=read_json(paths['pool16']);f16=f32[p16['original_candidate_indices']]
    c12,t12=select_seed_budget(p16,f16,16);saved=read_json(paths['trace16'])
    if c12!=read_json(paths['selection16'])['selected_indices']:raise ValueError('Exp12 M16 control mismatch')
    c07=selection_from_prefix(p16,t12['prefix_trace'])
    if c07!=read_json(paths['exp07'])['selected_indices']:raise ValueError('Exp07 M16 control mismatch')
    for a,b in zip(t12['prefix_trace']['regions'],saved['prefix_trace']['regions']):
        if a['prefix']!=b['prefix'] or not np.allclose(a['coverage_curve'],b['coverage_curve'],atol=1e-6,rtol=0):
            raise ValueError('M16 prefix/coverage mismatch')
    # 步骤2：严格按目标映射筛回M8补查池，不新增解码采样或视觉模型调用。
    pool,features=retarget_pool(p32,f32,config(8,8));rd,allocation,rdtrace=two_pass(pool,features,config(8,8))
    e12,e12trace=select_seed_budget(pool,features,8);prefix=e12trace['prefix_trace']
    e07=sorted([i for r in prefix['regions'] for i in r['prefix'][:prefix['region_budgets'][str(r['region_id'])]]],
        key=lambda i:(pool['candidates'][i]['source_pts'],pool['candidates'][i]['source_frame_index']))
    independent,_=select_relevance_coverage(pool,features,prefix)
    if independent!=e12:raise ValueError('Shared prefix mismatch')
    vid=Path(pool['video']['path']).stem;record=read_json(INDEX/'uniform_records'/(vid+'.json'))
    pts_path=INDEX/record['pts_file']
    if sha256(pts_path)!=record['pts_sha256']:raise ValueError('Source PTS index changed')
    sources[str(pts_path)]=record['pts_sha256'];sources[str(INDEX/'uniform_records'/(vid+'.json'))]=sha256(INDEX/'uniform_records'/(vid+'.json'))
    selections={METHODS[0]:dict(video=record['video'],selected_frames=native_frames(np.load(pts_path,allow_pickle=False),record['video'],8)),
        METHODS[1]:dict(video=pool['video'],selected_frames=select_topk(pool['candidates'][:pool['initial_count']],pool['initial_scores'],8))}
    for m,ids in [(METHODS[2],rd),(METHODS[3],e07),(METHODS[4],e12)]:
        selections[m]=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in ids])
    # 步骤3：CPU官方预处理与Token核验；相同源帧只复用已核验内容，分组不接触答案。
    if _CPU_TOOLS is None:raise ValueError('CPU components must be initialized before parallel preparation')
    torch,tokenizer,processor,tensor_hash=_CPU_TOOLS
    pre=read_json(root/'preflight.json');prompt,tokens=prepared_prompt(row,tokenizer);items={};bank={}
    for method,selection in selections.items():
        validate_frames(selection,8);key=digest(source_key(selection))
        if key not in bank:
            images,hashes=FrameProvider(selection).decode()
            try:pixel=tensor_hash(processor.preprocess(images,return_tensors='pt')['pixel_values'].to(torch.bfloat16))
            finally:
                for image in images:image.close()
            bank[key]=(hashes,pixel)
        hashes,pixel=bank[key]
        identity=dict(source=source_key(selection),rgb_sha256=hashes,pixel_sha256=pixel,prompt=prompt,
            input_token_ids=tokens,model_files_sha256=digest(pre['model_files']),
            protocol_input_sha256=pre['input_protocol_sha256'])
        items[method]=dict(question_id=q,method=method,selection=selection,rgb_sha256=hashes,
            input_identity=identity,configuration='K8' if method in METHODS[:2] else 'M8/K8',sources=sources)
    groups=input_groups(items);files={}
    values={f'selections/{m}/{q}.json':v for m,v in items.items()}
    values.update({f'inputs/groups/{q}.json':groups,f'inputs/pools/{q}.json':pool,
        f'inputs/traces/{q}.json':dict(rd_allocation=allocation,rd_selection=rdtrace,exp07=prefix,exp12=e12trace)})
    for rel,value in values.items():frozen(root/rel,value);files[rel]=sha256(root/rel)
    result=dict(question_id=q,sources=sources,files=files,unique_inputs=len(groups),m16_controls_exact=3,
        sharing={m:next(g['representative'] for g in groups if m in g['methods']) for m in METHODS},
        seconds=time.perf_counter()-started,frame_counts={m:len(s['selected_frames']) for m,s in selections.items()})
    frozen(check,result);return result


def prepare(root,gpus):
    """接口：冻结环境/代码/输入、全量控制和真实生成上限U，恢复不重新选择。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'prepare.lock'):
        from . import run as oldruntime
        oldruntime.verify(SOURCE/'llava16')
        from videoqa_budget32.prefix_full import release_check
        release_check();source=read_json(SOURCE/'preflight.json')
        baseline=read_json(LLAVA_UNIFORM/'protocol.json')
        model=baseline['model'];model_path=read_json(ROOT/'configs/local_model_inventory.json')['models']['llava']['path']
        model_files={p:v for p,v in source['model_files'].items() if p.startswith(model_path+'/')}
        if not model_files:raise ValueError('Missing LLaVA weight manifest')
        pre=dict(model=model,model_path=model_path,model_files=model_files,video_stats=source['video_stats'],video_hashes=source['video_hashes'],
            environment=environment(LPY),framework=framework_identity(),input_protocol_sha256=digest(dict(
                model_files=model_files,framework=framework_identity(),generation=dict(do_sample=False,max_new_tokens=16),
                dtype='bfloat16',attention='sdpa',conversation='qwen_1_5',size=384,seed=2027)))
        frozen(root/'preflight.json',pre)
        if (root/'protocol.json').exists():return verify(root)
        started=time.perf_counter()
        # 步骤1：900视频和权重字节核验；已有快照只作为来源，不冒充本轮验收。
        for i,(vid,st) in enumerate(pre['video_stats'].items(),1):
            if sha256(st['path'])!=pre['video_hashes'][vid]:raise ValueError('Video changed')
            if i%50==0:durable(root/'input_status.json',dict(status='hashing',completed=i,total=900,utc=utc()))
        for p,item in model_files.items():
            if sha256(p)!=item['sha256']:raise ValueError('Model changed')
        results=[]
        # spawn隔离解码器及懒导入状态；每进程CPU/解码各2线程，准备仍为零模型调用。
        with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),
                initializer=initialize_cpu_tools,initargs=(str(root),)) as executor:
            for r in executor.map(prepare_job,[(str(root),row) for row in rows()]):
                results.append(r);durable(root/'input_status.json',dict(status='preparing',completed=len(results),total=2700,
                    elapsed_seconds=time.perf_counter()-started,utc=utc()))
        # 步骤2：准备完成后锁定每个唯一输入的主记录；代码/环境指纹在GPU前冻结。
        code=oldruntime.code_sources()
        for name in ['scripts/run_llava_budget8_full.py','src/videoqa_generalization/llava_budget8.py']:
            code[name]=sha256(ROOT/name)
        for rel in code:
            dest=root/'code_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        exports={rel:h for r in results for rel,h in r['files'].items()}
        maximum=sum(r['unique_inputs'] for r in results)
        spec=dict(model=model,model_kind='llava',methods=list(METHODS),source_frame_budget=8,
            max_calls=maximum,question_count=2700,allowed_gpus=gpus,code_sha256=code,
            preflight_sha256=sha256(root/'preflight.json'),selection_exports=exports,
            environment=pre['environment'],framework=pre['framework'],pilot_question_ids=pilot_ids(rows()),
            model_files=model_files,video_stats=pre['video_stats'],video_hashes=pre['video_hashes'],
            question_ordinals={r['question_id']:i for i,r in enumerate(rows())},max_new_tokens=16,seed=2027,utc=utc())
        frozen(root/'input_summary.json',dict(status='passed',questions=2700,method_results=13500,unique_inputs=maximum,
            shared_results=13500-maximum,m16_control_replays=8100,
            frame_counts={m:sorted({r['frame_counts'][m] for r in results}) for m in METHODS},
            primary_calls_by_method={m:sum(r['sharing'][m]==m for r in results) for m in METHODS},new_selection_model_calls=0))
        frozen(root/'protocol.json',spec)
        durable(root/'input_execution.json',dict(wall_seconds=time.perf_counter()-started,utc=utc()))
        return spec


def verify(root):
    """接口：核验运行指纹及准备产物，代码变更时暂停，不绕过历史指纹。"""
    root=Path(root);spec=read_json(root/'protocol.json')
    if framework_identity()!=spec['framework'] or environment(LPY)!=spec['environment']:raise ValueError('Runtime changed')
    if sha256(root/'preflight.json')!=spec['preflight_sha256']:raise ValueError('Preflight changed')
    for rel,h in spec['code_sha256'].items():
        if sha256(ROOT/rel)!=h:raise ValueError('Code changed: '+rel)
    for rel,h in spec['selection_exports'].items():
        if sha256(root/rel)!=h:raise ValueError('Input changed: '+rel)
    return spec


def validate(root,row,spec):
    """接口：五方法结果核验至同一原始调用，共享成本不重复记账。"""
    root=Path(root);q=row['question_id'];fp=sha256(root/'protocol.json');results=[]
    for method in METHODS:
        item=read_json(root/'selections'/method/(q+'.json'));r=read_json(root/'results'/method/(q+'.json'))
        if r['question']!=row['question'] or r['options']!=row['options'] or r['protocol_sha256']!=fp:
            raise ValueError('Question/protocol mismatch')
        if r['selection']!=item['selection'] or r['rgb_sha256']!=item['rgb_sha256']:raise ValueError('Source mismatch')
        primary=r['representative'];group=next(g for g in read_json(root/'inputs/groups'/(q+'.json')) if method in g['methods'])
        if primary!=group['representative']:raise ValueError('Sharing group changed')
        call=read_json(root/'calls'/primary/(q+'.json'));raw=read_json(root/'raw_generations'/primary/(q+'.json'))
        a=r['answer'];o=a['observed'];identity=item['input_identity']
        if call['status']!='completed' or call['answer']!=a or call['gpu']!=r['gpu']:raise ValueError('Call mismatch')
        if raw['identity']!=dict(question_id=q,method=primary,protocol_sha256=fp,gpu=r['gpu']) or raw['generation_state']!={'started':True,'returned':True}:
            raise ValueError('Raw return mismatch')
        from .llava_observe import result_from_raw
        restored=result_from_raw(raw,row)
        if dict(restored['answer'],observed=restored['observed'])!=a:raise ValueError('Raw answer changed')
        if o['pixel_sha256']!=identity['pixel_sha256'] or o['input_token_ids']!=identity['input_token_ids'] or a['prompt']!=identity['prompt']:
            raise ValueError('Actual pixels/prompt differ')
        n=len(item['selection']['selected_frames']);validate_frames(item['selection'],8)
        if o['visual_tokens']!=n*210 or o['max_new_tokens']!=16 or o['generate_calls']!=1 or o['prefill_calls']!=1:
            raise ValueError('Visual/generation protocol mismatch')
        metric=load_task().videomme_process_results(task_doc(row),[a['raw_output']])['videomme_perception_score']
        if metric!=a['framework_metric'] or r['correct']!=(metric['answer']==metric['pred_answer']):raise ValueError('Scoring mismatch')
        if r['correct']!=(a['parsed_answer']=='ABCD'[row['answer_index']]):raise ValueError('Independent score mismatch')
        results.append(r)
    if len({r['gpu'] for r in results})!=1:raise ValueError('Same question used different GPUs')
    return results


def recover_call(root,row,primary,fp):
    """接口：完整原始返回可恢复账本；未知生成状态绝不再次调用。"""
    path=Path(root)/'calls'/primary/(row['question_id']+'.json')
    if not path.exists():return
    call=read_json(path)
    if call['protocol_sha256']!=fp:raise ValueError('Call fingerprint mismatch')
    if call['status']!='completed':
        raw=read_json(Path(root)/'raw_generations'/primary/(row['question_id']+'.json'))
        if raw['identity']!=dict(question_id=row['question_id'],method=primary,protocol_sha256=fp,gpu=call['gpu']):
            raise ValueError('Recovery identity mismatch')
        from .llava_observe import result_from_raw
        result=result_from_raw(raw,row)
        call.update(status='completed',answer=dict(result['answer'],observed=result['observed']),
            generation_state=raw['generation_state'],recovered_postprocessing=True)
        durable(path,call)


def process(root,row,backend,adapter,gpu,spec):
    """接口：同题循环方法顺序，完整身份一致的帧组只生成一次，逐方法保存来源。"""
    root=Path(root);q=row['question_id'];fp=sha256(root/'protocol.json')
    groups=read_json(root/'inputs/groups'/(q+'.json'));ordinal=spec['question_ordinals'][q]
    order=METHODS[ordinal%5:]+METHODS[:ordinal%5]
    groups=sorted(groups,key=lambda g:min(order.index(m) for m in g['methods']))
    # 步骤1：任何共享源调用先核查返回状态，不能从缺方法结果推断需要重新生成。
    with exclusive(root/'locks'/(q+'.lock')):
        for group in groups:
            primary=group['representative'];recover_call(root,row,primary,fp);started=time.perf_counter()
            item=read_json(root/'selections'/primary/(q+'.json'));images=[];decode_seconds=0
            call_path=root/'calls'/primary/(q+'.json');new=not call_path.exists()
            if new:
                images,hashes=FrameProvider(item['selection']).decode();decode_seconds=time.perf_counter()-started
                if hashes!=item['rgb_sha256']:raise ValueError('Runtime RGB changed')
                from videoqa_lmms_bridge.llava_adapter import tensor_hash
                pixel=tensor_hash(backend.processor.preprocess(images,return_tensors='pt')['pixel_values'].to(backend.torch.bfloat16))
                if pixel!=group['identity']['pixel_sha256']:raise ValueError('Runtime processor changed')
            from .llava_observe import observe
            def invoke(state):
                result=observe(backend,adapter,images,item['selection'],row,state,root/'raw_generations'/primary/(q+'.json'),
                    dict(question_id=q,method=primary,protocol_sha256=fp,gpu=gpu))
                return dict(result['answer'],observed=result['observed'])
            try:answer,reused,checkpoint=invoke_once(root,q,primary,fp,gpu,invoke)
            finally:
                for image in images:image.close()
            call=read_json(call_path)
            if call['gpu']!=gpu:raise ValueError('Partial question GPU differs')
            # 步骤2：每个方法保持自身选择元数据；共享回答只引用主调用，物理成本记一次。
            for method in group['methods']:
                dest=root/'results'/method/(q+'.json')
                if dest.exists():continue
                own=read_json(root/'selections'/method/(q+'.json'));metric=answer['framework_metric']
                timings={} if reused or method!=primary else dict(final_decode_seconds=decode_seconds,
                    cached_input_to_persisted_answer_seconds=time.perf_counter()-started,call_start_checkpoint_seconds=checkpoint)
                durable(dest,dict(status='completed',model=spec['model'],method=method,representative=primary,
                    protocol_sha256=fp,**{k:row[k] for k in ['question_id','video_id','question','options','stratum']},
                    selection=own['selection'],rgb_sha256=own['rgb_sha256'],answer=answer,gpu=gpu,
                    correct=metric['answer']==metric['pred_answer'],call_id=call['call_id'],
                    shared=method!=primary,timing_kind='primary' if timings else 'shared_or_resumed',timings=timings,utc=utc()))
        validate(root,row,spec)


def adapter_for(backend):
    """接口：独立八帧输入适配，仅接管冻结帧读取，继承LMMs-Eval生成实现。"""
    from videoqa_lmms_bridge.llava_adapter import FrozenInputLlavaVid
    class Budget8Adapter(FrozenInputLlavaVid):
        def __init__(self,backend):
            super().__init__(backend);self.max_frames_num=8
        def load_video(self,path,max_frames_num,fps,force_sample=False):
            if max_frames_num!=8 or self.frozen is None or path!=self.frozen['path']:raise ValueError('Unexpected media request')
            self.load_calls+=1
            return np.stack([np.asarray(im.convert('RGB')) for im in self.frozen['frames']]),'',self.frozen['duration']
    return Budget8Adapter(backend)


def synthetic(root,gpus):
    """接口：正式八帧适配覆盖1/7/8帧和横竖屏，合成调用与正式账本分开。"""
    from videoqa_runtime.baseline_run import available_gpus
    root=Path(root)
    if (root/'synthetic/acceptance.json').exists():return
    while True:
        try:free,_=available_gpus(gpus)
        except RuntimeError:free=[]
        if free:break
        durable(root/'synthetic_status.json',dict(status='waiting_resources',utc=utc()));time.sleep(15)
    os.environ['CUDA_VISIBLE_DEVICES']=free[0];offline_environment()
    import torch
    from PIL import Image
    from videoqa_runtime.llava_backend import LlavaBackend
    from videoqa_lmms_bridge.bridge import synthetic_row
    from .llava_observe import observe
    torch.set_num_threads(2);torch.set_num_interop_threads(1);backend=LlavaBackend();adapter=adapter_for(backend)
    # 步骤1：每个合成开始前记账，返回检查点用于验收恢复，不允许不确定时重发。
    for n,size in [(1,(896,672)),(7,(672,896)),(8,(896,672))]:
        path=root/'synthetic'/f'{n}.json';raw_path=root/'synthetic'/f'{n}_raw.json'
        if path.exists():
            if read_json(path)['status']!='completed':raise ValueError('Uncertain synthetic generation')
            continue
        images=[Image.new('RGB',size,(40,20+i*7,150)) for i in range(n)]
        selection=dict(video=dict(path='synthetic.mp4',duration_seconds=float(n)),selected_frames=[dict(timestamp_seconds=float(i)) for i in range(n)])
        durable(path,dict(status='started',utc=utc()))
        result=observe(backend,adapter,images,selection,synthetic_row(n),{},raw_path,dict(synthetic_frames=n))
        for im in images:im.close()
        if result['observed']['visual_tokens']!=210*n:raise ValueError('Synthetic visual budget differs')
        durable(path,dict(status='completed',result=result,utc=utc()))
    durable(root/'synthetic/acceptance.json',dict(status='passed',synthetic_generations=3,load_report=backend.load_report,utc=utc()))


def worker(root,gpu,jobs,events,stop):
    """接口：单卡长期模型进程，合成预热记账，任务异常停止继续领取。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    offline_environment();root=Path(root);active=None
    with (root/'logs'/f'gpu_{gpu}.log').open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import torch
            from PIL import Image
            from videoqa_runtime.llava_backend import LlavaBackend
            from videoqa_lmms_bridge.bridge import synthetic_row
            from .llava_observe import observe
            torch.set_num_threads(2);torch.set_num_interop_threads(1);spec=read_json(root/'protocol.json')
            start=time.perf_counter();backend=LlavaBackend();adapter=adapter_for(backend);loading=time.perf_counter()-start
            start=time.perf_counter();images=[Image.new('RGB',(896,672),(30,i*8,150)) for i in range(8)]
            selection=dict(video=dict(path='synthetic.mp4',duration_seconds=8.),selected_frames=[dict(timestamp_seconds=float(i)) for i in range(8)])
            warm=observe(backend,adapter,images,selection,synthetic_row(8),{})
            for im in images:im.close()
            durable(root/'workers'/f'{gpu}_{os.getpid()}.json',dict(gpu=gpu,pid=os.getpid(),synthetic_generations=1,
                load_seconds=loading,warmup_seconds=time.perf_counter()-start,warmup=warm,load_report=backend.load_report))
            events.put(dict(type='ready',gpu=gpu))
            # 步骤1：一题的五组固定同卡，调用异常不能继续下一题。
            while not stop.is_set():
                row=jobs.get()
                if row is None:break
                active=row['question_id'];events.put(dict(type='active',gpu=gpu,question_id=active))
                process(root,row,backend,adapter,gpu,spec)
                events.put(dict(type='completed',gpu=gpu,question_id=active));active=None
        except BaseException as exc:
            stop.set();traceback.print_exc();durable(root/'failures'/f'{gpu}_{time.time_ns()}.json',
                dict(error=repr(exc),question_id=active,traceback=traceback.format_exc(),utc=utc()))
            events.put(dict(type='failed',gpu=gpu,error=repr(exc)))


def execute(root):
    """接口：后台协调全部唯一输入，先行验收后动态派发，完成恢复不加载模型。"""
    from videoqa_runtime.baseline_run import available_gpus
    root=Path(root)
    with exclusive(root/'coordinator.lock'):
        spec=verify(root);data=rows();byid={r['question_id']:r for r in data};done=set();pinned={};fp=sha256(root/'protocol.json')
        calls=list((root/'calls').glob('*/*.json'))
        if len(calls)>spec['max_calls']:raise ValueError('Call budget exceeded')
        # 步骤1：恢复先核查全部已有调用的归属/状态，完整返回只恢复后处理。
        for path in calls:
            q=path.stem;primary=path.parent.name
            if q not in byid or primary not in [g['representative'] for g in read_json(root/'inputs/groups'/(q+'.json'))]:
                raise ValueError('Unknown call group')
            recover_call(root,byid[q],primary,fp);gpu=read_json(path)['gpu']
            if gpu not in spec['allowed_gpus'] or (q in pinned and pinned[q]!=gpu):raise ValueError('Pinned GPU mismatch')
            pinned[q]=gpu
        for row in data:
            if all((root/'results'/m/(row['question_id']+'.json')).exists() for m in METHODS):
                validate(root,row,spec);done.add(row['question_id'])
        if len(done)==2700:return
        if read_json(root/'synthetic/acceptance.json')['status']!='passed':raise ValueError('Synthetic gate incomplete')
        start=time.perf_counter();active={};gpus=[];phase='waiting_resources'
        def publish():
            elapsed=time.perf_counter()-start;count=len(list((root/'calls').glob('*/*.json')))
            durable(root/'status.json',dict(status=phase,completed_questions=len(done),total=2700,
                started_unique_calls=count,max_calls=spec['max_calls'],active=active,gpus=gpus,pid=os.getpid(),sid=os.getsid(0),
                elapsed_seconds=elapsed,eta_seconds=(2700-len(done))*elapsed/len(done) if done else None,utc=utc()))
        required={g for q,g in pinned.items() if q not in done}
        while True:
            try:gpus,states=available_gpus(spec['allowed_gpus'])
            except RuntimeError:gpus=[];states={}
            if gpus and required<=set(gpus):break
            publish();time.sleep(15)
        for folder in ['logs','workers','failures']:(root/folder).mkdir(exist_ok=True)
        durable(root/'gpu_preflight.json',dict(actual=gpus,states=states,utc=utc()))
        ctx=mp.get_context('spawn');jobs={g:ctx.Queue() for g in gpus};events=ctx.Queue();stop=ctx.Event()
        children={};ready=set();busy=set();pending=[]
        def schedule():
            nonlocal phase,pending
            pilot=spec['pilot_question_ids']
            if any(q not in done for q in pilot):phase='pilot';pending=[q for q in pilot if q not in done]
            else:
                for q in pilot:validate(root,byid[q],spec)
                durable(root/'pilot_acceptance.json',dict(status='passed',question_ids=pilot,accuracy_not_gate=True))
                phase='running';pending=sorted([q for q in byid if q not in done],key=lambda q:(-spec['video_stats'][byid[q]['video_id']]['duration_seconds'],q))
        def refill():
            for g in gpus:
                if g in busy:continue
                q=next((q for q in pending if q not in pinned or pinned[q]==g),None)
                if q is not None:pending.remove(q);busy.add(g);jobs[g].put(byid[q])
        # 步骤2：先行放行依据实际输入/生成，不读取中途分数。
        try:
            phase='loading'
            for g in gpus:
                child=ctx.Process(target=worker,args=(str(root),g,jobs[g],events,stop));child.start();children[g]=child
            publish()
            while len(done)<2700:
                try:event=events.get(timeout=5)
                except queue.Empty:event=None
                if event:
                    g=event['gpu']
                    if event['type']=='failed':raise RuntimeError(str(event))
                    if event['type']=='ready':
                        ready.add(g)
                        if len(ready)==len(gpus):schedule();refill()
                    elif event['type']=='active':active[g]=event['question_id']
                    elif event['type']=='completed':
                        done.add(event['question_id']);busy.remove(g);active.pop(g,None)
                        if phase=='pilot' and not pending and not busy:schedule()
                        refill()
                for g,child in children.items():
                    if child.exitcode is not None:raise RuntimeError(f'Worker {g} exited {child.exitcode}')
                publish()
        except BaseException:phase='paused';stop.set();publish();raise
        finally:
            for job in jobs.values():job.put(None)
            while any(c.is_alive() for c in children.values()):
                for child in children.values():child.join(timeout=1)
                publish()
        phase='completed';publish()
        durable(root/'execution.json',dict(status='completed',wall_seconds=time.perf_counter()-start,actual_gpus=gpus,utc=utc()))


def audit(root):
    """接口：独立闭合13500结果/唯一调用，完成恢复实际核验零生成。"""
    root=Path(root);spec=verify(root);data=rows();ids={r['question_id'] for r in data};calls=[]
    # 步骤1：每方法人口闭合，每题共享来源核验，所有调用ID必须唯一。
    for method in METHODS:
        if {p.stem for p in (root/'results'/method).glob('*.json')}!=ids:raise ValueError('Incomplete method')
    expected=set()
    for row in data:
        validate(root,row,spec)
        for group in read_json(root/'inputs/groups'/(row['question_id']+'.json')):
            primary=group['representative'];expected.add((primary,row['question_id']))
            calls.append(read_json(root/'calls'/primary/(row['question_id']+'.json'))['call_id'])
    for folder in ['calls','raw_generations']:
        if {(p.parent.name,p.stem) for p in (root/folder).glob('*/*.json')}!=expected:raise ValueError('Call population differs')
    if len(calls)!=len(set(calls)) or len(calls)!=spec['max_calls']:raise ValueError('Unique call budget differs')
    before={str(p):sha256(p) for f in ['results','calls','raw_generations'] for p in (root/f).glob('*/*.json')}
    # 步骤2：真实执行已完成恢复路径，无模型加载，无重复生成。
    execute(root)
    after={str(p):sha256(p) for f in ['results','calls','raw_generations'] for p in (root/f).glob('*/*.json')}
    if before!=after:raise ValueError('Completed recovery changed files')
    durable(root/'completion_audit.json',dict(status='passed',method_results=13500,new_calls=len(calls),
        shared_results=13500-len(calls),unchanged_files=len(before),extra_generations=0,utc=utc()))


def summarize(root):
    """接口：五组全量、十组配对及8→16比较；物理成本仅统计本轮实际调用。"""
    from videoqa_full.report import compare
    from videoqa_runtime.baseline_report import wilson,distribution
    from .report import reference
    root=Path(root);spec=verify(root);data=rows();groups={m:{} for m in METHODS}
    for row in data:
        for r in validate(root,row,spec):groups[r['method']][row['question_id']]=r
    dev=read_json(DEV)['rows'];dq={r['question_id'] for r in dev};dv={r['video_id'] for r in dev}
    subsets=dict(all2700=data,development200=[r for r in data if r['question_id'] in dq],
        other2500=[r for r in data if r['question_id'] not in dq],outside_development_videos2100=[r for r in data if r['video_id'] not in dv],
        common_full_budget=[r for r in data if all(len(groups[m][r['question_id']]['selection']['selected_frames'])==8 for m in METHODS)])
    for layer in ['short','medium','long']:subsets[layer]=[r for r in data if r['stratum']==layer]
    for task in sorted({r.get('task_type','unknown') for r in data}):subsets['task__'+task]=[r for r in data if r.get('task_type','unknown')==task]
    metrics={m:{name:dict(n=len(rs),correct=sum(groups[m][r['question_id']]['correct'] for r in rs),
        accuracy=sum(groups[m][r['question_id']]['correct'] for r in rs)/len(rs),
        wilson_95=wilson(sum(groups[m][r['question_id']]['correct'] for r in rs),len(rs))) for name,rs in subsets.items() if rs} for m in METHODS}
    pairs=[(a,b) for i,a in enumerate(METHODS) for b in METHODS[:i]]
    comparisons={a+'_minus_'+b:{name:compare(rs,groups[a],groups[b]) for name,rs in subsets.items() if rs} for a,b in pairs}
    old={METHODS[0]:reference(LLAVA_UNIFORM,METHODS[0],data,uniform=True)}
    for method in METHODS[1:3]:old[method]=reference(LLAVA_BASE,method,data)
    old[METHODS[3]]=reference(EXP07/'llava16',METHODS[3],data)
    old[METHODS[4]]=reference(SOURCE/'llava16',METHODS[4],data)
    budget_comparisons={m:{name:compare(rs,old[m],groups[m]) for name,rs in subsets.items() if rs} for m in METHODS}
    primary=[r for group in groups.values() for r in group.values() if r['timing_kind']=='primary']
    performance={}
    for layer in ['all','short','medium','long']:
        values=[r for r in primary if layer=='all' or r['stratum']==layer]
        if values:performance[layer]=dict(n=len(values),observed={k:distribution([r['answer']['observed'][k] for r in values])
            for k in ['qa_seconds','visual_tokens','prefill_tokens','peak_allocated_gib','peak_reserved_gib']},
            timings={k:distribution([r['timings'][k] for r in values]) for k in values[0]['timings']})
    result=dict(status='completed',metrics=metrics,comparisons=comparisons,budget16_minus8=budget_comparisons,
        new_calls=spec['max_calls'],shared_results=13500-spec['max_calls'],performance=performance,
        preparation=read_json(root/'input_execution.json'),execution=read_json(root/'execution.json'),
        parse_failures={m:[q for q,r in groups[m].items() if not r['answer']['parsed_answer']] for m in METHODS},
        workers=[read_json(p) for p in (root/'workers').glob('*.json')],utc=utc())
    durable(root/'summary.json',result)
    flat=[dict(method=m,question_id=q,correct=r['correct'],parsed_answer=r['answer']['parsed_answer'],
        raw_output=r['answer']['raw_output'],representative=r['representative'],gpu=r['gpu']) for m,g in groups.items() for q,r in g.items()]
    with (root/'per_question.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# LLaVA-Video八帧五策略全量结果','','| 方法 | 正确数 | 准确率 | 对均匀增益 |','|---|---:|---:|---:|']
    uniform=metrics[METHODS[0]]['all2700']['accuracy']
    for m in METHODS:
        a=metrics[m]['all2700'];lines.append(f"| {m} | {a['correct']}/2700 | {a['accuracy']:.2%} | {100*(a['accuracy']-uniform):+.2f}个百分点 |")
    lines+=['',f"13500条方法结果，{spec['max_calls']}次唯一新问答，{13500-spec['max_calls']}条同输入共享。",
        '完整十组配对、预算比较、时长/题型/子集和物理调用成本见运行目录summary.json。',
        '同帧共享是输入核验后的引用，不是重复性实验。M8/K8→M16/K16为完整配置变化。数据已被观察，区间为探索性统计。',
        '本轮停止在固定8帧评测，不自动调参；下一步统一review已有模型/预算结果。']
    report='\n'.join(lines)+'\n';(root/'report.md').write_text(report,encoding='utf-8')
    doc=ROOT/'docs/experiments/video_mme_full'/spec['utc'][:10]/'llava8/实验结果.md'
    doc.parent.mkdir(parents=True,exist_ok=True);doc.write_text(report,encoding='utf-8')
    return result
