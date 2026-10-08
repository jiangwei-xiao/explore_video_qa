"""exp27双配置开发问答：输入身份冻结、同输入导入、一次生成与后台阶段切换。"""
import contextlib
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os,queue,shutil,subprocess,time,traceback
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,rows,utc,digest,invoke_once
from videoqa_full.prepare import environment
from videoqa_budget32.frames import FrameProvider,validate_frames
from videoqa_lmms_bridge.common import framework_identity,load_task,task_doc,POST_PROMPT
from videoqa_audit.region_scope_experiments import METHOD,DEV,FILES
from videoqa_audit.framework_contraction import RUN32,RUN8,BASE

PYTHONS={'qwen32':'/App/conda/envs/explore_video_qa_lmms_qwen/bin/python',
         'llava8':'/App/conda/envs/explore_video_qa_lmms/bin/python'}
QA_FILES=('src/videoqa_audit/region_scope_runtime.py','src/videoqa_audit/region_scope_review.py',
          'src/videoqa_budget32/observe.py','src/videoqa_budget32/qwen_adapter.py',
          'src/videoqa_full/qwen_fa2_backend.py','src/videoqa_full/qwen_backend.py',
          'src/videoqa_generalization/llava_observe.py','src/videoqa_generalization/llava_budget8.py',
          'src/videoqa_runtime/llava_backend.py','src/videoqa_lmms_bridge/llava_adapter.py',
          'src/videoqa_lmms_bridge/common.py','src/videoqa_full/state.py')


def development_rows():
    """接口：冻结200题和顺序，不重新抽样、不按旧预测筛题。"""
    release=read_json(ROOT/'configs/releases/rd_1_2_exp07_qwen32_dev200_20260928.json')
    if sha256(DEV)!=release['dataset_manifest_sha256']:raise ValueError('Frozen development manifest changed')
    ids=[r['question_id'] for r in read_json(DEV)['rows']];bank={r['question_id']:r for r in rows()}
    if len(ids)!=200 or len(set(ids))!=200:raise ValueError('Development population differs')
    return [bank[q] for q in ids]


def source_order(selection):
    """接口：候选编号可改变，模型媒体身份按有序源帧与视频时钟定义。"""
    v=selection['video']
    return dict(video={k:v[k] for k in ('path','start_pts','time_base','duration_fraction')},
                frames=[(f['source_frame_index'],f['source_pts']) for f in selection['selected_frames']])


def parent_result(profile,q,method='RD-1.2-exp07'):
    """接口：唯一来源兼容Qwen的200题导入，不择优选择历史答案。"""
    base=RUN32 if profile=='qwen32' else RUN8
    if method in ('BASE-Uniform','BASE-BLIP-TopK') and profile=='qwen32':base=BASE
    if method=='RD-1.2-exp12' and profile=='qwen32':base=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
    choices=[base/'results'/method/(q+'.json'),base/'imported/results'/method/(q+'.json')]
    found=[p for p in choices if p.exists()]
    if len(found)!=1:raise ValueError('Missing or ambiguous parent result: '+q)
    return found[0]


def parent_call_sources(profile,path,old):
    """接口：解析物理主调用，LLaVA同帧共享只引用真正的主账本。"""
    q=old['question_id'];primary=old.get('representative',old['method'])
    base=RUN32 if profile=='qwen32' else RUN8
    folder='imported/' if path.is_relative_to(base/'imported') else ''
    cp=base/(folder+'calls')/primary/(q+'.json');rp=base/(folder+'raw_generations')/primary/(q+'.json')
    c,raw=read_json(cp),read_json(rp)
    if c['status']!='completed' or c['answer']!=old['answer'] or c['protocol_sha256']!=old['protocol_sha256']:
        raise ValueError('Parent physical call not completed')
    if raw['generation_state']!={'started':True,'returned':True}:raise ValueError('Parent raw return uncertain')
    if raw['identity']!=dict(question_id=q,method=primary,protocol_sha256=old['protocol_sha256'],gpu=old['gpu']):
        raise ValueError('Parent raw identity differs')
    return {str(p):sha256(p) for p in (path,cp,rp)}


def cpu_identity(profile,tools,images,row):
    """接口：官方CPU处理器与完整模板→真实像素/Token指纹；不加载语言或视觉模型。"""
    import torch
    from videoqa_budget32.observe import tensor_hash
    processor,tokenizer=tools
    context=load_task().videomme_doc_to_text(task_doc(row),{'post_prompt':POST_PROMPT})
    if profile=='qwen32':
        prompt=processor.apply_chat_template([dict(role='system',content='You are a helpful assistant.'),
            dict(role='user',content=[dict(type='video'),dict(type='text',text=context)])],tokenize=False,add_generation_prompt=True)
        x=processor(text=[prompt],videos=[np.stack([np.asarray(im.convert('RGB')) for im in images])],
            fps=2.0,images_kwargs={'size':dict(shortest_edge=100352,longest_edge=602112)},truncation=False,return_tensors='pt')
        return dict(prompt=prompt,input_token_ids=x['input_ids'][0].tolist(),
            pixel_sha256=tensor_hash(x['pixel_values_videos']),video_grid_thw=x['video_grid_thw'].tolist())
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    conv=conv_templates['qwen_1_5'].copy();conv.append_message(conv.roles[0],'<image>\n'+context);conv.append_message(conv.roles[1],None)
    prompt=conv.get_prompt()
    return dict(prompt=prompt,input_token_ids=tokenizer_image_token(prompt,tokenizer,-200),
                pixel_sha256=tensor_hash(processor.preprocess(images,return_tensors='pt')['pixel_values'].to(torch.bfloat16)))


def prepare_inputs(root,profile,gpus):
    """接口：冻结200组真实输入及同帧导入划分，生成开始前完成全部身份核验。"""
    if profile not in PYTHONS or not gpus or len(set(gpus))!=len(gpus) or not set(gpus)<=set(map(str,range(8))):
        raise ValueError('Invalid profile or GPU scope')
    offline_environment();root=Path(root);run=root/'qa'/profile;run.mkdir(parents=True,exist_ok=True)
    gate=root/'review_gate.json'
    if not gate.exists() or read_json(gate)['status']!='passed':raise ValueError('Offline semantic gate not passed')
    if read_json(gate)['offline_protocol_sha256']!=sha256(root/'offline_protocol.json'):raise ValueError('Review gate source differs')
    if read_json(gate)['offline_summary_sha256']!=sha256(root/'offline_summary.json'):raise ValueError('Approved offline outputs changed')
    offline_exports=read_json(root/'offline_summary.json')['files']
    parent=RUN32 if profile=='qwen32' else RUN8;old_spec=read_json(parent/'protocol.json');data=development_rows()
    k=32 if profile=='qwen32' else 8
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():return verify(run)
        if framework_identity()!=old_spec['framework'] or environment(PYTHONS[profile])!=old_spec['environment']:
            raise ValueError('Evaluation framework or environment changed')
        # 步骤1：共享权重只读核验；关键推理代码必须与父问答谱系一致。
        for p,v in old_spec['model_files'].items():
            if sha256(p)!=v['sha256']:raise ValueError('Frozen model weights changed')
        # 开发视频首次重哈希；第二配置只在路径/大小/mtime及预期摘要完全一致时复用验收。
        video_sources={}
        for vid in sorted({r['video_id'] for r in data}):
            st=old_spec['video_stats'][vid];p=Path(st['path']);actual=p.stat()
            identity=dict(path=str(p),bytes=actual.st_size,mtime_ns=actual.st_mtime_ns,
                          sha256=old_spec['video_hashes'][vid])
            check=root/'video_checks'/(vid+'.json')
            if check.exists():
                if read_json(check)!=identity:raise ValueError('Verified video changed')
            else:
                if sha256(p)!=identity['sha256']:raise ValueError('Video bytes changed')
                durable(check,identity)
            video_sources[str(check)]=sha256(check)
        code={n:sha256(ROOT/n) for n in FILES+QA_FILES}
        for n in QA_FILES:
            if n in old_spec['code_sha256'] and code[n]!=old_spec['code_sha256'][n]:raise ValueError('Parent inference code changed: '+n)
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if profile=='qwen32':
            from transformers import AutoProcessor
            from videoqa_full.state import QWEN_PATH
            tools=(AutoProcessor.from_pretrained(QWEN_PATH,local_files_only=True,use_fast=False,
                min_pixels=100352,max_pixels=602112,size=dict(shortest_edge=100352,longest_edge=602112)),None)
        else:
            from transformers import AutoTokenizer
            from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
            tools=(SigLipImageProcessor(),AutoTokenizer.from_pretrained('/home/models/lmms-lab/LLaVA-Video-7B-Qwen2',local_files_only=True))
        load_task();exports={};imports=[];new=[];sources={str(gate):sha256(gate),str(parent/'protocol.json'):sha256(parent/'protocol.json'),str(DEV):sha256(DEV),**video_sources}
        # 步骤2：独立解码、CPU官方预处理；同源只按输入决定导入，与答对与否无关。
        def one(row):
            q=row['question_id'];path=root/'offline'/profile/(q+'.json');item=read_json(path);s=item['selection']
            if sha256(path)!=offline_exports[str(path)]:raise ValueError('Approved question input changed')
            old_path=parent_result(profile,q);old=read_json(old_path);dest=run/'selections'/METHOD/(q+'.json')
            images,rgb=FrameProvider(s).decode()
            try:identity=cpu_identity(profile,tools,images,row)
            finally:
                for im in images:im.close()
            same=source_order(s)==source_order(old['selection'])
            old_sources={str(old_path):sha256(old_path)}
            if same:
                obs=old['answer']['observed'];pixel=obs['processor_pixel_sha256'] if profile=='qwen32' else obs['pixel_sha256']
                if rgb!=old['rgb_sha256'] or identity['pixel_sha256']!=pixel or identity['input_token_ids']!=obs['input_token_ids'] or identity['prompt']!=obs['prompt']:
                    raise ValueError('Same-source import input mismatch: '+q)
                if profile=='qwen32' and identity['video_grid_thw']!=obs['video_grid_thw']:raise ValueError('Import grid differs')
                old_sources=parent_call_sources(profile,old_path,old)
            value=dict(question_id=q,video_id=row['video_id'],method=METHOD,selection=s,rgb_sha256=rgb,input_identity=identity,
                imported=same,parent_result=str(old_path),parent_sources=old_sources,
                sources={str(path):sha256(path)},frame_budget=k)
            durable(dest,value);return value,dest
        # 预先编译官方模板，避免首次并行懒加载；不调用模型。
        if profile=='qwen32':tools[0].apply_chat_template([dict(role='user',content='synthetic')],tokenize=False,add_generation_prompt=True)
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i,(item,dest) in enumerate(executor.map(one,data),1):
                q=item['question_id'];exports[str(dest.relative_to(run))]=sha256(dest)
                (imports if item['imported'] else new).append(q)
                sources.update(item['sources']);sources.update(item['parent_sources'])
                if i%10==0:durable(run/'prepare_status.json',dict(completed=i,total=200,utc=utc()))
        # 步骤3：调用清单先冻结；200结果与物理新调用额度分开保存。
        spec=dict(profile=profile,method=METHOD,model=old_spec['model'],frame_budget=k,question_ids=[r['question_id'] for r in data],
            imported_ids=imports,new_ids=new,max_new_calls=len(new),max_results=200,allowed_gpus=gpus,
            pilot_question_ids=[r['question_id'] for r in data[:5]],model_files=old_spec['model_files'],
            environment=old_spec['environment'],framework=old_spec['framework'],code_sha256=code,
            selection_exports=exports,sources=sources,video_stats=old_spec['video_stats'],utc=utc())
        for n in code:
            out=run/'code_snapshot'/n;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,out)
        durable(run/'protocol.json',spec)
        import_answers(run,spec);return spec


def verify(run):
    """接口：恢复时来源、环境、代码或权重身份变化暂停；已完成生成不重发。"""
    run=Path(run);spec=read_json(run/'protocol.json')
    if spec['max_new_calls']!=len(spec['new_ids']) or len(spec['question_ids'])!=200:raise ValueError('Call budget differs')
    if set(spec['new_ids'])&set(spec['imported_ids']) or set(spec['new_ids'])|set(spec['imported_ids'])!=set(spec['question_ids']):raise ValueError('Call partition differs')
    if framework_identity()!=spec['framework'] or environment(PYTHONS[spec['profile']])!=spec['environment']:raise ValueError('Environment fingerprint changed')
    for n,h in spec['code_sha256'].items():
        if sha256(ROOT/n)!=h:raise ValueError('Frozen code changed')
    for p,v in spec['model_files'].items():
        st=Path(p).stat()
        if st.st_size!=v['bytes'] or st.st_mtime_ns!=v['mtime_ns']:raise ValueError('Model identity changed')
    for p,h in spec['sources'].items():
        if sha256(p)!=h:raise ValueError('Frozen input source changed')
    for p,h in spec['selection_exports'].items():
        if sha256(run/p)!=h:raise ValueError('Frozen selection changed')
    return spec


def import_answers(run,spec):
    """接口：输入身份已验收才导入，保存物理来源哈希，引用不伪装为新调用。"""
    bank={r['question_id']:r for r in development_rows()};fp=sha256(Path(run)/'protocol.json')
    for q in spec['imported_ids']:
        path=Path(run)/'results'/METHOD/(q+'.json')
        if path.exists():continue
        item=read_json(Path(run)/'selections'/METHOD/(q+'.json'));old=read_json(item['parent_result']);row=bank[q]
        value=dict(status='completed',model=spec['model'],method=METHOD,protocol_sha256=fp,
            **{k:row[k] for k in ('question_id','video_id','question','options','stratum')},
            selection=item['selection'],rgb_sha256=item['rgb_sha256'],answer=old['answer'],correct=old['correct'],
            imported=True,parent_sources=item['parent_sources'],timings={},timing_kind='historical_input_verified_import',utc=utc())
        durable(path,value)


def validate_result(run,row,spec):
    """接口：统一核验新调用及引用，模型实际像素/提示与事前冻结输入一致。"""
    run=Path(run);q=row['question_id'];r=read_json(run/'results'/METHOD/(q+'.json'))
    item=read_json(run/'selections'/METHOD/(q+'.json'));a=r['answer'];o=a['observed'];expected=item['input_identity']
    if r['protocol_sha256']!=sha256(run/'protocol.json') or r['method']!=METHOD or r['model']!=spec['model']:raise ValueError('Result protocol differs')
    if any(r[k]!=row[k] for k in ('question_id','video_id','question','options','stratum')):raise ValueError('Question identity differs')
    if r['selection']!=item['selection'] or r['rgb_sha256']!=item['rgb_sha256']:raise ValueError('Result input differs')
    pixel=o['processor_pixel_sha256'] if spec['profile']=='qwen32' else o['pixel_sha256']
    if pixel!=expected['pixel_sha256'] or o['input_token_ids']!=expected['input_token_ids'] or o['prompt']!=expected['prompt']:raise ValueError('Actual model input differs')
    if o['generate_calls']!=1 or o['prefill_calls']!=1 or not 1<=len(o['generated_token_ids'])<=16:raise ValueError('Generation observations differ')
    n=len(item['selection']['selected_frames']);validate_frames(item['selection'],spec['frame_budget'])
    if spec['profile']=='llava8' and (o['visual_tokens']!=210*n or o['max_new_tokens']!=16):raise ValueError('LLaVA visual protocol differs')
    if spec['profile']=='qwen32' and (o['video_grid_thw']!=expected['video_grid_thw'] or o['generation_kwargs']['max_new_tokens']!=16):raise ValueError('Qwen video protocol differs')
    metric=load_task().videomme_process_results(task_doc(row),[a['raw_output']])['videomme_perception_score']
    if metric!=a['framework_metric'] or r['correct']!=(a['parsed_answer']=='ABCD'[row['answer_index']]):raise ValueError('Independent scoring differs')
    if r['imported']:
        if q not in spec['imported_ids'] or (run/'calls'/METHOD/(q+'.json')).exists():raise ValueError('Import consumed a new call')
        for p,h in r['parent_sources'].items():
            if sha256(p)!=h:raise ValueError('Imported physical source changed')
    else:
        call=read_json(run/'calls'/METHOD/(q+'.json'));raw=read_json(run/'raw_generations'/METHOD/(q+'.json'))
        if q not in spec['new_ids'] or call['status']!='completed' or call['answer']!=a:raise ValueError('New call identity differs')
        if raw['identity']!=dict(question_id=q,method=METHOD,protocol_sha256=r['protocol_sha256'],gpu=r['gpu']) or raw['generation_state']!={'started':True,'returned':True}:raise ValueError('Raw generation uncertain')
    return r


def recover(run,row,spec):
    """接口：仅完整返回可恢复后处理，未知生成状态禁止重复。"""
    run=Path(run);p=run/'calls'/METHOD/(row['question_id']+'.json')
    if not p.exists():return
    c=read_json(p)
    if c['protocol_sha256']!=sha256(run/'protocol.json'):raise ValueError('Call fingerprint changed')
    if c['status']=='completed':return
    raw=read_json(run/'raw_generations'/METHOD/(row['question_id']+'.json'))
    if raw['identity']!=dict(question_id=row['question_id'],method=METHOD,protocol_sha256=c['protocol_sha256'],gpu=c['gpu']):raise ValueError('Raw return identity changed')
    if raw['generation_state']!={'started':True,'returned':True}:raise ValueError('Generation return uncertain')
    if spec['profile']=='qwen32':
        from videoqa_budget32.observe import result_from_raw
        answer=result_from_raw(raw,row)
    else:
        from videoqa_generalization.llava_observe import result_from_raw
        restored=result_from_raw(raw,row);answer=dict(restored['answer'],observed=restored['observed'])
    c.update(status='completed',answer=answer,generation_state={'started':True,'returned':True},recovered_postprocessing=True)
    durable(p,c)


def process(run,row,backend,adapter,gpu,spec):
    """接口：精确取帧→一次框架调用→立即保存答案；正确性不触发重跑。"""
    run=Path(run);q=row['question_id'];dest=run/'results'/METHOD/(q+'.json')
    if dest.exists():return validate_result(run,row,spec)
    if q not in spec['new_ids']:raise ValueError('Question outside frozen new-call plan')
    with exclusive(run/'locks'/(q+'.lock')):
        recover(run,row,spec);item=read_json(run/'selections'/METHOD/(q+'.json'));start=time.perf_counter();images=[]
        p=run/'calls'/METHOD/(q+'.json')
        if not p.exists():
            st=spec['video_stats'][row['video_id']];current=Path(st['path']).stat()
            if current.st_size!=st['bytes'] or current.st_mtime_ns!=st['mtime_ns']:raise ValueError('Runtime source video changed')
            images,rgb=FrameProvider(item['selection']).decode()
            if rgb!=item['rgb_sha256']:raise ValueError('Runtime source pixels changed')
            # 生成前重建实际后端的CPU输入，防止不符输入已经消耗一次正式调用。
            actual=cpu_identity(spec['profile'],(backend.processor,backend.tokenizer if spec['profile']=='llava8' else None),images,row)
            if actual!=item['input_identity']:raise ValueError('Runtime processor input changed before generation')
        decoded=time.perf_counter();fp=sha256(run/'protocol.json');identity=dict(question_id=q,method=METHOD,protocol_sha256=fp,gpu=gpu)
        def invoke(state):
            if spec['profile']=='qwen32':
                from videoqa_budget32.observe import observe
                return observe(backend,adapter,images,row,state,run/'raw_generations'/METHOD/(q+'.json'),identity)
            from videoqa_generalization.llava_observe import observe
            value=observe(backend,adapter,images,item['selection'],row,state,run/'raw_generations'/METHOD/(q+'.json'),identity)
            return dict(value['answer'],observed=value['observed'])
        try:answer,reused,checkpoint=invoke_once(run,q,METHOD,fp,gpu,invoke)
        finally:
            for im in images:im.close()
        call=read_json(p)
        if call['gpu']!=gpu:raise ValueError('Partial question moved physical GPU')
        result=dict(status='completed',model=spec['model'],method=METHOD,protocol_sha256=fp,
            **{k:row[k] for k in ('question_id','video_id','question','options','stratum')},
            selection=item['selection'],rgb_sha256=item['rgb_sha256'],answer=answer,gpu=gpu,imported=False,
            correct=answer['parsed_answer']=='ABCD'[row['answer_index']],
            timings=dict(final_decode_seconds=decoded-start,cached_input_to_persisted_answer_seconds=time.perf_counter()-start,
                         call_start_checkpoint_seconds=checkpoint),utc=utc())
        durable(dest,result);return validate_result(run,row,spec)


def worker(run,gpu,jobs,events,stop):
    """接口：一卡长期后端，合成预热单列，异常立即停止派发与生成。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    offline_environment();run=Path(run);active=None
    with (run/'logs'/('gpu_'+gpu+'.log')).open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import torch
            from PIL import Image
            from videoqa_lmms_bridge.bridge import synthetic_row
            torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(2027)
            spec=read_json(run/'protocol.json');t=time.perf_counter()
            if spec['profile']=='qwen32':
                from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
                from videoqa_budget32.qwen_adapter import FrozenInputQwen
                from videoqa_budget32.observe import observe
                backend=QwenFA2VideoBackend();adapter=FrozenInputQwen(backend)
            else:
                from videoqa_runtime.llava_backend import LlavaBackend
                from videoqa_generalization.llava_budget8 import adapter_for
                from videoqa_generalization.llava_observe import observe
                backend=LlavaBackend();adapter=adapter_for(backend)
            loading=time.perf_counter()-t;n=spec['frame_budget'];images=[Image.new('RGB',(896,672),(30,10+i*3,150)) for i in range(n)]
            warm_path=run/'workers'/('gpu_'+gpu+'.json');durable(warm_path,dict(status='started_synthetic',gpu=gpu,utc=utc()))
            t=time.perf_counter()
            if spec['profile']=='qwen32':warm=observe(backend,adapter,images,synthetic_row(n),{})
            else:
                s=dict(video=dict(path='synthetic.mp4',duration_seconds=float(n)),selected_frames=[dict(timestamp_seconds=float(i)) for i in range(n)])
                warm=observe(backend,adapter,images,s,synthetic_row(n),{})
            for im in images:im.close()
            durable(warm_path,dict(status='completed',gpu=gpu,pid=os.getpid(),sid=os.getsid(0),load_seconds=loading,
                warmup_seconds=time.perf_counter()-t,synthetic_generations=1,warmup=warm,load_report=backend.load_report))
            events.put(dict(type='ready',gpu=gpu))
            # 步骤1：只消费新调用清单；一个任务一个账本，不新增原输入复测。
            while not stop.is_set():
                row=jobs.get()
                if row is None:break
                active=row['question_id'];events.put(dict(type='active',gpu=gpu,question_id=active))
                process(run,row,backend,adapter,gpu,spec)
                events.put(dict(type='completed',gpu=gpu,question_id=active));active=None
        except BaseException as error:
            stop.set();traceback.print_exc();durable(run/'failures'/('gpu_'+gpu+'.json'),dict(error=repr(error),question_id=active,utc=utc()))
            events.put(dict(type='failed',gpu=gpu,error=repr(error)))


def execute(run):
    """接口：先导入/恢复再分配空闲卡，完成态不加载模型或增加生成。"""
    from videoqa_runtime.baseline_run import available_gpus
    run=Path(run);spec=verify(run);data=development_rows();bank={r['question_id']:r for r in data}
    with exclusive(run/'coordinator.lock'):
        import_answers(run,spec);done=set();pinned={}
        for p in (run/'calls'/METHOD).glob('*.json'):
            if p.stem not in spec['new_ids']:raise ValueError('Unexpected formal call')
            recover(run,bank[p.stem],spec);pinned[p.stem]=read_json(p)['gpu']
        if len(pinned)>spec['max_new_calls']:raise ValueError('Formal call budget exceeded')
        for p in (run/'results'/METHOD).glob('*.json'):
            if p.stem not in bank:raise ValueError('Unknown result')
            validate_result(run,bank[p.stem],spec);done.add(p.stem)
        if len(done)==200:
            durable(run/'pilot_acceptance.json',dict(status='passed',accuracy_not_gate=True,question_ids=spec['pilot_question_ids']))
            return audit(run)
        for name in ('logs','workers','failures'):(run/name).mkdir(exist_ok=True)
        required={g for q,g in pinned.items() if q not in done};gpus=[]
        while not gpus:
            try:free,states=available_gpus(spec['allowed_gpus'])
            except RuntimeError:free=[];states={}
            if free and required<=set(free):
                ordered=sorted(required)+[g for g in free if g not in required]
                gpus=ordered[:min(len(ordered),len(spec['new_ids'])-len(done-set(spec['imported_ids'])))]
            else:
                durable(run/'status.json',dict(status='waiting_resources',completed=len(done),total=200,utc=utc()));time.sleep(15)
        durable(run/'gpu_preflight.json',dict(actual=gpus,states=states,utc=utc()))
        ctx=mp.get_context('spawn');jobs={g:ctx.Queue() for g in gpus};events=ctx.Queue();stop=ctx.Event();children={};busy=set();ready=set();active={}
        started=time.perf_counter();phase='loading';pending=[]
        def schedule():
            nonlocal phase,pending
            pilot=spec['pilot_question_ids'];missing=[q for q in pilot if q not in done]
            phase='pilot' if missing else 'running'
            if not missing:
                for q in pilot:validate_result(run,bank[q],spec)
                durable(run/'pilot_acceptance.json',dict(status='passed',accuracy_not_gate=True,question_ids=pilot))
            pending=missing or sorted([q for q in bank if q not in done],key=lambda q:(-read_json(run/'selections'/METHOD/(q+'.json'))['selection']['video']['duration_seconds'],q))
        def refill():
            for g in gpus:
                if g in busy:continue
                q=next((q for q in pending if q not in pinned or pinned[q]==g),None)
                if q is not None:pending.remove(q);busy.add(g);jobs[g].put(bank[q])
        # 步骤1：后台完成先行放行和动态队列；观察断线不重启子任务。
        try:
            for g in gpus:
                p=ctx.Process(target=worker,args=(str(run),g,jobs[g],events,stop));p.start();children[g]=p
            while len(done)<200:
                try:e=events.get(timeout=5)
                except queue.Empty:e=None
                if e:
                    if e['type']=='failed':raise RuntimeError(str(e))
                    if e['type']=='ready':
                        ready.add(e['gpu'])
                        if len(ready)==len(gpus):schedule();refill()
                    elif e['type']=='active':active[e['gpu']]=e['question_id']
                    elif e['type']=='completed':
                        done.add(e['question_id']);busy.remove(e['gpu']);active.pop(e['gpu'],None)
                        if phase=='pilot' and not pending and not busy:schedule()
                        refill()
                for g,p in children.items():
                    if p.exitcode is not None:raise RuntimeError('Worker exited before completion')
                elapsed=time.perf_counter()-started
                durable(run/'status.json',dict(status=phase,completed=len(done),total=200,new_completed=len(done)-len(spec['imported_ids']),
                    new_total=spec['max_new_calls'],pid=os.getpid(),sid=os.getsid(0),gpus=gpus,active=active,elapsed_seconds=elapsed,utc=utc()))
        except BaseException as error:
            stop.set();durable(run/'status.json',dict(status='paused',error=repr(error),completed=len(done),total=200,utc=utc()));raise
        finally:
            for g in gpus:jobs[g].put(None)
            for p in children.values():
                p.join(20)
                if p.is_alive():p.terminate();p.join()
        # 步骤2：只有完整结果才汇总，恢复时不重复执行这一批生成。
        durable(run/'execution.json',dict(seconds=time.perf_counter()-started,actual_gpus=gpus,
            new_calls=len(list((run/'calls'/METHOD).glob('*.json'))),utc=utc()))
        return audit(run)


def audit(run):
    """接口：200结果、事前导入划分和物理账本闭合，调用失败不能伪装完成。"""
    run=Path(run);spec=verify(run);data=development_rows();expected=set(spec['question_ids'])
    if {p.stem for p in (run/'results'/METHOD).glob('*.json')}!=expected:raise ValueError('Incomplete results')
    if {p.stem for p in (run/'calls'/METHOD).glob('*.json')}!=set(spec['new_ids']):raise ValueError('Call ledger population differs')
    if {p.stem for p in (run/'raw_generations'/METHOD).glob('*.json')}!=set(spec['new_ids']):raise ValueError('Raw return population differs')
    for row in data:validate_result(run,row,spec)
    files=list((run/'results'/METHOD).glob('*.json'))+list((run/'calls'/METHOD).glob('*.json'))+list((run/'raw_generations'/METHOD).glob('*.json'))
    value=dict(status='passed',results=200,imports=len(spec['imported_ids']),new_calls=len(spec['new_ids']),
        artifacts={str(p):sha256(p) for p in files},utc=utc())
    previous=run/'completion_audit.json'
    if previous.exists() and read_json(previous)['artifacts']!=value['artifacts']:raise ValueError('Completion recovery changed generation files')
    durable(previous,value);return value


def report(root):
    """接口：开发成绩、配对得失与输入变化分开统计，不外推全量收益。"""
    from videoqa_runtime.baseline_report import wilson,distribution
    from videoqa_full.report import compare
    root=Path(root);data=development_rows();profiles={}
    for profile in PYTHONS:
        run=root/'qa'/profile;check=audit(run);spec=read_json(run/'protocol.json')
        results={r['question_id']:validate_result(run,r,spec) for r in data}
        refs={m:{r['question_id']:read_json(parent_result(profile,r['question_id'],m)) for r in data}
              for m in ('RD-1.2-exp07','RD-1.2-exp12','BASE-Uniform','BASE-BLIP-TopK')}
        expected=136 if profile=='qwen32' else 126
        if sum(x['correct'] for x in refs['RD-1.2-exp07'].values())!=expected:raise ValueError('Parent development reference differs')
        metrics={}
        for layer in ('all','short','medium','long'):
            group=[results[r['question_id']] for r in data if layer=='all' or r['stratum']==layer];c=sum(x['correct'] for x in group)
            metrics[layer]=dict(n=len(group),correct=c,accuracy=c/len(group),wilson_95=wilson(c,len(group)))
        fresh=[x for x in results.values() if not x['imported']]
        profiles[profile]=dict(results=metrics,comparisons={m:compare(data,results,rs) for m,rs in refs.items()},
            imported=len(spec['imported_ids']),new_calls=len(spec['new_ids']),
            parse_failures=[q for q,x in results.items() if x['answer']['parsed_answer'] is None],
            new_qa_seconds=distribution([x['answer']['observed']['qa_seconds'] for x in fresh]),
            completion_audit=check,actual_frame_counts={str(n):sum(len(x['selection']['selected_frames'])==n for x in results.values())
                for n in sorted({len(x['selection']['selected_frames']) for x in results.values()})})
    q=profiles['qwen32']['results']['all']['correct'];l=profiles['llava8']['results']['all']['correct']
    value=dict(method=METHOD,profiles=profiles,unified_non_degrading_candidate=q>=136 and l>=126,
        full_validation_priority=q>=136 and l>=126 and (q>=140 or l>=128),
        additional_full_qa=0,scope='Repeated development200; input imports verified before generation; no new full evaluation',utc=utc())
    durable(root/'summary.json',value);return value


def dispatch(root,stage,profile,gpus):
    """接口：跨环境阶段显式传参，后台自动完成两个工作点，不调用全量入口。"""
    root=Path(root)
    if stage=='all':
        with exclusive(root/'pipeline.lock'):
            for p,python in PYTHONS.items():
                for sub in ('prepare-inputs','qa'):
                    durable(root/'pipeline_status.json',dict(stage=sub,profile=p,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                    subprocess.run([python,str(ROOT/'scripts/run_region_scope_experiments.py'),'--run-id',root.name,
                        '--stage',sub,'--profile',p,'--gpus',*gpus],cwd=ROOT,check=True)
            subprocess.run([PYTHONS['qwen32'],str(ROOT/'scripts/run_region_scope_experiments.py'),'--run-id',root.name,'--stage','report'],cwd=ROOT,check=True)
            durable(root/'pipeline_status.json',dict(status='completed',utc=utc()))
    elif stage=='prepare-inputs':prepare_inputs(root,profile,gpus)
    elif stage=='qa':execute(root/'qa'/profile)
    elif stage=='audit':
        for p in PYTHONS:audit(root/'qa'/p)
    else:report(root)
