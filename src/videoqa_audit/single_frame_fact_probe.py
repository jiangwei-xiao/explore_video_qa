"""单帧事实最小接口诊断：48个固定源图、程序绑定来源、独立辅助调用账本。"""
from pathlib import Path
import copy
import html
import json
import os
import re
import shutil
import subprocess
import time
import traceback

import numpy as np
from PIL import Image

from videoqa_runtime.common import ROOT, read_json, sha256, offline_environment
from videoqa_full.state import durable, digest, exclusive, invoke_once, utc
from videoqa_full.prepare import environment
from videoqa_full.qwen_backend import MIN_PIXELS, MAX_PIXELS
from videoqa_audit.evidence_observation_backend import EvidenceObservationBackend, _hash_array
from videoqa_audit.evidence_observation_probe import load_tools, _read_images, publish

PYTHON='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'
PARENT=ROOT/'outputs/diagnostics/evidence_observation__pilot12__20261016_r1'
FILES=('src/videoqa_audit/single_frame_fact_probe.py','scripts/probe_single_frame_facts.py')
INSTRUCTION='''Inspect the one still image supplied with this request.
Use the question only to decide what visible details to inspect.
Record legible text verbatim in visible_text, and directly visible objects,
attributes, or physical states in observations. Omit uncertain details.
Describe what is visible; do not provide an option letter or a final video answer.
Do not infer unseen actions, intentions, causes, personal identities, whole-video
ordering, or total event counts. First/last wording may be quoted as visible text.
Return one JSON object with exactly two keys: visible_text and observations.
Both values must be arrays of strings, with at most three items in each array.
Use empty arrays when no relevant content can be established.
Do not output source identifiers, timestamps, Markdown, or other explanation.'''


def parse_facts(raw, truncated=False):
    """接口：固定允许完整单层围栏；不截取正文、不补键、不修复事实，截断即失败。"""
    stripped=raw.strip();match=re.fullmatch(r'```(?:json)?[ \t]*\n(.*)\n```',stripped,re.S)
    body=match.group(1) if match else stripped
    def decode(text):
        value=json.loads(text)
        if not isinstance(value,dict) or set(value)!={'visible_text','observations'}:
            raise ValueError('unexpected_json_shape')
        for values in value.values():
            if not isinstance(values,list) or len(values)>3 or any(not isinstance(x,str) or not x.strip() for x in values):
                raise ValueError('invalid_fact_array')
        return value
    try:
        if truncated:raise ValueError('generation_limit_without_eos')
        value=decode(body)
        try:decode(stripped);strict=True
        except (ValueError,TypeError):strict=False
        return dict(valid=True,strict_valid=strict,outer_fence=bool(match),value=value,error=None)
    except (ValueError,TypeError) as error:
        return dict(valid=False,strict_valid=False,outer_fence=bool(match),value=None,error=str(error))


def fact_text(question):
    """接口：模型文本只接收原问题；来源ID、PTS和人工事实不进入该函数。"""
    if not isinstance(question,str) or not question.strip():raise ValueError('Original question required')
    return INSTRUCTION+'\n\nQuestion:\n'+question


def prepare_single(processor, frames, question, frame_ids, times, config):
    """接口：一个RGB→原官方单帧视频处理；来源信息仅写日志，不进入prompt。"""
    if len(frames)!=1 or len(frame_ids)!=1 or len(times)!=1:raise ValueError('Exactly one source frame required')
    source=np.ascontiguousarray(np.asarray(frames[0].convert('RGB')))
    # 步骤1：保持旧单帧的内部自身复制和官方像素处理，不引入新的源证据。
    message=[dict(role='system',content='You are a helpful assistant.'),dict(role='user',content=[
        dict(type='video'),dict(type='text',text=fact_text(question))])]
    prompt=processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[prompt],videos=[np.stack([source,source])],fps=2.0,
        images_kwargs={'size':{'shortest_edge':MIN_PIXELS,'longest_edge':MAX_PIXELS}},
        padding=False,truncation=False,return_tensors='pt')
    grid=inputs['video_grid_thw'].tolist();height,width=grid[0][1]*14,grid[0][2]*14
    visual=int(np.prod(grid[0])//4)
    if len(grid)!=1 or grid[0][0]!=1 or height%28 or width%28 or not MIN_PIXELS<=height*width<=MAX_PIXELS:
        raise ValueError('Unexpected official single-frame geometry')
    if int((inputs['input_ids']==config.video_token_id).sum())!=visual or int((inputs['input_ids']==config.image_token_id).sum()):
        raise ValueError('Media placeholder identity differs')
    if list(inputs['second_per_grid_ts'])!=[1.0] or len(inputs['input_ids'][0])+256>config.max_position_embeddings:
        raise ValueError('Encoding FPS or input budget differs')
    # 步骤2：ID/真实时间只留在程序身份中，避免模型把提示元数据当画面文字。
    identity=dict(prompt=prompt,input_token_ids=inputs['input_ids'][0].tolist(),
        token_sha256=_hash_array(inputs['input_ids'].numpy()),pixel_sha256=_hash_array(inputs['pixel_values_videos'].numpy()),
        source_rgb_sha256=[_hash_array(source)],frame_ids=list(frame_ids),source_times_seconds=list(times),
        source_frames=1,encoded_frames=2,internal_padding_frames=1,video_grid_thw=grid,
        processed_size=[height,width],visual_tokens=visual,prefill_tokens=len(inputs['input_ids'][0]),
        encoding_fps=2.0,max_new_tokens=256)
    return inputs,identity


class SingleFrameBackend(EvidenceObservationBackend):
    """接口：复用已验收加载/生成/原文检查点，只替换本轮输入文本与解析合同。"""
    def prepare(self,frames,question,frame_ids,times):
        """步骤1：实际生成前重新计算官方CPU输入，供调用层逐字节核对。"""
        return prepare_single(self.processor,frames,question,frame_ids,times,self.model.config)

    def generate_prepared(self,inputs,identity,state,raw_path,call_identity):
        """步骤2：父生成器先落盘原文；旧三字段解析不采用，按本轮双数组合同解析。"""
        answer=super().generate_prepared(inputs,identity,state,raw_path,call_identity)
        answer['parsed']=parse_facts(answer['raw_output'],answer['metadata']['truncated'])
        return answer


def prepare(root):
    """接口：按每组2/4号固定导出48图及旧输出；CPU准备不运行模型。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);offline_environment()
    with exclusive(root/'prepare.lock'):
        if (root/'assets.json').exists():return verify_assets(root)
        began=time.perf_counter();parent=read_json(PARENT/'assets.json')
        if environment(PYTHON)!=parent['environment']:raise ValueError('Frozen environment differs')
        # 步骤1：核查旧来源和冻结算法；旧运行目录只读，不调用会改写其审计的入口。
        sources=dict(parent['sources'])
        for path,old in sources.items():
            if sha256(path)!=old:raise ValueError('Parent source changed: '+path)
        for p in (PARENT/'assets.json',PARENT/'protocol.json',PARENT/'evaluation_reference.json'):
            sources[str(p)]=sha256(p)
        for name in FILES:
            p=ROOT/name;sources[str(p)]=sha256(p)
            dest=root/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        processor,config=load_tools();jobs=[]
        for group in parent['groups']:
            for frame in group['frames']:
                if frame['frame_id'] not in (2,4):continue
                jid=group['group_id']+'__single_'+str(frame['frame_id'])
                old_result=PARENT/'results'/(jid+'.json');old_input=PARENT/'inputs'/(jid+'.json')
                for p in (old_result,old_input):sources[str(p)]=sha256(p)
                entry=copy.deepcopy(frame);dest=root/'images'/(jid+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(entry['image'],dest);entry['parent_image']=entry['image'];entry['image']=str(dest)
                sources[str(dest)]=sha256(dest)
                job=dict(job_id=jid,case_id=group['case_id'],group_id=group['group_id'],kind='single',
                    dataset=group['dataset'],question=group['question'],frames=[entry],parent_result=str(old_result))
                ims=_read_images(job)
                try:_,identity=prepare_single(processor,ims,job['question'],[entry['frame_id']],[entry['timestamp_seconds']],config)
                finally:
                    for im in ims:im.close()
                old=read_json(old_result)['answer']['metadata']
                for key in ('pixel_sha256','source_rgb_sha256','source_frames','encoded_frames','internal_padding_frames',
                            'video_grid_thw','processed_size','visual_tokens','encoding_fps'):
                    if identity[key]!=old[key]:raise ValueError('Same-image preprocessing differs: '+jid+' '+key)
                job['input_identity']=identity;path=root/'inputs'/(jid+'.json');publish(path,job)
                jobs.append(dict(job_id=jid,case_id=job['case_id'],input_file=str(path),input_sha256=sha256(path)))
        if len(jobs)!=48 or len({x['job_id'] for x in jobs})!=48:raise ValueError('Expected 48 fixed inputs')
        assets=dict(task='single_frame_facts',jobs=jobs,sources=sources,environment=parent['environment'],
            source_stats=parent['source_stats'],code={str(ROOT/n):sha256(ROOT/n) for n in FILES},
            prompt=INSTRUCTION,real_auxiliary_budget=48,synthetic_budget=2,final_qa_budget=0,
            prepare_seconds=time.perf_counter()-began,utc=utc())
        publish(root/'assets.json',assets);render(root);return assets


def verify_assets(root):
    """接口：恢复核对源码/输入/大文件stat；完整来源哈希在准备及最终审计核对。"""
    root=Path(root);assets=read_json(root/'assets.json')
    for path,fp in assets['code'].items():
        if sha256(path)!=fp:raise ValueError('Probe code changed')
    for path,st in assets['source_stats'].items():
        actual=Path(path).stat()
        if (actual.st_size,actual.st_mtime_ns)!=(st['bytes'],st['mtime_ns']):raise ValueError('Source stat changed')
    for item in assets['jobs']:
        if sha256(item['input_file'])!=item['input_sha256']:raise ValueError('Input changed')
    if environment(PYTHON)!=assets['environment']:raise ValueError('Environment changed')
    return assets


def freeze_reference(root):
    """接口：48张单帧各≤3目标事实，必须在任何辅助调用开始前冻结。"""
    root=Path(root)
    if (root/'protocol.json').exists():return verify(root)
    assets=verify_assets(root);ref=read_json(root/'evaluation_reference.json')
    if set(ref['items'])!={j['job_id'] for j in assets['jobs']} or ref.get('stage')!='before_model_generation':
        raise ValueError('Expected exact 48 pre-model annotations')
    for item in ref['items'].values():
        facts=item['facts']
        if len(facts)>3 or any(set(f)!={'id','kind','text'} or f['kind'] not in ('text','object','state') or not f['text'].strip() for f in facts):
            raise ValueError('Invalid per-image target facts')
        if item['negative']!= (not facts):raise ValueError('Negative frame reference differs')
    if list((root/'calls').glob('*/*.json')):raise ValueError('Reference must precede all model calls')
    value=dict(task='single_frame_facts',assets_sha256=sha256(root/'assets.json'),
        reference_sha256=sha256(root/'evaluation_reference.json'),real_budget=48,synthetic_budget=2,
        final_qa_budget=0,format_normalization='whole-response-single-json-or-bare-fence-v1',utc=utc())
    publish(root/'protocol.json',value);return value


def verify(root):
    """接口：调用前核对冻结的提示、输入和人工评价版本，不接受指纹漂移。"""
    root=Path(root);verify_assets(root);p=read_json(root/'protocol.json')
    if p['assets_sha256']!=sha256(root/'assets.json') or p['reference_sha256']!=sha256(root/'evaluation_reference.json'):
        raise ValueError('Frozen protocol differs')
    return p


def restore_raw(root,jid,method='facts'):
    """接口：完成调用只读；完整原始返回只恢复解析；不确定调用禁止重发。"""
    root=Path(root);path=root/'calls'/method/(jid+'.json')
    if not path.exists():return None
    call=read_json(path)
    if call['protocol_sha256']!=sha256(root/'protocol.json'):raise ValueError('Call fingerprint changed')
    if call['status']=='completed':return call['answer']
    rawpath=root/'raw_generations'/method/(jid+'.json')
    if not rawpath.exists():raise RuntimeError('Uncertain generation; no retry')
    raw=read_json(rawpath)
    if raw['generation_state']!={'started':True,'returned':True} or raw['identity']!=dict(
        job_id=jid,method=method,protocol_sha256=call['protocol_sha256'],gpu=call['gpu']):
        raise RuntimeError('Uncertain raw identity; no retry')
    answer=dict(raw_output=raw['raw_output'],metadata=raw['metadata'],
        parsed=parse_facts(raw['raw_output'],raw['metadata']['truncated']))
    call.update(status='completed',answer=answer,recovered_postprocessing=True,generation_state=raw['generation_state'])
    durable(path,call);return answer


def process_job(root,job,backend,gpu,method='facts'):
    """接口：唯一源图身份核验→生成前记账→原文即时检查点→程序来源关联。"""
    root=Path(root);jid=job['job_id'];dest=root/'results'/(jid+'.json')
    with exclusive(root/'job_locks'/(method+'_'+jid+'.lock')):
        answer=restore_raw(root,jid,method)
        if method=='facts' and dest.exists():return read_json(dest)
        if answer is None:
            if (root/'PAUSE.json').exists():raise RuntimeError('Probe paused')
            ims=_read_images(job)
            try:
                # 步骤1：原问题/源像素及处理张量均先核验，再占用生成额度。
                began=time.perf_counter();f=job['frames'][0]
                inputs,identity=backend.prepare(ims,job['question'],[f['frame_id']],[f['timestamp_seconds']])
                if identity!=job['input_identity']:raise ValueError('Actual input differs')
                preprocess=time.perf_counter()-began;fp=sha256(root/'protocol.json')
                def invoke(state):
                    return backend.generate_prepared(inputs,identity,state,root/'raw_generations'/method/(jid+'.json'),
                        dict(job_id=jid,method=method,protocol_sha256=fp,gpu=gpu))
                answer,_,checkpoint=invoke_once(root,jid,method,fp,gpu,invoke)
                answer=dict(answer,preprocessing_seconds=preprocess,checkpoint_seconds=checkpoint)
            finally:
                for im in ims:im.close()
        # 步骤2：来源绑定只说明输入身份，不把模型陈述自动标成正确。
        value=dict(job_id=jid,gpu=gpu,input_sha256=digest(job),program_source=job['frames'][0],answer=answer,utc=utc())
        if method=='facts':durable(dest,value)
        return value


def worker(root,gpu):
    """接口：同卡长期进程，一次合成后处理指定题；进程恢复不重做合成或未知调用。"""
    root=Path(root);verify(root);assets=read_json(root/'assets.json');assignment=read_json(root/'assignment.json')
    with exclusive(root/'worker_locks'/(gpu+'.lock')):
        session=root/'workers'/(gpu+'.json')
        assigned=[j for j in assets['jobs'] if j['case_id'] in assignment['cases'][gpu]]
        if session.exists():
            for item in assigned:
                restored=restore_raw(root,item['job_id'])
                if restored is not None:process_job(root,read_json(item['input_file']),None,gpu)
            if all((root/'results'/(j['job_id']+'.json')).exists() for j in assigned):return
            raise RuntimeError('Previous process interrupted; no automatic generation restart')
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(2027)
        durable(session,dict(status='loading',pid=os.getpid(),sid=os.getsid(0),gpu=gpu,utc=utc()))
        backend=SingleFrameBackend();durable(root/'load_reports'/(gpu+'.json'),backend.load_report)
        # 步骤1：每进程只运行一次单帧合成，独立额度；不据内容调提示。
        im=Image.new('RGB',(640,360),(30,100,200));path=root/'synthetic'/gpu/'surface.png';path.parent.mkdir(parents=True,exist_ok=True);im.save(path)
        _,identity=backend.prepare([im],'What visible colors or objects are present?',[1],[0.0])
        job=dict(job_id='warmup',question='What visible colors or objects are present?',
            frames=[dict(frame_id=1,timestamp_seconds=0.0,image=str(path),rgb_sha256=_hash_array(np.asarray(im)))],input_identity=identity)
        im.close();process_job(root,job,backend,gpu,'synthetic_'+gpu)
        # 步骤2：48图事前固定，不根据输出正确性改变后续顺序或提示。
        for item in assigned:
            if (root/'PAUSE.json').exists():return
            durable(session,dict(status='running',pid=os.getpid(),gpu=gpu,job_id=item['job_id'],utc=utc()))
            process_job(root,read_json(item['input_file']),backend,gpu)
        durable(session,dict(status='completed',pid=os.getpid(),gpu=gpu,real_jobs=len(assigned),utc=utc()))


def run(root):
    """接口：最多两张空闲卡，后台协调全部48调用及技术审计；完成恢复不加载模型。"""
    root=Path(root);verify(root);assets=read_json(root/'assets.json')
    with exclusive(root/'coordinator.lock'):
        if all((root/'results'/(j['job_id']+'.json')).exists() for j in assets['jobs']):return audit(root)
        if (root/'PAUSE.json').exists():raise RuntimeError('Paused run requires inspection')
        assignment=root/'assignment.json'
        if assignment.exists():gpus=list(read_json(assignment)['cases'])
        else:
            from videoqa_runtime.baseline_run import available_gpus
            while True:
                try:free,_=available_gpus(list(map(str,range(8))))
                except RuntimeError as error:
                    if str(error)!='No requested GPU is currently idle':raise
                    durable(root/'heartbeat.json',dict(status='waiting_for_idle_gpu',pid=os.getpid(),utc=utc()));time.sleep(5);continue
                gpus=free[-2:];break
            cases=list(dict.fromkeys(j['case_id'] for j in assets['jobs']))
            publish(assignment,dict(cases={g:cases[i::len(gpus)] for i,g in enumerate(gpus)},
                uuids={g:subprocess.check_output(['nvidia-smi','-i',g,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip() for g in gpus}))
        children=[]
        try:
            for gpu in gpus:
                log=(root/('worker_'+gpu+'.log')).open('a',buffering=1)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
                p=subprocess.Popen([PYTHON,str(ROOT/'scripts/probe_single_frame_facts.py'),'--run-id',root.name,'--stage','worker','--gpu',gpu],
                    cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);children.append((gpu,p,log))
            while any(p.poll() is None for _,p,_ in children):
                if any(p.poll() not in (None,0) for _,p,_ in children):raise RuntimeError('Worker failed; no retry')
                durable(root/'heartbeat.json',dict(status='generating',pid=os.getpid(),sid=os.getsid(0),
                    workers={g:p.pid for g,p,_ in children},completed=len(list((root/'results').glob('*.json'))),total=48,utc=utc()));time.sleep(3)
            if any(p.returncode for _,p,_ in children):raise RuntimeError('Worker exited abnormally')
            value=audit(root);durable(root/'heartbeat.json',dict(status='generation_completed',completed=48,total=48,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            return value
        except BaseException as error:
            durable(root/'PAUSE.json',dict(error=repr(error),traceback=traceback.format_exc(),utc=utc()));raise
        finally:
            for _,p,log in children:
                if p.poll() is None:p.wait()
                log.close()


def audit(root):
    """接口：48真实/≤2合成闭合，逐项复核输入与原文；不将格式合法计为事实正确。"""
    root=Path(root);verify(root);assets=read_json(root/'assets.json');jobs={j['job_id']:j for j in assets['jobs']}
    calls=list((root/'calls/facts').glob('*.json'))
    if {p.stem for p in calls}!=set(jobs) or {p.stem for p in (root/'results').glob('*.json')}!=set(jobs):raise ValueError('48 results/calls do not close')
    assignment=read_json(root/'assignment.json');owners={case:g for g,cs in assignment['cases'].items() for case in cs}
    for jid,item in jobs.items():
        job=read_json(item['input_file']);out=read_json(root/'results'/(jid+'.json'));call=read_json(root/'calls/facts'/(jid+'.json'))
        raw=read_json(root/'raw_generations/facts'/(jid+'.json'));answer=out['answer'];meta=answer['metadata']
        if call['status']!='completed' or raw['generation_state']!={'started':True,'returned':True}:raise ValueError('Incomplete generation')
        if raw['identity']!=dict(job_id=jid,method='facts',protocol_sha256=sha256(root/'protocol.json'),gpu=owners[job['case_id']]):raise ValueError('GPU/call identity differs')
        if raw['raw_output']!=answer['raw_output'] or raw['metadata']!=meta or call['answer']['raw_output']!=answer['raw_output']:raise ValueError('Raw return differs')
        if out['input_sha256']!=digest(job) or out['program_source']!=job['frames'][0] or any(meta[k]!=v for k,v in job['input_identity'].items()):raise ValueError('Actual input/source binding differs')
        if answer['parsed']!=parse_facts(answer['raw_output'],meta['truncated']) or call['answer']['parsed']!=answer['parsed']:raise ValueError('Format replay differs')
    synthetic=list((root/'calls').glob('synthetic_*/*.json'))
    if len(synthetic)!=len(assignment['cases']) or len(synthetic)>2 or any(read_json(p)['status']!='completed' for p in synthetic):raise ValueError('Synthetic budget differs')
    if len({read_json(p)['call_id'] for p in calls+synthetic})!=len(calls)+len(synthetic):raise ValueError('Duplicate generation IDs')
    for path,old in assets['sources'].items():
        if sha256(path)!=old:raise ValueError('Frozen source changed: '+path)
    artifacts={str(p.relative_to(root)):sha256(p) for folder in ('calls','raw_generations','results') for p in (root/folder).rglob('*.json')}
    previous=root/'completion_audit.json'
    if previous.exists() and read_json(previous)['artifacts']!=artifacts:raise ValueError('Recovery changed original generations')
    answers=[read_json(root/'results'/(jid+'.json'))['answer'] for jid in jobs]
    value=dict(status='passed',real_auxiliary_generations=48,synthetic_generations=len(synthetic),final_qa_generations=0,
        valid_json=sum(a['parsed']['valid'] for a in answers),strict_json=sum(a['parsed']['strict_valid'] for a in answers),
        truncated=sum(a['metadata']['truncated'] for a in answers),artifacts=artifacts,utc=utc())
    durable(previous,value);render(root);return value


def render(root):
    """接口：48图的新旧输出并排复核；人工评价文件另存，不由模型输出改写真值。"""
    root=Path(root);assets=read_json(root/'assets.json');esc=html.escape
    page=['<!doctype html><meta charset="utf-8"><title>单帧事实最小接口复核</title><style>body{max-width:1200px;margin:24px auto;font:16px/1.7 sans-serif}img{max-width:100%}pre{white-space:pre-wrap;background:#eee;padding:12px}section{border-top:1px solid #999;padding:12px}</style>',
          '<h1>48张单帧：事实内容与来源程序绑定</h1><p>旧编号错误单列；来源绑定不能证明陈述正确。最终QA新增0。</p>']
    for item in assets['jobs']:
        job=read_json(item['input_file']);f=job['frames'][0];old=read_json(job['parent_result'])['answer']['raw_output']
        target=root/'results'/(item['job_id']+'.json');new=read_json(target)['answer']['raw_output'] if target.exists() else '尚未生成'
        page+=['<section id="'+esc(item['job_id'])+'"><h2>'+esc(item['job_id'])+'</h2><p>'+esc(job['question'])+'</p>',
            '<p>源PTS='+str(f['source_pts'])+'；源号='+str(f['source_frame_index'])+'；实际秒='+str(f['timestamp_seconds'])+'</p>',
            '<a href="images/'+esc(item['job_id'])+'.png"><img src="images/'+esc(item['job_id'])+'.png"></a>',
            '<details><summary>旧单帧输出</summary><pre>'+esc(old)+'</pre></details><details open><summary>新接口原文</summary><pre>'+esc(new)+'</pre></details></section>']
    (root/'index.html').write_text('\n'.join(page))
