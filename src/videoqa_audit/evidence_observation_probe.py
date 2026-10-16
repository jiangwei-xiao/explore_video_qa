"""12题帧组事实观察诊断：先冻结图片/评价，再生成，最后人工语义复核。"""
from pathlib import Path
import copy
import hashlib
import html
import os
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from videoqa_runtime.common import ROOT, read_json, sha256, offline_environment
from videoqa_full.state import durable, digest, exclusive, invoke_once, utc
from videoqa_full.prepare import environment
from videoqa_budget32.frames import FrameProvider
from videoqa_audit.framework_contraction import RUN32
from videoqa_audit.evidence_observation_backend import (
    EvidenceObservationBackend, prepare_observation, parse_observations, INSTRUCTION,
)

PYTHON='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'
MODEL='/home/models/Qwen/Qwen2.5-VL-7B-Instruct'
REVIEW=ROOT/'outputs/analysis/dual_dataset_selection_review_20261015_r2'
FILES=('src/videoqa_audit/evidence_observation_backend.py',
       'src/videoqa_audit/evidence_observation_probe.py','scripts/probe_evidence_observations.py')
DEPENDENCIES=('src/videoqa_full/qwen_backend.py','src/videoqa_full/qwen_fa2_backend.py',
              'src/videoqa_budget32/frames.py','src/videoqa_methods/region_context.py')
SEEDS=(
    ('videomme','012-3',[246784]),
    ('videomme','035-2',[707584,720384,758784]),
    ('videomme','103-1',[114114,138138,174174]),
    ('videomme','113-3',[833333,878378]),
    ('videomme','383-2',[28283255,28328300]),
    ('videomme','386-1',[1462784,1468928]),
    ('videomme','538-1',[263263]),
    ('videomme','584-3',[7453184]),
    ('mlvu','mlvu-dev-Q8',[1936384,1942528]),
    ('mlvu','mlvu-dev-Q866',[6393856,6401536]),
    ('mlvu','mlvu-dev-Q624',[1401856,1878016,2215936,2753536]),
    ('mlvu','mlvu-dev-Q367',[80896]),
)


def publish(path, value):
    """接口：已经冻结的输入只允许一致恢复，不静默覆盖。"""
    path=Path(path)
    if path.exists():
        if read_json(path)!=value:raise ValueError('Frozen artifact differs: '+str(path))
    else:durable(path,value)


def choose_groups(pool, parent_selection, witness_pts):
    """接口：旧见证＋真实时间最近补齐，与父实际选择中原分数Top-4粗帧对照。"""
    cs=pool['candidates'];n=pool['initial_count'];scores=pool['initial_scores']
    key=lambda i:(int(cs[i]['source_pts']),int(cs[i]['source_frame_index']))
    lookup={key(i):i for i in range(len(cs))}
    h=[i for i,c in enumerate(cs) if int(c['source_pts']) in set(witness_pts)]
    if len(h)!=len(set(witness_pts)) or not 1<=len(h)<=4:
        raise ValueError('Exact frozen witness PTS absent or ambiguous')
    # 步骤1：所有距离在同一视频真实PTS刻度上比较，不用平均FPS或新打分。
    available=[i for i in range(len(cs)) if i not in h]
    available.sort(key=lambda i:(min(abs(key(i)[0]-key(j)[0]) for j in h),key(i),i))
    witness=sorted(h+available[:4-len(h)],key=key)
    chosen=[lookup[(f['source_pts'],f['source_frame_index'])]
            for f in parent_selection['selected_frames']]
    other=[i for i in chosen if i<n and i not in witness]
    other.sort(key=lambda i:(-scores[i],key(i),i))
    comparison=sorted(other[:4],key=key)
    if len(witness)!=4 or len(comparison)!=4 or set(witness)&set(comparison):
        raise ValueError('Expected two disjoint four-frame groups')
    return dict(witness=witness,comparison=comparison)


def load_tools():
    """接口：CPU准备只加载官方处理器和配置，不加载模型或运行前向。"""
    from transformers import AutoProcessor, AutoConfig
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,use_fast=False,
        min_pixels=100352,max_pixels=602112,
        size={'shortest_edge':100352,'longest_edge':602112})
    return processor,AutoConfig.from_pretrained(MODEL,local_files_only=True)


def prepare(root):
    """接口：冻结12题/24组/96源帧/120输入；本阶段视觉/语言前向均0。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);offline_environment()
    with exclusive(root/'prepare.lock'):
        if (root/'assets.json').exists():return verify_assets(root)
        began=time.perf_counter();sources={};stats={};groups=[];jobs=[]
        expected=read_json(REVIEW/'sources.json')
        exports=read_json(REVIEW/'summary.json')['exports']
        old_model=read_json(RUN32/'protocol.json')
        if environment(PYTHON)!=old_model['environment']:
            raise ValueError('Frozen Qwen environment changed')
        # 步骤1：核验旧清单、模型、代码与源文件；手工见证只用于建诊断输入。
        for name in FILES+DEPENDENCIES:
            sources[str(ROOT/name)]=sha256(ROOT/name)
            dest=root/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,dest)
        for path,info in old_model['model_files'].items():
            if sha256(path)!=info['sha256']:raise ValueError('Frozen model bytes differ')
            sources[path]=info['sha256'];st=Path(path).stat()
            stats[path]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
        for path in (RUN32/'protocol.json',REVIEW/'summary.json',REVIEW/'sources.json'):
            sources[str(path)]=sha256(path)
        processor,config=load_tools()
        # Q367等不在开发210；问题只能按身份从冻结全量清单读取，不能重新抽样。
        from videoqa_full.state import rows as vmrows
        from videoqa_mlvu.data import rows as mlrows
        from videoqa_audit.adjacent_context_development import ML
        rowbank={'videomme':{x['question_id']:x for x in vmrows()},
                 'mlvu':{x['question_id']:x for x in mlrows(ML)}}
        for ordinal,(dataset,q,witness_pts) in enumerate(SEEDS):
            record_path=REVIEW/'records'/dataset/'qwen32'/(q+'.json')
            if sha256(record_path)!=exports[str(record_path.relative_to(REVIEW))]:
                raise ValueError('Historical review record differs')
            record=read_json(record_path);poolpath=Path(record['pool_source'])
            parentpath=Path(record['answer_sources']['RD-1.2-exp12'])
            for path in (poolpath,parentpath):
                actual=sha256(path)
                if str(path) not in expected or actual!=expected[str(path)]:
                    raise ValueError('Frozen candidate/parent source differs: '+str(path))
                sources[str(path)]=actual
            sources[str(record_path)]=sha256(record_path)
            pool=read_json(poolpath);parent=read_json(parentpath)
            video=Path(pool['video']['path']);vstat=video.stat()
            if str(video) not in sources:
                actual=sha256(video)
                expected_video=expected.get(str(video))
                if expected_video is None:
                    expected_video=(old_model['video_hashes'][record['video_id']] if dataset=='videomme'
                        else read_json(ML/'manifest.json')['video_hashes'][str(video)])
                if actual!=expected_video:
                    raise ValueError('Frozen video identity not found or changed: '+str(video))
                sources[str(video)]=actual
                stats[str(video)]=dict(bytes=vstat.st_size,mtime_ns=vstat.st_mtime_ns)
            case_id=dataset+'__'+q;question=rowbank[dataset][q]['question']
            selected=choose_groups(pool,parent['selection'],witness_pts)
            for label,indices in selected.items():
                gid=case_id+'__'+label;frames=[copy.deepcopy(pool['candidates'][i]) for i in indices]
                images,rgb=FrameProvider(dict(video=pool['video'],selected_frames=frames)).decode()
                frame_entries=[]
                try:
                    contact=Image.new('RGB',(1440,920),'white');draw=ImageDraw.Draw(contact)
                    for local,(frame,image,raw_sha) in enumerate(zip(frames,images,rgb),1):
                        path=root/'images'/gid/(str(local)+'.png');path.parent.mkdir(parents=True,exist_ok=True)
                        image.save(path);sources[str(path)]=sha256(path)
                        entry=dict(frame_id=local,candidate_index=indices[local-1],source_pts=frame['source_pts'],
                            source_frame_index=frame['source_frame_index'],timestamp_seconds=frame['timestamp_seconds'],
                            timestamp_fraction=str((frame['source_pts']-pool['video']['start_pts'])*__import__('fractions').Fraction(pool['video']['time_base'])),
                            image=str(path),rgb_sha256=raw_sha)
                        frame_entries.append(entry)
                        show=image.copy();show.thumbnail((710,420));x=((local-1)%2)*720;y=((local-1)//2)*460
                        contact.paste(show,(x,y+30));show.close()
                        draw.text((x+8,y+7),f'Frame {local}: {frame["timestamp_seconds"]:.6f}s PTS={frame["source_pts"]}',fill='black')
                    out=root/'review'/gid/'contact.jpg';out.parent.mkdir(parents=True,exist_ok=True)
                    contact.save(out,quality=95);contact.close()
                    group=dict(group_id=gid,case_id=case_id,dataset=dataset,question_id=q,label=label,
                        question=question,frames=frame_entries,contact=str(out.relative_to(root)),
                        witness_pts=witness_pts,video_sha256=sources[str(video)],video=str(video))
                    groups.append(group)
                    items=[('group',[1,2,3,4])]+[('single_'+str(i),[i]) for i in range(1,5)]
                    if ordinal%2:items=items[1:]+items[:1]
                    for kind,ids in items:
                        picked=[images[i-1] for i in ids]
                        _,identity=prepare_observation(processor,picked,question,ids,
                            [frame_entries[i-1]['timestamp_seconds'] for i in ids],config)
                        job=dict(job_id=gid+'__'+kind,group_id=gid,case_id=case_id,kind=kind,
                                 question=question,frames=[frame_entries[i-1] for i in ids],input_identity=identity)
                        path=root/'inputs'/(job['job_id']+'.json');publish(path,job)
                        jobs.append(dict(job_id=job['job_id'],case_id=case_id,group_id=gid,kind=kind,
                                         input_file=str(path),input_sha256=sha256(path)))
                finally:
                    for image in images:image.close()
            durable(root/'prepare_status.json',dict(completed=ordinal+1,total=12,utc=utc()))
        if len(groups)!=24 or len(jobs)!=120 or sum(len(g['frames']) for g in groups)!=96:
            raise ValueError('Fixed diagnostic counts differ')
        if len({(g['video_sha256'],f['source_pts']) for g in groups for f in g['frames']})!=96:
            raise ValueError('Diagnostic source frames duplicated')
        assets=dict(task='evidence_observation',groups=groups,jobs=jobs,sources=sources,source_stats=stats,
            model_files=old_model['model_files'],environment=old_model['environment'],
            code={name:sha256(ROOT/name) for name in FILES+DEPENDENCIES},prompt=INSTRUCTION,
            auxiliary_budget=120,synthetic_budget=4,final_qa_budget=0,
            prepare_seconds=time.perf_counter()-began,utc=utc())
        publish(root/'assets.json',assets);render(root);return assets


def verify_assets(root):
    """接口：已冻结输入恢复核验；大文件用已绑定的stat，最终审计再检查完整来源。"""
    root=Path(root);assets=read_json(root/'assets.json')
    for name,old in assets['code'].items():
        if sha256(ROOT/name)!=old:raise ValueError('Observer code changed: '+name)
    for path,info in assets['source_stats'].items():
        st=Path(path).stat()
        if (st.st_size,st.st_mtime_ns)!=(info['bytes'],info['mtime_ns']):raise ValueError('Source stat changed')
    for job in assets['jobs']:
        if sha256(job['input_file'])!=job['input_sha256']:raise ValueError('Frozen observer input changed')
    return assets


def freeze_reference(root):
    """接口：在首个生成前冻结事前评价依据；评价内容不会传给模型。"""
    root=Path(root)
    if (root/'protocol.json').exists():return verify(root)
    assets=verify_assets(root);path=root/'evaluation_reference.json';ref=read_json(path)
    expected={g['case_id'] for g in assets['groups']}
    if set(ref['cases'])!=expected or ref.get('stage')!='before_model_generation':
        raise ValueError('Expected all 12 pre-model annotated cases')
    for case in ref['cases'].values():
        if not 1<=len(case['facts'])<=3 or len(case.get('relations',[]))>1:
            raise ValueError('At most three visible facts and one relation per case')
        for fact in case['facts']+case.get('relations',[]):
            for group in ('witness','comparison'):
                if any(type(i) is not int or i not in (1,2,3,4) for i in fact['frames'][group]):
                    raise ValueError('Invalid reference frame citation')
    if list((root/'calls').glob('*/*.json')):raise ValueError('Cannot refreeze after a model invocation')
    spec=dict(task='evidence_observation',assets_sha256=sha256(root/'assets.json'),
        reference_sha256=sha256(path),max_real_calls=120,max_synthetic_calls=4,
        final_qa_calls=0,code=assets['code'],utc=utc())
    publish(root/'protocol.json',spec);return spec


def verify(root):
    """接口：模型运行前核验提示、图片和评价版本；未知调用不得重发。"""
    root=Path(root);spec=read_json(root/'protocol.json');assets=verify_assets(root)
    if environment(PYTHON)!=assets['environment']:raise ValueError('Observer environment changed')
    if sha256(root/'assets.json')!=spec['assets_sha256'] or sha256(root/'evaluation_reference.json')!=spec['reference_sha256']:
        raise ValueError('Frozen assets or reference changed')
    return spec


def _read_images(job):
    """接口：读取冻结无损PNG并复算RGB；只返回模型真正允许看到的画面。"""
    images=[]
    for frame in job['frames']:
        im=Image.open(frame['image']).convert('RGB')
        if hashlib.sha256(np.asarray(im).tobytes()).hexdigest()!=frame['rgb_sha256']:
            im.close();raise ValueError('Source RGB changed')
        images.append(im)
    return images


def restore_raw(root,job_id,method):
    """接口：模型已完整返回时只恢复解析/结果；不确定生成立即暂停。"""
    root=Path(root);cp=root/'calls'/method/(job_id+'.json')
    if not cp.exists():return None
    call=read_json(cp)
    if call['protocol_sha256']!=sha256(root/'protocol.json'):raise ValueError('Call protocol changed')
    if call['status']=='completed':return call['answer']
    rawpath=root/'raw_generations'/method/(job_id+'.json')
    if not rawpath.exists():raise RuntimeError('Uncertain generation; no retry')
    raw=read_json(rawpath)
    if raw['generation_state']!={'started':True,'returned':True}:
        raise RuntimeError('Incomplete raw generation; no retry')
    if raw['identity']!=dict(job_id=job_id,method=method,protocol_sha256=call['protocol_sha256'],gpu=call['gpu']):
        raise ValueError('Raw call identity differs')
    meta=raw['metadata']
    answer=dict(raw_output=raw['raw_output'],metadata=meta,
        parsed=parse_observations(raw['raw_output'],meta['frame_ids'],meta['truncated']))
    call.update(status='completed',answer=answer,recovered_postprocessing=True,
                generation_state=raw['generation_state']);durable(cp,call)
    return answer


def process_job(root,job,backend,gpu,method='facts'):
    """接口：冻结输入→一次生成→原文检查点→解析；调用前验证像素和Token。"""
    root=Path(root);qid=job['job_id'];dest=root/'results'/(qid+'.json')
    with exclusive(root/'job_locks'/(method+'_'+qid+'.lock')):
        answer=restore_raw(root,qid,method)
        if dest.exists() and method=='facts':return read_json(dest)
        if answer is None:
            if (root/'PAUSE.json').exists():raise RuntimeError('Observer pipeline paused')
            images=_read_images(job)
            try:
                start=time.perf_counter();inputs,identity=backend.prepare(images,job['question'],
                    [f['frame_id'] for f in job['frames']],[f['timestamp_seconds'] for f in job['frames']])
                if identity!=job['input_identity']:raise ValueError('Actual observer input differs')
                preprocessing=time.perf_counter()-start
                fp=sha256(root/'protocol.json');raw=root/'raw_generations'/method/(qid+'.json')
                call_identity=dict(job_id=qid,method=method,protocol_sha256=fp,gpu=gpu)
                def invoke(state):
                    return backend.generate_prepared(inputs,identity,state,raw,call_identity)
                answer,_,checkpoint=invoke_once(root,qid,method,fp,gpu,invoke)
                answer=dict(answer,preprocessing_seconds=preprocessing,checkpoint_seconds=checkpoint)
            finally:
                for im in images:im.close()
        value=dict(job_id=qid,group_id=job.get('group_id'),kind=job.get('kind'),gpu=gpu,
                   input_sha256=digest(job),answer=answer,utc=utc())
        if method=='facts':durable(dest,value)
        return value


def worker(root,gpu):
    """接口：一张固定物理卡，一个长期观察进程；真实120次与合成分别记账。"""
    root=Path(root);verify(root);assets=read_json(root/'assets.json');assignment=read_json(root/'assignment.json')
    with exclusive(root/'worker_locks'/(gpu+'.lock')):
        session=root/'workers'/(gpu+'.json')
        assigned=[j for j in assets['jobs'] if j['case_id'] in assignment['cases'][gpu]]
        if session.exists():
            for item in assigned:
                answer=restore_raw(root,item['job_id'],'facts')
                if answer is not None and not (root/'results'/(item['job_id']+'.json')).exists():
                    job=read_json(item['input_file'])
                    durable(root/'results'/(item['job_id']+'.json'),dict(job_id=item['job_id'],group_id=job['group_id'],
                        kind=job['kind'],gpu=gpu,input_sha256=digest(job),answer=answer,recovered=True))
            if all((root/'results'/(j['job_id']+'.json')).exists() for j in assigned):return
            raise RuntimeError('Previous worker interrupted; no automatic process/synthetic retry')
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(2027)
        durable(session,dict(pid=os.getpid(),sid=os.getsid(0),gpu=gpu,status='loading',utc=utc()))
        backend=EvidenceObservationBackend();durable(root/'load_reports'/(gpu+'.json'),backend.load_report)
        # 步骤1：每个实际进程仅两次合成，分别覆盖1帧与4帧，不按输出调提示词。
        for n,size in ((1,(640,360)),(4,(360,640))):
            ims=[Image.new('RGB',size,(i*50,100,180)) for i in range(n)]
            entries=[]
            for i,im in enumerate(ims,1):
                path=root/'synthetic'/gpu/(str(n)+'_'+str(i)+'.png');path.parent.mkdir(parents=True,exist_ok=True)
                im.save(path);entries.append(dict(frame_id=i,image=str(path),timestamp_seconds=float(i-1),
                    rgb_sha256=hashlib.sha256(np.asarray(im).tobytes()).hexdigest()))
            _,identity=backend.prepare(ims,'Describe the visible colored surfaces.',list(range(1,n+1)),list(map(float,range(n))))
            for im in ims:im.close()
            synthetic=dict(job_id='warmup_'+str(n),kind='synthetic',question='Describe the visible colored surfaces.',
                           frames=entries,input_identity=identity)
            process_job(root,synthetic,backend,gpu,'synthetic_'+gpu)
        # 步骤2：同题单帧/帧组保持同卡同进程；按冻结序列交替先后。
        for item in assigned:
            if (root/'PAUSE.json').exists():return
            durable(session,dict(pid=os.getpid(),gpu=gpu,status='running',job_id=item['job_id'],utc=utc()))
            process_job(root,read_json(item['input_file']),backend,gpu)
        durable(session,dict(pid=os.getpid(),gpu=gpu,status='completed',real_jobs=len(assigned),utc=utc()))


def run(root):
    """接口：最多两张实际空闲卡，独立后台完成生成和技术审计；不启动正式QA。"""
    root=Path(root);verify(root);assets=read_json(root/'assets.json')
    with exclusive(root/'coordinator.lock'):
        if all((root/'results'/(x['job_id']+'.json')).exists() for x in assets['jobs']):
            return technical_report(root)
        if (root/'PAUSE.json').exists():raise RuntimeError('Paused observer run requires state inspection')
        assignment=root/'assignment.json'
        if assignment.exists():
            gpus=list(read_json(assignment)['cases'])
        else:
            from videoqa_runtime.baseline_run import available_gpus
            while True:
                try:free,_=available_gpus(list(map(str,range(8))))
                except RuntimeError as error:
                    if str(error)!='No requested GPU is currently idle':raise
                    durable(root/'heartbeat.json',dict(status='waiting_for_idle_gpu',pid=os.getpid(),utc=utc()));time.sleep(5);continue
                gpus=free[-2:];break
            ids=list(dict.fromkeys(g['case_id'] for g in assets['groups']))
            publish(assignment,dict(cases={g:ids[i::len(gpus)] for i,g in enumerate(gpus)},
                uuids={g:subprocess.check_output(['nvidia-smi','-i',g,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip() for g in gpus}))
        children=[]
        try:
            for gpu in gpus:
                log=(root/('worker_'+gpu+'.log')).open('a',buffering=1)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
                p=subprocess.Popen([PYTHON,str(ROOT/'scripts/probe_evidence_observations.py'),
                    '--run-id',root.name,'--stage','worker','--gpu',gpu],cwd=ROOT,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                children.append((gpu,p,log))
            while any(p.poll() is None for _,p,_ in children):
                if any(p.poll() not in (None,0) for _,p,_ in children):raise RuntimeError('Observer worker failed; no retry')
                durable(root/'heartbeat.json',dict(status='generating',pid=os.getpid(),sid=os.getsid(0),
                    workers={g:p.pid for g,p,_ in children},completed=len(list((root/'results').glob('*.json'))),
                    total=120,utc=utc()));time.sleep(3)
            if any(p.returncode for _,p,_ in children):raise RuntimeError('Observer worker exited abnormally')
            return technical_report(root)
        except BaseException as error:
            durable(root/'PAUSE.json',dict(error=repr(error),traceback=traceback.format_exc(),utc=utc()));raise
        finally:
            for _,p,log in children:
                if p.poll() is None:p.wait()
                log.close()


def audit(root):
    """接口：120结果/物理调用与≤4合成闭合，完成恢复不改变原生成文件。"""
    root=Path(root);verify(root);assets=read_json(root/'assets.json')
    expected={j['job_id']:j for j in assets['jobs']}
    if {p.stem for p in (root/'results').glob('*.json')}!=set(expected):raise ValueError('Observer results incomplete')
    calls=list((root/'calls'/'facts').glob('*.json'))
    if {p.stem for p in calls}!=set(expected):raise ValueError('Observer real-call count/IDs differ')
    assignment=read_json(root/'assignment.json')
    owners={case:gpu for gpu,cases in assignment['cases'].items() for case in cases}
    if len(owners)!=12:raise ValueError('Question/GPU assignment differs')
    for job_id,item in expected.items():
        job=read_json(item['input_file']);result=read_json(root/'results'/(job_id+'.json'))
        answer=result['answer'];call=read_json(root/'calls/facts'/(job_id+'.json'));raw=read_json(root/'raw_generations/facts'/(job_id+'.json'))
        if result['input_sha256']!=digest(job) or call['status']!='completed' or raw['generation_state']!={'started':True,'returned':True}:
            raise ValueError('Observer physical generation identity differs')
        if raw['raw_output']!=answer['raw_output'] or raw['metadata']!=answer['metadata']:
            raise ValueError('Raw generation/result metadata differs')
        if raw['identity']!=dict(job_id=job_id,method='facts',protocol_sha256=sha256(root/'protocol.json'),gpu=owners[job['case_id']]):
            raise ValueError('Raw generation GPU/protocol identity differs')
        if call['answer']['raw_output']!=answer['raw_output'] or call['answer']['parsed']!=answer['parsed']:
            raise ValueError('Call/result raw or parsed observation differs')
        if any(answer['metadata'][k]!=v for k,v in job['input_identity'].items()):
            raise ValueError('Actual observed pixel/token identity differs')
        if answer['parsed']!=parse_observations(answer['raw_output'],job['input_identity']['frame_ids'],answer['metadata']['truncated']):
            raise ValueError('JSON parse replay differs')
    synthetic=list((root/'calls').glob('synthetic_*/*.json'))
    if len(synthetic)!=2*len(assignment['cases']) or len(synthetic)>4 or any(read_json(p)['status']!='completed' for p in synthetic):
        raise ValueError('Synthetic budget/state differs')
    for path,expected_hash in assets['sources'].items():
        if sha256(path)!=expected_hash:raise ValueError('Frozen source hash changed: '+path)
    files={str(p.relative_to(root)):sha256(p) for folder in ('calls','raw_generations','results') for p in (root/folder).rglob('*.json')}
    previous=root/'completion_audit.json'
    if previous.exists() and read_json(previous)['artifacts']!=files:raise ValueError('Completed restore changed generation artifacts')
    ids=[read_json(p)['call_id'] for p in calls+synthetic]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate physical call ID')
    value=dict(status='passed',real_auxiliary_generations=120,synthetic_generations=len(synthetic),
        final_qa_generations=0,completed_restore_new_generations=0,artifacts=files,utc=utc())
    durable(previous,value);return value


def technical_report(root):
    """接口：自动报告仅格式、来源与成本；事实真伪留给有界语义复核。"""
    root=Path(root);result=audit(root);assets=read_json(root/'assets.json')
    kinds={}
    for kind in ('group','single'):
        selected=[x for x in assets['jobs'] if (x['kind']=='group')==(kind=='group')]
        answers=[read_json(root/'results'/(x['job_id']+'.json'))['answer'] for x in selected]
        kinds[kind]=dict(n=len(answers),valid_json=sum(x['parsed']['valid'] for x in answers),
            truncated=sum(x['metadata']['truncated'] for x in answers),
            output_tokens=sum(len(x['metadata']['generated_token_ids']) for x in answers),
            generation_seconds=sum(x['metadata']['generation_seconds'] for x in answers))
    summary=dict(status='generated_semantic_review_pending',technical=result['status'],counts=kinds,
                 real_auxiliary_generations=120,synthetic_generations=result['synthetic_generations'],
                 final_qa_generations=0,utc=utc())
    durable(root/'technical_summary.json',summary);render(root);return summary


def render(root):
    """接口：统一展示冻结帧组、单帧输出与联合输出；不从生成结果改写评价依据。"""
    root=Path(root);assets=read_json(root/'assets.json')
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>帧组事实观察诊断</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1450px;margin:25px auto}img{max-width:100%}pre{white-space:pre-wrap;background:#f5f5f5;padding:10px}table{width:100%}td{vertical-align:top;width:50%}</style>',
        '<h1>帧组事实提取：固定12题、24组</h1><p>图片为原源RGB；只评价观察能力，不是选帧算法成绩。模型输入无选项、标准答案或人工评价标签。</p>']
    for group in assets['groups']:
        page+=['<h2>'+html.escape(group['group_id'])+'</h2><p>'+html.escape(group['question'])+'</p>',
               '<a href="../'+group['contact']+'"><img src="../'+group['contact']+'"></a><p>']
        for frame in group['frames']:
            page+=['<a href="../'+str(Path(frame['image']).relative_to(root))+'">原帧'+str(frame['frame_id'])+'</a> ']
        page+=['</p>']
        relevant=[j for j in assets['jobs'] if j['group_id']==group['group_id']]
        for item in relevant:
            out=root/'results'/(item['job_id']+'.json')
            if not out.exists():continue
            answer=read_json(out)['answer']
            page+=['<details><summary>'+html.escape(item['kind'])+'：'+('JSON有效' if answer['parsed']['valid'] else '格式失败')+'</summary>',
                   '<pre>'+html.escape(answer['raw_output'])+'</pre></details>']
    path=root/'review/index.html';path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(page)+'</html>')
