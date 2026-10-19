"""长视频先导与一次训练的独立调度：先导门槛、标签隔离和严格一次生成。"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import copy
import hashlib
import html
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import time
import traceback

import numpy as np
from PIL import Image

from videoqa_runtime.common import ROOT,read_json,sha256,offline_environment
from videoqa_full.state import durable,digest,exclusive,invoke_once,utc
from videoqa_full.prepare import environment
from videoqa_lmms_bridge.common import framework_identity
from videoqa_set_value.data import METHODS
from videoqa_set_value.data_acquisition import stable_key
from videoqa_set_value.runtime import PREP_PYTHON,QA_PYTHON,MODEL,load_images
from videoqa_set_value import longvideo_task as task
from videoqa_set_value.longvideo_trial import OLD_HEAD,OLD_HEAD_SHA,SOURCE_SCOPE

_TOOLS=None
ENTRY=ROOT/'scripts/run_longvideo_set_value_trial.py'
CONDITIONS=(*METHODS,'G','W')
PHASE_CAPS=dict(pilot=1200,train_rest=2400,val=1600)


def rows(root,phase=None):
    """接口：仅清单公开行；任何训练阶段都不能获取验证反馈。"""
    data=read_json(Path(root)/'manifest.json')['rows']
    if phase=='pilot':return [r for r in data if r['pilot']]
    if phase=='train_rest':return [r for r in data if r['split']=='train' and not r['pilot']]
    if phase in ('train','val'):return [r for r in data if r['split']==phase]
    return data


def scoring_row(root,row):
    """接口：计分专用标签仅在生成调用的后处理侧绑定，公开prompt投影永不读取该字段。"""
    manifest=read_json(Path(root)/'manifest.json')
    labels_path=next(Path(p) for p in manifest['identity_sources'] if p.endswith('/scoring_labels.json'))
    if sha256(labels_path)!=manifest['identity_sources'][str(labels_path)]:raise ValueError('Scoring labels mutated')
    record=next(r for r in read_json(labels_path)['rows'] if r['question_id']==row['question_id'])
    match=re.fullmatch(r'\s*<answer>\s*([A-D])\s*</answer>\s*',record['reference_answer'])
    if not match:raise ValueError('Ambiguous official reference answer; do not modify split')
    return dict(row,answer_letter=match.group(1))


def freeze_sources(root):
    """接口：下载等待前即冻结算法/适配源码，不能在数天后把新实现悄悄当作本轮固定候选。"""
    root=Path(root);dest=root/'source_freeze.json'
    if dest.exists():
        record=read_json(dest)
        for relative,expected in record['code'].items():
            if sha256(ROOT/relative)!=expected:raise ValueError('Pre-download frozen source changed: '+relative)
        if framework_identity()!=record['framework'] or sha256(OLD_HEAD)!=record['old_head_sha256']:
            raise ValueError('Pre-download framework/old-head identity changed')
        return record
    paths=sorted(set([*ROOT.glob('src/videoqa_set_value/*.py'),*ROOT.glob('src/videoqa_methods/*.py'),
        *ROOT.glob('src/videoqa_runtime/*.py'),ENTRY,ROOT/'src/videoqa_audit/set_value_trainability.py',
        ROOT/'src/videoqa_mlvu/data.py',ROOT/'src/videoqa_full/state.py',ROOT/'src/videoqa_full/prepare.py',
        ROOT/'src/videoqa_lmms_bridge/common.py',ROOT/'src/videoqa_lmms_bridge/llava_adapter.py',
        ROOT/'src/videoqa_generalization/llava_observe.py',ROOT/'src/videoqa_budget32/frames.py',
        ROOT/'configs/retrieval_density_v1.json',ROOT/'configs/method_versions.json',ROOT/'configs/local_model_inventory.json',
        ROOT/'configs/llava_source_manifest.json']))
    code={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    for name in code:
        path=root/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
    if sha256(OLD_HEAD)!=OLD_HEAD_SHA:raise ValueError('Old NExT-QA head changed')
    record=dict(code=code,framework=framework_identity(),old_head_sha256=OLD_HEAD_SHA,
        stage='pre_download_source_freeze',new_model_calls=0,utc=utc())
    durable(dest,record);return record


def preflight(root):
    """接口：教师前冻结代码/依赖/模型/数据身份；恢复不覆盖已登记协议。"""
    root=Path(root)
    if (root/'protocol.json').exists():return verify(root)
    frozen_sources=freeze_sources(root);manifest=read_json(root/'manifest.json');data=manifest['rows']
    if len(data)!=1200 or len(rows(root,'train'))!=800 or len(rows(root,'val'))!=400 or len(rows(root,'pilot'))!=200:
        raise ValueError('Fixed 800/400 and pilot200 denominator differs')
    if len({r['content_group'] for r in data})!=1200:raise ValueError('Content groups overlap')
    if sha256(OLD_HEAD)!=OLD_HEAD_SHA:raise ValueError('Frozen NExT-QA180 head changed')
    fingerprints=frozen_sources['code']
    for name in fingerprints:
        dest=root/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    models={};model_roots=set()
    for descriptor in read_json(ROOT/'configs/local_model_inventory.json')['models'].values():
        model_roots.add(descriptor['path'])
        for name,info in descriptor['files'].items():
            p=Path(descriptor['path'])/name
            if sha256(p)!=info['sha256']:raise ValueError('Frozen model bytes changed')
            models[str(p)]=dict(sha256=info['sha256'],bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
    if not {MODEL,'/home/models/Salesforce/blip-itm-base-coco','/home/models/google/siglip-so400m-patch14-384'}<=model_roots:
        raise ValueError('Missing teacher/vision/BLIP inventories')
    value=dict(task='longvideo_set_value_diagnostic',manifest_sha256=sha256(root/'manifest.json'),code=fingerprints,
        source_freeze_sha256=sha256(root/'source_freeze.json'),
        models=models,video_stats={r['video_path']:dict(sha256=r['video_sha256'],bytes=Path(r['video_path']).stat().st_size,
            mtime_ns=Path(r['video_path']).stat().st_mtime_ns) for r in data},framework=framework_identity(),
        environments=dict(prepare=environment(PREP_PYTHON),llava=environment(QA_PYTHON)),
        maximum_real_generations=5200,maximum_synthetic_generations=8,phase_caps=PHASE_CAPS,
        teacher=dict(model=MODEL,dtype='BF16',attention='SDPA',frames=8,visual_tokens=1680,max_new_tokens=16,seed=2027),
        old_head_sha256=OLD_HEAD_SHA,validation_predictions_before_qa=True,utc=utc())
    durable(root/'protocol.json',value);return value


def verify(root,full=False):
    """接口：代码/清单/模型/数据漂移立即暂停，不自动重建选择、重评分或重复问答。"""
    root=Path(root);p=read_json(root/'protocol.json')
    if sha256(root/'source_freeze.json')!=p['source_freeze_sha256']:raise ValueError('Pre-download source receipt changed')
    freeze_sources(root)
    if sha256(root/'manifest.json')!=p['manifest_sha256'] or framework_identity()!=p['framework']:
        raise ValueError('Manifest/framework changed')
    for name,expected in p['code'].items():
        if sha256(ROOT/name)!=expected:raise ValueError('Frozen code changed: '+name)
    for name,record in {**p['models'],**p['video_stats']}.items():
        st=Path(name).stat()
        if (st.st_size,st.st_mtime_ns)!=(record['bytes'],record['mtime_ns']):raise ValueError('Source/model stat changed')
        if full and sha256(name)!=record['sha256']:raise ValueError('Source/model SHA changed')
    if sha256(OLD_HEAD)!=OLD_HEAD_SHA:raise ValueError('Old head changed')
    return p


def donor_for(row,data):
    """接口：供体只能来自训练800内容组，以时长差/固定哈希选择，不读取任何标签。"""
    eligible=[r for r in data if r['split']=='train' and r['content_group']!=row['content_group']]
    if not eligible:raise ValueError('No distinct training donor')
    return min(eligible,key=lambda r:(abs(r['duration_seconds']-row['duration_seconds']),
        stable_key(SOURCE_SCOPE+'/donor',r['video_id'])))


def diagnostic_record(root,row,condition):
    """接口：G为人工8槽位；W只读取训练供体原片的native均匀帧，不运行供体BLIP。"""
    from videoqa_budget32.frames import native_frames
    from videoqa_set_value.media import SequentialFrameProvider
    root=Path(root);q=row['question_id'];cp=root/'diagnostic_inputs'/condition/(q+'.json')
    if cp.exists():return read_json(cp)
    if condition=='W':
        donor=donor_for(row,rows(root));index=read_json(donor['index_path']);video=index['video']
        frames=native_frames(np.load(index['pts_path'],allow_pickle=False),video,8)
        images,rgb=SequentialFrameProvider(dict(video=video,selected_frames=frames)).decode()
        extra=dict(donor_video_id=donor['video_id'],donor_content_group=donor['content_group'],donor_split='train')
    elif condition=='G':
        images=[Image.new('RGB',(384,384),(128,128,128)) for _ in range(8)]
        rgb=[hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in images]
        video=dict(path='artificial-gray-'+q+'.mp4',duration_seconds=8.)
        frames=[dict(source_pts=i,source_frame_index=i,timestamp_seconds=float(i)) for i in range(8)]
        extra=dict(artificial=True)
    else:raise ValueError('Unknown diagnostic condition')
    records=[]
    try:
        for i,(frame,image,h) in enumerate(zip(frames,images,rgb)):
            path=root/'diagnostic_rgb'/condition/q/(str(i)+'.png');path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
            records.append(dict(frame,image=str(path),rgb_sha256=h))
    finally:
        for image in images:image.close()
    result=dict(question_id=q,method=condition,selection=dict(video=video,selected_frames=frames),frames=records,**extra)
    durable(cp,result);return result


def init_cpu():
    """接口：spawn CPU进程仅加载Tokenizer与官方预处理；Torch每进程两线程。"""
    global _TOOLS
    import torch
    offline_environment();torch.set_num_threads(2);torch.set_num_interop_threads(1);_TOOLS=task.cpu_tools()


def freeze_one(job):
    """接口：只在同题内按源身份/RGB/官方像素/完整提示Token建立共享组，不读取答案。"""
    root,row,phase=Path(job[0]),job[1],job[2];q=row['question_id'];dest=root/'inputs'/phase/(q+'.json')
    if dest.exists():return read_json(dest)
    entries={};groups=[];conditions=CONDITIONS if phase=='pilot' else METHODS
    for method in conditions:
        path=root/'selections'/method/(q+'.json') if method in METHODS else root/'diagnostic_inputs'/method/(q+'.json')
        record=read_json(path) if method in METHODS else diagnostic_record(root,row,method)
        ims=load_images(record)
        try:identity=task.cpu_identity(_TOOLS,ims,row)
        finally:
            for im in ims:im.close()
        source=[['artificial_gray',i] for i in range(8)] if record.get('artificial') else [[f['source_pts'],f['source_frame_index']] for f in record['frames']]
        if len(source)!=8 or (not record.get('artificial') and (source!=sorted(source) or len({f[0] for f in source})!=8)):
            raise ValueError('Expected eight ordered unique source frames')
        key=digest(dict(source=source,rgb=[f['rgb_sha256'] for f in record['frames']],input=identity))
        entries[method]=dict(path=str(path),sha256=sha256(path),identity=identity,input_key=key,source_order=source)
        group=next((g for g in groups if g['input_key']==key),None)
        if group is None:groups.append(dict(input_key=key,representative=method,methods=[method]))
        else:group['methods'].append(method)
    value=dict(question_id=q,phase=phase,methods=entries,groups=groups,partition_uses_labels=False)
    durable(dest,value);return value


def freeze_inputs(root,phase):
    """接口：所有阶段实际额度在首个真实生成前冻结；验证冻结不读取验证计分文件。"""
    root=Path(root);dest=root/'input_plans'/(phase+'.json');verify(root)
    if dest.exists():return read_json(dest)
    selected=rows(root,phase);exports={};calls=0
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=init_cpu) as pool:
        for value in pool.map(freeze_one,[(str(root),r,phase) for r in selected]):
            exports[value['question_id']]=sha256(root/'inputs'/phase/(value['question_id']+'.json'));calls+=len(value['groups'])
            durable(root/'freeze_progress.json',dict(phase=phase,completed=len(exports),total=len(selected),utc=utc()))
    if calls>PHASE_CAPS[phase] or len(exports)!=len(selected):raise ValueError('Phase call budget/denominator mismatch')
    value=dict(phase=phase,exports=exports,max_real_calls=calls,hard_cap=PHASE_CAPS[phase],
        method_results=len(selected)*(6 if phase=='pilot' else 4),utc=utc())
    durable(dest,value);return value


def phase_protocol(root,phase):
    """接口：教师调用ID与输入计划绑定；val协议额外绑定事前预测和训练检查点。"""
    root=Path(root);stage=root/'qa'/phase;path=stage/'protocol.json'
    value=dict(parent_sha256=sha256(root/'protocol.json'),input_plan_sha256=sha256(root/'input_plans'/(phase+'.json')),
        max_real_calls=read_json(root/'input_plans'/(phase+'.json'))['max_real_calls'],
        predictions_sha256=sha256(root/'validation_predictions.json') if phase=='val' else None)
    if phase=='val' and read_json(root/'validation_predictions.json')['checkpoint_sha256']!=sha256(root/'head/state.pt'):
        raise ValueError('Validation prediction head changed')
    if path.exists():
        if read_json(path)!=value:raise ValueError('QA phase fingerprint differs')
    else:durable(path,value)
    return stage


def restore_raw(stage,row,method,gpu):
    """接口：已返回只恢复任务后处理；未确认生成、换卡或协议变化均禁止重发。"""
    stage=Path(stage);q=row['question_id'];cp=stage/'calls'/method/(q+'.json')
    if not cp.exists():return None
    call=read_json(cp);identity=dict(question_id=q,method=method,gpu=gpu,protocol_sha256=sha256(stage/'protocol.json'))
    if call['protocol_sha256']!=identity['protocol_sha256'] or call['gpu']!=gpu:raise ValueError('Call identity/GPU changed')
    if call['status']=='completed':return call['answer']
    path=stage/'raw_generations'/method/(q+'.json')
    if not path.exists():raise RuntimeError('Uncertain generation; repetition forbidden')
    raw=read_json(path)
    if raw['identity']!=identity:raise ValueError('Raw return identity differs')
    if raw['generation_state']!={'started':True,'returned':True}:raise RuntimeError('Unconfirmed raw generation cannot be restored')
    answer=task.answer_from_raw(raw,row)
    durable(cp,dict(call,status='completed',answer=answer,generation_state=raw['generation_state'],recovered_postprocessing=True))
    return answer


def process_question(root,public,phase,backend,adapter,gpu):
    """接口：先一次物理生成并存原文，再严格计分并发布所有同输入引用结果。"""
    root=Path(root);q=public['question_id'];plan_path=root/'inputs'/phase/(q+'.json')
    input_plan=root/'input_plans'/(phase+'.json');frozen=read_json(input_plan)
    if sha256(input_plan)!=read_json(root/'qa'/phase/'protocol.json')['input_plan_sha256']:
        raise ValueError('Phase input plan fingerprint changed')
    if sha256(plan_path)!=frozen['exports'][q]:raise ValueError('Frozen actual input changed')
    plan=read_json(plan_path);stage=root/'qa'/phase;groups=plan['groups'];shift=public['ordinal']%len(groups)
    row=dict(public,_scoring_root=str(root))
    # 步骤1：同题固定同进程；候选次序循环轮换，G/W只出现在先导。
    for group in groups[shift:]+groups[:shift]:
        method=group['representative'];entry=plan['methods'][method]
        if all((stage/'results'/m/(q+'.json')).exists() for m in group['methods']):continue
        answer=restore_raw(stage,row,method,gpu)
        if answer is None:
            if (root/'PAUSE.json').exists():return
            if len([p for p in (stage/'calls').glob('*/*.json') if p.parent.name!='synthetic'])>=frozen['max_real_calls']:
                raise ValueError('Frozen real/synthetic call limit exhausted')
            if sha256(entry['path'])!=entry['sha256']:raise ValueError('Selection changed')
            record=read_json(entry['path']);ims=load_images(record)
            identity=dict(question_id=q,method=method,gpu=gpu,protocol_sha256=sha256(stage/'protocol.json'))
            try:
                answer,_,_=invoke_once(stage,q,method,identity['protocol_sha256'],gpu,lambda state:
                    task.observe(backend,adapter,ims,record['selection'],row,state,stage/'raw_generations'/method/(q+'.json'),identity))
            finally:
                for im in ims:im.close()
        expected=entry['identity'];observed=answer['observed']
        if any(expected[k]!=observed[k] for k in ('prompt','input_token_ids','pixel_sha256','source_frames','visual_tokens')):
            raise ValueError('Actual model input differs from frozen CPU identity')
        if observed['visual_tokens']!=1680 or observed['max_new_tokens']!=16:raise ValueError('Teacher budget changed')
        # 步骤2：原文已持久化后，才将参考答案用于正确性标签；引用不重复投票。
        score=task.metric(row,answer['raw_output'])
        for m in group['methods']:
            value=dict(question_id=q,method=m,phase=phase,correct=score['correct'],answer=answer,input_key=entry['input_key'],
                call_path=str(stage/'calls'/method/(q+'.json')),shared=m!=method,gpu=gpu)
            dest=stage/'results'/m/(q+'.json')
            if dest.exists() and read_json(dest)!=value:raise ValueError('Published result changed')
            if not dest.exists():durable(dest,value)


def idle_gpus(root,phase):
    """接口：仅GPU0–3，阶段启动时冻结实际空闲卡；无卡后台等待而非抢占。"""
    from videoqa_runtime.baseline_run import available_gpus
    while True:
        try:free,_=available_gpus(['0','1','2','3']);return free
        except RuntimeError as error:
            if str(error)!='No requested GPU is currently idle':raise
            durable(Path(root)/'heartbeat.json',dict(stage=phase,status='waiting_for_gpu0_3',utc=utc()));time.sleep(5)


def assign(root,phase,gpus):
    """接口：按视频时长降序做固定贪心分卡，恢复保持原物理卡；不由对错调度。"""
    root=Path(root);path=root/'assignments'/(phase+'.json')
    if path.exists():return read_json(path)
    selection=phase.startswith('select_');part=phase.split('_',1)[1] if selection else phase
    selected=rows(root,part)
    if part=='main':selected=rows(root,'train_rest')+rows(root,'val')
    load=dict.fromkeys(gpus,0.);allocation={g:[] for g in gpus}
    for row in sorted(selected,key=lambda r:(-r['duration_seconds'],r['ordinal'])):
        gpu=min(gpus,key=lambda g:(load[g],g));allocation[gpu].append(row['question_id']);load[gpu]+=row['duration_seconds']
    value=dict(questions=allocation,gpu_uuids={g:subprocess.check_output(['nvidia-smi','-i',g,'--query-gpu=uuid',
        '--format=csv,noheader'],text=True).strip() for g in gpus},utc=utc());durable(path,value);return value


def worker(root,phase,gpu):
    """接口：一张物理卡一个长期进程；先导后释放教师，主进程等待训练/验证预测发布。"""
    import torch
    root=Path(root);verify(root);offline_environment();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.manual_seed(2027);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ids=set(read_json(root/'assignments'/(phase+'.json'))['questions'][gpu]);selected=[r for r in rows(root) if r['question_id'] in ids]
    allocation=read_json(root/'assignments'/(phase+'.json'))
    actual_uuid=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if actual_uuid!=allocation['gpu_uuids'][gpu]:raise ValueError('Assigned physical GPU UUID changed')
    selected.sort(key=lambda r:(-r['duration_seconds'],r['ordinal']))
    session=root/'workers'/(phase+'_'+gpu+'.json')
    if session.exists():raise RuntimeError('Existing worker session requires state audit, not automatic restart')
    durable(session,dict(status='loading',pid=os.getpid(),gpu=gpu,utc=utc()))
    if phase.startswith('select_'):
        from videoqa_set_value.selection import prepare_one
        start=time.monotonic();scorer=make_scorer(root)
        durable(root/'load_reports'/(phase+'_'+gpu+'.json'),dict(model='BLIP-1',load_seconds=time.monotonic()-start,
            synthetic_forwards=0,gpu=gpu,utc=utc()))
        for row in selected:
            if (root/'PAUSE.json').exists():return
            durable(session,dict(status='selecting',question_id=row['question_id'],pid=os.getpid(),gpu=gpu,utc=utc()))
            scorer.question_id=row['question_id'];scorer.usage_ordinal=0
            prepare_one(root,row,scorer)
    else:
        from videoqa_runtime.llava_backend import LlavaBackend
        started=time.monotonic();backend=LlavaBackend();adapter=task.make_adapter(backend)
        durable(root/'load_reports'/(phase+'_'+gpu+'.json'),dict(backend.load_report,wall_seconds=time.monotonic()-started))
        first='pilot' if phase=='pilot' else 'train_rest';stage=root/'qa'/first
        # 步骤1：每个实际教师进程一次合成，先导≤4、主进程≤4，总计≤8。
        synthetic=dict(question_id='synthetic-'+phase+'-'+gpu,question='Which synthetic color is visible?',
            options=['Gray','Red','Blue','Green'],option_letters=list('ABCD'),answer_letter='A')
        ims=[Image.new('RGB',(384,384),(128,128,128)) for _ in range(8)]
        selection=dict(video=dict(path='synthetic-longvideo.mp4',duration_seconds=8.),selected_frames=[dict(timestamp_seconds=float(i)) for i in range(8)])
        q=synthetic['question_id'];identity=dict(question_id=q,method='synthetic',gpu=gpu,protocol_sha256=sha256(stage/'protocol.json'))
        try:invoke_once(stage,q,'synthetic',identity['protocol_sha256'],gpu,lambda state:
            task.observe(backend,adapter,ims,selection,synthetic,state,stage/'raw_generations/synthetic'/(q+'.json'),identity))
        finally:
            for im in ims:im.close()
        for part in (('pilot',) if phase=='pilot' else ('train_rest','val')):
            # 步骤2：验证教师在冻结预测门之后才能运行；CPU训练完成前不读取验证标签。
            if part=='val':
                while not (root/'release_validation.json').exists():
                    if (root/'PAUSE.json').exists() or (root/'STOP.json').exists():return
                    durable(session,dict(status='waiting_for_validation_predictions',pid=os.getpid(),gpu=gpu,utc=utc()));time.sleep(3)
                release=read_json(root/'release_validation.json')
                if release['predictions_sha256']!=sha256(root/'validation_predictions.json'):raise ValueError('Validation release identity changed')
            allowed={r['question_id'] for r in rows(root,part)}
            pending=[r for r in selected if r['question_id'] in allowed]
            # 前5题按冻结清单序先行，其后恢复时长调度；只按工程正确性继续。
            first_ids={r['question_id'] for r in rows(root,part)[:5]}
            pending.sort(key=lambda r:(r['question_id'] not in first_ids,-r['duration_seconds'],r['ordinal']))
            for row in pending:
                if (root/'PAUSE.json').exists():return
                durable(session,dict(status='generating_'+part,question_id=row['question_id'],pid=os.getpid(),gpu=gpu,utc=utc()))
                process_question(root,row,part,backend,adapter,gpu)
    durable(session,dict(status='completed',pid=os.getpid(),gpu=gpu,utc=utc()))
    return dict(status='completed',phase=phase,gpu=gpu)


def make_scorer(root):
    """接口：原BLIP数值函数原样复用；只旁路区分粗评分、补查视觉与集合表示额外前向。"""
    from videoqa_set_value.selection import ValueScorer
    class TrialScorer(ValueScorer):
        """不改模型、分数或候选，成本语义由实际调用入口确定。"""
        usage_scope='set_representation_extra_itm'
        question_id=None
        usage_ordinal=0

        def score_batch_features(self,encoded,images):
            """步骤1：标记粗评分旁路，返回值完全沿用冻结父类。"""
            self.usage_scope='coarse_question_itm'
            try:return super().score_batch_features(encoded,images)
            finally:self.usage_scope='set_representation_extra_itm'

        def visual_batch(self,images):
            """步骤2：补查仍只用纯视觉，额外头特征前向不回写选择。"""
            self.usage_scope='refinement_visual_only'
            try:return super().visual_batch(images)
            finally:self.usage_scope='set_representation_extra_itm'

        def _forward(self,encoded,images,kind):
            """步骤3：原前向/缓存已完成后写用途账本；成本不冒充免费或独立原片E2E。"""
            values,pre,forward=super()._forward(encoded,images,kind)
            ordinal=self.usage_ordinal;self.usage_ordinal+=1
            path=Path(root)/'blip_usage'/self.question_id/(str(ordinal)+'.json')
            value=dict(question_id=self.question_id,scope=self.usage_scope,kind=kind,frames=len(images),
                preprocess_seconds=pre,forward_seconds=forward,cache_reused=forward==0.)
            if not path.exists():durable(path,value)
            return values,pre,forward
    return TrialScorer(root)


def launch(root,phase):
    """接口：子进程关闭stdin并完整重定向；环境按BLIP/QA分离，不升级历史依赖。"""
    root=Path(root);assignment_path=root/'assignments'/(phase+'.json')
    if assignment_path.exists():
        allocation=read_json(assignment_path)['questions'];gpus=list(allocation)
        free=idle_gpus(root,phase)
        if not set(gpus)<=set(free):raise RuntimeError('Original physical GPU unavailable; do not relocate silently')
    else:gpus=idle_gpus(root,phase);assign(root,phase,gpus)
    children=[];python=PREP_PYTHON if phase.startswith('select_') else QA_PYTHON
    for gpu in gpus:
        log=(root/(phase+'_'+gpu+'.log')).open('a',buffering=1)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        process=subprocess.Popen([python,str(ENTRY),'--stage','worker','--run-id',root.name,'--phase',phase,'--gpu',gpu],
            cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        children.append((gpu,process,log))
    return children


def wait(root,children,phase,condition=None):
    """接口：观察进度不重新生成；失败暂停全流水线，等待在途原文正常落盘。"""
    root=Path(root);began=time.monotonic()
    while True:
        if any(p.poll() not in (None,0) for _,p,_ in children):raise RuntimeError('Worker failed: '+phase)
        if condition is not None and condition():return
        if not any(p.poll() is None for _,p,_ in children):
            if condition is not None and not condition():raise RuntimeError('Worker ended before phase completion')
            break
        calls=[p for p in (root/'qa').glob('*/calls/*/*.json') if p.parent.name!='synthetic']
        elapsed=time.monotonic()-began;completed=None;total=None
        if phase in PHASE_CAPS and (root/'input_plans'/(phase+'.json')).exists():
            phase_calls=[p for p in (root/'qa'/phase/'calls').glob('*/*.json') if p.parent.name!='synthetic']
            completed=sum(read_json(p)['status']=='completed' for p in phase_calls)
            total=read_json(root/'input_plans'/(phase+'.json'))['max_real_calls']
        durable(root/'heartbeat.json',dict(stage=phase,real_calls_started=len(calls),maximum_real_calls=5200,
            completed_phase_calls=completed,total_phase_calls=total,
            eta_seconds=elapsed*(total-completed)/completed if completed is not None and completed>=100 else None,
            pid=os.getpid(),sid=os.getsid(0),elapsed_seconds=elapsed,workers={g:p.pid for g,p,_ in children},utc=utc()))
        time.sleep(3)


def close(children):
    """接口：阶段结束释放所有工作模型；不通过杀进程中断正在返回的生成。"""
    for _,process,log in children:process.wait();log.close()


def audit_phase(root,phase):
    """接口：独立复算原始输出、共享引用与调用闭合，完成恢复只读不再生成。"""
    root=Path(root);stage=root/'qa'/phase;selected=rows(root,phase);expected={r['question_id'] for r in selected}
    conditions=CONDITIONS if phase=='pilot' else METHODS;counts=dict.fromkeys(conditions,0);artifacts={};physical=set();parse={}
    for method in conditions:
        paths=list((stage/'results'/method).glob('*.json'))
        if {p.stem for p in paths}!=expected:raise ValueError('QA result pairing incomplete')
        parse[method]=0
        for row in selected:
            q=row['question_id'];p=stage/'results'/method/(q+'.json');record=read_json(p)
            plan_path=root/'inputs'/phase/(q+'.json');frozen=read_json(root/'input_plans'/(phase+'.json'))
            if sha256(plan_path)!=frozen['exports'][q]:raise ValueError('Frozen input export changed')
            entry=read_json(plan_path)['methods'][method]
            if record['input_key']!=entry['input_key']:raise ValueError('Result input binding changed')
            observed=record['answer']['observed']
            if any(observed[k]!=entry['identity'][k] for k in ('prompt','input_token_ids','pixel_sha256','source_frames','visual_tokens')):
                raise ValueError('Actual observed input differs from frozen input')
            label=task.metric(scoring_row(root,row),record['answer']['raw_output'])
            if record['correct']!=label['correct'] or record['answer']['parsed_answer']!=label['pred_answer']:
                raise ValueError('Independent score recount differs')
            counts[method]+=int(label['correct']);parse[method]+=int(label['parse_failed']);physical.add(record['call_path'])
            artifacts[str(p.relative_to(root))]=sha256(p)
    calls=list((stage/'calls').glob('*/*.json'));real=[p for p in calls if p.parent.name!='synthetic']
    maximum=read_json(root/'input_plans'/(phase+'.json'))['max_real_calls']
    if len(real)!=maximum or {str(p) for p in real}!=physical:raise ValueError('Call budget/coverage differs')
    for cp in calls:
        call=read_json(cp);raw_path=stage/'raw_generations'/cp.parent.name/cp.name;raw=read_json(raw_path)
        if call['status']!='completed' or raw['generation_state']!={'started':True,'returned':True}:
            raise ValueError('Uncertain generation cannot pass audit')
        if raw['raw_generated_text'].strip()!=call['answer']['raw_output']:raise ValueError('Raw output changed')
        artifacts[str(cp.relative_to(root))]=sha256(cp);artifacts[str(raw_path.relative_to(root))]=sha256(raw_path)
    value=dict(phase=phase,questions=len(selected),counts=counts,parse_failures=parse,real_calls=len(real),
        synthetic_calls=len(calls)-len(real),artifacts=artifacts)
    dest=stage/'completion_audit.json'
    if dest.exists() and read_json(dest)!=value:raise ValueError('Completed phase changed')
    if not dest.exists():durable(dest,value)
    return value


def paired_interval(differences):
    """接口：每个独立内容组仅一道题，视频簇bootstrap等价于对这些视频重抽10000次。"""
    values=np.asarray(differences,dtype=float)
    if not len(values):raise ValueError('Empty video denominator')
    rng=np.random.default_rng(2027);samples=[]
    for _ in range(50):samples.extend(values[rng.integers(0,len(values),size=(200,len(values)))].mean(1).tolist())
    return list(map(float,np.quantile(samples,[.025,.975])))


def pilot_statistics(root):
    """接口：工程验收后计算事前先导门槛，审核面板先冻结，再等待实际画面核查。"""
    root=Path(root);audit_phase(root,'pilot');selected=rows(root,'pilot');labels=[]
    for row in selected:
        q=row['question_id'];labels.append([int(read_json(root/'qa/pilot/results'/m/(q+'.json'))['correct']) for m in CONDITIONS])
    y=np.asarray(labels);mixed=np.flatnonzero((y[:,:4].min(1)==0)&(y[:,:4].max(1)==1))
    gap=int(y[:,:4].max(1).sum()-y[:,:4].sum(0).max())
    intervals={m:paired_interval(y[:,0]-y[:,CONDITIONS.index(m)]) for m in ('G','W')}
    criteria=dict(engineering=True,mixed_at_least_40=len(mixed)>=40,oracle_gap_at_least_10=gap>=10,
        uniform_over_gray_lower_positive=intervals['G'][0]>0,uniform_over_wrong_video_lower_positive=intervals['W'][0]>0)
    ids=[r['question_id'] for r in selected[:6]]
    extra=[selected[i]['question_id'] for i in mixed if selected[i]['question_id'] not in ids][:6]
    panel=ids+extra
    result=dict(questions=200,counts=dict(zip(CONDITIONS,map(int,y.sum(0)))),mixed_videos=len(mixed),
        any_correct=int(y[:,:4].max(1).sum()),gap_to_best_fixed=gap,uniform_diagnostic_difference_95=intervals,
        criteria=criteria,quantitative_pass=all(criteria.values()),review_question_ids=panel,
        first_six=ids,mixed_six=extra,mixed_question_ids=[selected[i]['question_id'] for i in mixed],
        review_max_minutes=60,review_required_supported=10,review_required_tradeoffs=3)
    durable(root/'pilot_report.json',result);durable(root/'pilot_review_panel.json',dict(question_ids=panel,
        inputs={q:sha256(root/'inputs/pilot'/(q+'.json')) for q in panel},pilot_report_sha256=sha256(root/'pilot_report.json')))
    review_page(root,panel)
    return result


def review_page(root,ids):
    """接口：事前固定12题可点击画面页，事实审核只评价，不影响候选或问答。"""
    root=Path(root);parts=['<!doctype html><meta charset="utf-8"><title>LongVideo先导审核</title>']
    lookup={r['question_id']:r for r in rows(root)}
    for q in ids:
        row=lookup[q];scored=scoring_row(root,row)
        parts.append('<h2>'+html.escape(q)+'</h2><pre>'+html.escape(task.context(row))+'\n参考：'+scored['answer_letter']+'</pre>')
        for method in METHODS:
            record=read_json(root/'selections'/method/(q+'.json'))
            parts.append('<h3>'+html.escape(method)+'</h3>')
            for f in record['frames']:
                parts.append('<a href="'+html.escape(f['image'],quote=True)+'"><img width="240" src="'+html.escape(f['image'],quote=True)+'"></a>')
    path=root/'pilot_review.html';path.write_text('\n'.join(parts))


def review_passes(root):
    """接口：没有实际审核记录不得自动通过；未确认不计通过，也不能无限扩展面板。"""
    root=Path(root);path=root/'pilot_review.json'
    if not path.exists():return None
    record=read_json(path);panel=read_json(root/'pilot_review_panel.json');expected=panel['question_ids']
    if record['panel_sha256']!=sha256(root/'pilot_review_panel.json'):raise ValueError('Review used a different panel')
    if len(record['cases'])!=len(expected) or {c['question_id'] for c in record['cases']}!=set(expected):raise ValueError('Review pairing differs')
    if not 0<record['minutes']<=60:raise ValueError('Bounded review time differs')
    for case in record['cases']:
        if not case.get('evidence') or not isinstance(case.get('supported'),bool) or not isinstance(case.get('information_tradeoff'),bool):
            raise ValueError('Review requires actual evidence and explicit judgments')
    supported=sum(c['supported'] for c in record['cases'])
    report=read_json(root/'pilot_report.json');mixed=set(report.get('mixed_question_ids',report['mixed_six']))
    tradeoffs=sum(c['information_tradeoff'] and c['question_id'] in mixed for c in record['cases'])
    return dict(passed=supported>=10 and tradeoffs>=3,supported=supported,confirmed_mixed_tradeoffs=tradeoffs,
        review_sha256=sha256(path))


def training_assets(root):
    """接口：只加载训练800特征与先导/后续训练标签，验证结果路径不在本函数数据流内。"""
    from videoqa_audit.set_value_trainability import TrainingAssets
    root=Path(root);selected=rows(root,'train');xx=[];yy=[];reps=[];orders=[]
    for row in selected:
        q=row['question_id'];phase='pilot' if row['pilot'] else 'train_rest'
        plan=read_json(root/'inputs'/phase/(q+'.json'))
        with np.load(root/'value_features'/(q+'.npz'),allow_pickle=False) as z:x=np.stack([z[m] for m in METHODS]).astype(np.float32)
        y=[int(read_json(root/'qa'/phase/'results'/m/(q+'.json'))['correct']) for m in METHODS]
        rr=[]
        for group in plan['groups']:
            members=[METHODS.index(m) for m in group['methods'] if m in METHODS]
            if members:
                if len({y[i] for i in members})!=1 or any(not np.array_equal(x[i],x[members[0]]) for i in members):
                    raise ValueError('Same input has different labels/features')
                rr.append(members[0])
        xx.append(x);yy.append(y);reps.append(rr);orders.append([plan['methods'][m]['source_order'] for m in METHODS])
    return TrainingAssets(selected,np.stack(xx),np.asarray(yy,dtype=np.float32),reps,orders)


def train_and_predict(root):
    """接口：只训练一条CPU轨迹；训练通过后冻结400题预测，不读取验证标签或结果。"""
    import torch
    from videoqa_set_value.learner import SetValueHead
    from videoqa_audit.set_value_trainability import train_trajectory,score_inputs,select_methods
    root=Path(root);torch.set_num_threads(2);data=training_assets(root)
    trained=train_trajectory(data,root/'head',[r['question_id'] for r in data.rows])
    if trained['status']!='fit_passed':return dict(status='training_fit_failed',training=trained)
    model=SetValueHead().float();model.load_state_dict(torch.load(root/'head/state.pt',map_location='cpu',weights_only=False)['model'])
    old=SetValueHead().float();old.load_state_dict(torch.load(OLD_HEAD,map_location='cpu',weights_only=False)['model'])
    if sha256(OLD_HEAD)!=OLD_HEAD_SHA:raise ValueError('Old frozen head changed')
    selected=rows(root,'val');xx=[];orders=[]
    for row in selected:
        q=row['question_id'];plan=read_json(root/'inputs/val'/(q+'.json'))
        with np.load(root/'value_features'/(q+'.npz'),allow_pickle=False) as z:xx.append(np.stack([z[m] for m in METHODS]))
        orders.append([plan['methods'][m]['source_order'] for m in METHODS])
    x=np.asarray(xx,dtype=np.float32);scores=score_inputs(model,x);old_scores=score_inputs(old,x)
    choices=select_methods(scores,orders);old_choices=select_methods(old_scores,orders);mean_choices=select_methods(x[:,:,:,768].mean(2),orders)
    best=int(np.argmax(data.labels.sum(0)));predictions=[]
    for i,row in enumerate(selected):
        predictions.append(dict(question_id=row['question_id'],scores=scores[i].tolist(),chosen_method=METHODS[choices[i]],
            old_head_scores=old_scores[i].tolist(),old_head_method=METHODS[old_choices[i]],mean_itm_method=METHODS[mean_choices[i]],
            training_best_fixed_method=METHODS[best],feature_sha256=sha256(root/'value_features'/(row['question_id']+'.npz'))))
    value=dict(checkpoint_sha256=sha256(root/'head/state.pt'),old_head_sha256=OLD_HEAD_SHA,
        train_fixed_correct=dict(zip(METHODS,map(int,data.labels.sum(0)))),training_best_fixed_method=METHODS[best],
        predictions=predictions,validation_labels_read=False,utc=utc())
    dest=root/'validation_predictions.json'
    if dest.exists():
        if {k:v for k,v in read_json(dest).items() if k!='utc'}!={k:v for k,v in value.items() if k!='utc'}:
            raise ValueError('Frozen validation predictions changed')
    else:durable(dest,value)
    return dict(status='ready_for_validation',training=trained)


def validation_report(root):
    """接口：400独立视频一次评价；不依据验证收益选择检查点、候选或第二版结构。"""
    root=Path(root);audit_phase(root,'val');selected=rows(root,'val');frozen=read_json(root/'validation_predictions.json')
    lookup={p['question_id']:p for p in frozen['predictions']};values={m:[] for m in (*METHODS,'learned','mean_itm','old_head')}
    preference=[];records=[]
    for row in selected:
        q=row['question_id'];p=lookup[q]
        if sha256(root/'value_features'/(q+'.npz'))!=p['feature_sha256']:raise ValueError('Validation features changed')
        y={m:int(read_json(root/'qa/val/results'/m/(q+'.json'))['correct']) for m in METHODS}
        for m in METHODS:values[m].append(y[m])
        for key,field in [('learned','chosen_method'),('mean_itm','mean_itm_method'),('old_head','old_head_method')]:values[key].append(y[p[field]])
        plan=read_json(root/'inputs/val'/(q+'.json'));reps=[METHODS.index(g['representative']) for g in plan['groups']];pairs=[]
        for i in reps:
            for j in reps:
                if y[METHODS[i]] and not y[METHODS[j]]:
                    d=p['scores'][i]-p['scores'][j];pairs.append(1. if d>1e-8 else .5 if abs(d)<=1e-8 else 0.)
        if pairs:preference.append(float(np.mean(pairs)))
        records.append(dict(question_id=q,video_id=row['video_id'],duration_seconds=row['duration_seconds'],task_type=row['task_type'],
            labels=y,chosen_method=p['chosen_method'],preference_accuracy=float(np.mean(pairs)) if pairs else None))
    counts={m:sum(v) for m,v in values.items()};comparisons={};learned=np.array(values['learned'])
    for method,vector in values.items():
        if method=='learned':continue
        base=np.asarray(vector);comparisons[method]=dict(net_correct=int((learned-base).sum()),
            difference=float((learned-base).mean()),cluster_95_interval=paired_interval(learned-base),
            **{name:[row['question_id'] for row,yes in zip(selected,mask) if yes] for name,mask in (
                ('new_correct',(learned==1)&(base==0)),('degraded',(learned==0)&(base==1)),
                ('both_correct',(learned==1)&(base==1)),('both_wrong',(learned==0)&(base==0)) )})
    main=frozen['training_best_fixed_method'];criteria=dict(preference_videos_at_least_60=len(preference)>=60,
        preference_accuracy_at_least_60=bool(preference) and np.mean(preference)>=.6,
        net_gain_at_least_8=counts['learned']-counts[main]>=8,paired_lower_positive=comparisons[main]['cluster_95_interval'][0]>0,
        no_lower_than_best_fixed=counts['learned']>=max(counts[m] for m in METHODS))
    criteria={k:bool(v) for k,v in criteria.items()}
    strata={}
    for name,test in [('180–600s',lambda r:r['duration_seconds']<=600),('600–1800s',lambda r:r['duration_seconds']>600)]:
        ids=[i for i,r in enumerate(selected) if test(r)]
        strata[name]=dict(videos=len(ids),correct={m:sum(v[i] for i in ids) for m,v in values.items()})
    from videoqa_set_value.report import wilson
    result=dict(status='completed',questions=400,counts=counts,accuracy={m:v/400 for m,v in counts.items()},
        wilson_95={m:wilson(v,400) for m,v in counts.items()},
        main_fixed_comparator=main,comparisons=comparisons,preference_videos=len(preference),
        preference_accuracy=float(np.mean(preference)) if preference else None,criteria=criteria,
        transfer_recommended=all(criteria.values()),oracle_correct=sum(any(r['labels'].values()) for r in records),
        choices=dict(Counter(r['chosen_method'] for r in records)),duration_groups=strata,
        source_task_labels=dict(Counter(r['task_type'] for r in selected)),records=records,
        predictions_sha256=sha256(root/'validation_predictions.json'),costs=costs(root),utc=utc())
    durable(root/'report.json',result);return result


def costs(root):
    """接口：缓存共享物理成本只记一次，分离额外表示ITM、合成与真实QA，不冒充独立E2E。"""
    root=Path(root);blip={};teacher={}
    for path in (root/'blip_usage').glob('*/*.json'):
        record=read_json(path);group=blip.setdefault(record['scope'],dict(batches=0,new_batches=0,frames=0,new_frames=0,
            preprocess_seconds=0.,forward_seconds=0.))
        group['batches']+=1;group['new_batches']+=int(not record['cache_reused']);group['frames']+=record['frames']
        group['new_frames']+=record['frames']*int(not record['cache_reused'])
        group['preprocess_seconds']+=record['preprocess_seconds'];group['forward_seconds']+=record['forward_seconds']
    for path in (root/'qa').glob('*/calls/*/*.json'):
        call=read_json(path);kind='synthetic' if path.parent.name=='synthetic' else 'real_teacher'
        group=teacher.setdefault(kind,dict(calls=0,completed=0,failed_or_uncertain=0,parse_failed=0,generation_seconds=0.,
            qa_seconds=0.,output_tokens=0,peak_allocated_gib=0.))
        group['calls']+=1
        if call['status']!='completed':group['failed_or_uncertain']+=1;continue
        group['completed']+=1;observed=call['answer']['observed'];group['parse_failed']+=int(call['answer']['parsed_answer'] is None)
        group['generation_seconds']+=observed['generation_seconds'];group['qa_seconds']+=observed['qa_seconds']
        group['output_tokens']+=len(observed['generated_token_ids'])
        group['peak_allocated_gib']=max(group['peak_allocated_gib'],observed['peak_allocated_gib'])
    return dict(blip=blip,teacher=teacher,loads={p.name:read_json(p) for p in (root/'load_reports').glob('*.json')},
        training_seconds=read_json(root/'head/summary.json')['training_seconds'] if (root/'head/summary.json').exists() else 0.,
        meaning='共享资产准备与物理调用成本；非各方法独立原片直接E2E，导入/引用不伪造重复QA成本')


def freeze_subprocess(root,phase):
    """接口：CPU官方预处理必须在LLaVA冻结环境，协调器不污染自己的BLIP环境。"""
    root=Path(root)
    with (root/('freeze_'+phase+'.log')).open('a') as log:
        subprocess.run([QA_PYTHON,str(ENTRY),'--stage','freeze','--run-id',root.name,'--phase',phase],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)


def run(root):
    """接口：资产后自主执行先导；门槛通过接续唯一训练和独立验证，不启动目标评测QA。"""
    root=Path(root);children=[]
    try:
        preflight(root)
        if (root/'report.json').exists():return audit(root)
        if (root/'PAUSE.json').exists():raise RuntimeError('Paused pipeline requires inspection')
        if (root/'STOP.json').exists():return read_json(root/'STOP.json')
        if not (root/'pilot_report.json').exists():
            children=launch(root,'select_pilot');wait(root,children,'select_pilot');close(children);children=[]
            freeze_subprocess(root,'pilot');phase_protocol(root,'pilot')
            children=launch(root,'pilot');wait(root,children,'pilot');close(children);children=[]
            pilot=pilot_statistics(root)
        else:pilot=read_json(root/'pilot_report.json');audit_phase(root,'pilot')
        if not pilot['quantitative_pass']:
            result=dict(status='stopped_pilot_quantitative_gate',criteria=pilot['criteria'],utc=utc())
            durable(root/'STOP.json',result);return result
        # 步骤1：实际画面审核不可由统计自动冒充；待审核文件不会触发新问答或重复先导。
        while review_passes(root) is None:
            durable(root/'heartbeat.json',dict(stage='waiting_for_bounded_pilot_review',panel=str(root/'pilot_review.html'),
                pid=os.getpid(),sid=os.getsid(0),new_qa_while_waiting=0,utc=utc()));time.sleep(10)
        reviewed=review_passes(root)
        if not reviewed['passed']:
            result=dict(status='stopped_pilot_visual_gate',review=reviewed,utc=utc());durable(root/'STOP.json',result);return result
        durable(root/'pilot_gate.json',dict(passed=True,report_sha256=sha256(root/'pilot_report.json'),**reviewed))
        # 步骤2：先导教师已释放，才准备剩余资产；不能提前取验证反馈。
        if not (root/'input_plans/train_rest.json').exists() or not (root/'input_plans/val.json').exists():
            children=launch(root,'select_main');wait(root,children,'select_main');close(children);children=[]
            freeze_subprocess(root,'train_rest');freeze_subprocess(root,'val')
        phase_protocol(root,'train_rest');children=launch(root,'main')
        done=lambda:all((root/'qa/train_rest/results'/m/(r['question_id']+'.json')).exists() for r in rows(root,'train_rest') for m in METHODS)
        wait(root,children,'train_rest',done);audit_phase(root,'train_rest')
        trained=train_and_predict(root)
        if trained['status']!='ready_for_validation':
            result=dict(status='stopped_training_fit',training=trained,utc=utc());durable(root/'STOP.json',result)
            wait(root,children,'stopped');close(children);children=[];return result
        phase_protocol(root,'val');durable(root/'release_validation.json',dict(predictions_sha256=sha256(root/'validation_predictions.json'),utc=utc()))
        wait(root,children,'val');close(children);children=[]
        result=validation_report(root);audit(root)
        durable(root/'heartbeat.json',dict(stage='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()));return result
    except BaseException as error:
        durable(root/'PAUSE.json',dict(error=repr(error),traceback=traceback.format_exc(),utc=utc()));raise
    finally:close(children)


def audit(root):
    """接口：完成恢复只验证已完成资产/调用；不调用模型、训练器或触发下一阶段。"""
    root=Path(root);verify(root,full=True);phases={}
    for phase in PHASE_CAPS:
        if (root/'qa'/phase/'completion_audit.json').exists():phases[phase]=audit_phase(root,phase)
    calls=list((root/'qa').glob('*/calls/*/*.json'));real=[p for p in calls if p.parent.name!='synthetic']
    if len(real)>5200 or len(calls)-len(real)>8:raise ValueError('Total teacher budget exceeded')
    for path in (root/'selection_checkpoints').glob('*.json'):
        for relative,expected in read_json(path)['files'].items():
            if sha256(root/relative)!=expected:raise ValueError('Frozen selection asset changed')
    if (root/'validation_predictions.json').exists():
        predicted=read_json(root/'validation_predictions.json')
        if predicted['checkpoint_sha256']!=sha256(root/'head/state.pt'):raise ValueError('Prediction head changed')
        for p in predicted['predictions']:
            if sha256(root/'value_features'/(p['question_id']+'.npz'))!=p['feature_sha256']:raise ValueError('Validation feature changed')
    value=dict(status='audit_passed',phases=list(phases),real_calls=len(real),synthetic_calls=len(calls)-len(real),
        new_generations=0,updates=0,utc=utc())
    durable(root/'completion_audit.json',value);return value
