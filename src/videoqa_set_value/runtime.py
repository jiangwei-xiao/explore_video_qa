"""集合价值学习实验调度：准备、教师监督、固定训练、先预测后验证、严格恢复。"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import copy
import hashlib
import multiprocessing as mp
import os
import shutil
import subprocess
import time
import traceback
import numpy as np
from PIL import Image

from videoqa_runtime.common import ROOT,read_json,sha256,offline_environment
from videoqa_full.state import durable,digest,exclusive,invoke_once,utc
from videoqa_full.prepare import environment
from videoqa_lmms_bridge.common import framework_identity,LMMS
from videoqa_set_value.data import METHODS
from videoqa_set_value.selection import ValueScorer,prepare_one
from videoqa_set_value.task import cpu_tools,cpu_identity,make_adapter,observe,answer_from_raw,metric
from videoqa_set_value.learner import fit,predict_validation

PREP_PYTHON='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'
QA_PYTHON='/App/conda/envs/explore_video_qa_lmms/bin/python'
MODEL='/home/models/lmms-lab/LLaVA-Video-7B-Qwen2'
_TOOLS=None


def public_rows(root,split=None):
    """接口：只按固定清单/划分投影任务；训练/验证名单不能由答案或模型输出改变。"""
    rows=read_json(Path(root)/'manifest.json')['rows']
    return [r for r in rows if split is None or r['split']==split]


def preflight(root):
    """接口：数据就绪后冻结代码、环境、权重和LMMs接口，原评测与模型目录只读。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():return verify(root)
    manifest=read_json(root/'manifest.json')
    if len(manifest['rows'])!=800 or manifest['train_questions']!=600 or manifest['val_questions']!=200:raise ValueError('Fixed 800-question denominator differs')
    inventory=read_json(ROOT/'configs/local_model_inventory.json');models={}
    for descriptor in inventory['models'].values():
        for name,info in descriptor['files'].items():
            path=Path(descriptor['path'])/name
            if sha256(path)!=info['sha256']:raise ValueError('Frozen model bytes changed')
            models[str(path)]=dict(sha256=info['sha256'],bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
    code=sorted(set([*ROOT.glob('src/videoqa_set_value/*.py'),ROOT/'scripts/run_set_value_learning.py',
        ROOT/'scripts/prepare_nextqa_pilot.py',ROOT/'scripts/prepare_set_value_data.py',
        *ROOT.glob('src/videoqa_methods/*.py'),*ROOT.glob('src/videoqa_runtime/*.py'),
        ROOT/'src/videoqa_generalization/llava_observe.py',ROOT/'src/videoqa_lmms_bridge/llava_adapter.py',
        ROOT/'src/videoqa_full/state.py',ROOT/'src/videoqa_budget32/frames.py',ROOT/'configs/retrieval_density_v1.json']))
    fingerprints={str(p.relative_to(ROOT)):sha256(p) for p in code}
    for name in fingerprints:
        dest=root/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    sources={r['video_path']:r['video_sha256'] for r in manifest['rows']}
    task=LMMS/'lmms_eval/tasks/nextqa/utils.py'
    value=dict(task='set_value_learning_pilot',manifest_sha256=sha256(root/'manifest.json'),code=fingerprints,
        models=models,video_hashes=sources,video_stats={p:dict(bytes=Path(p).stat().st_size,mtime_ns=Path(p).stat().st_mtime_ns) for p in sources},
        framework=framework_identity(),task_formatter_sha256=sha256(task),
        environments=dict(prepare=environment(PREP_PYTHON),llava=environment(QA_PYTHON)),
        teacher=dict(model=MODEL,dtype='BF16',attention='SDPA',frame_budget=8,visual_tokens=1680,max_new_tokens=16,seed=2027),
        maximum_real_generations=3200,maximum_synthetic_generations=8,validation_prediction_before_generation=True,utc=utc())
    if (root/'preparation_reuse.json').exists():
        receipt=read_json(root/'preparation_reuse.json')
        if sha256(Path(receipt['parent_run'])/'protocol.json')!=receipt['parent_protocol_sha256']:
            raise ValueError('Parent preparation identity changed')
        if read_json(Path(receipt['parent_run'])/'protocol.json')['models']!=models:
            raise ValueError('Imported BLIP model identity differs')
        value['preparation_reuse_sha256']=sha256(root/'preparation_reuse.json')
    durable(root/'protocol.json',value);return value


def verify(root,full=False):
    """接口：恢复不接受输入/权重/代码漂移；完整哈希用于最终验收。"""
    root=Path(root);p=read_json(root/'protocol.json')
    if sha256(root/'manifest.json')!=p['manifest_sha256']:raise ValueError('Manifest changed')
    if p.get('preparation_reuse_sha256') is not None and sha256(root/'preparation_reuse.json')!=p['preparation_reuse_sha256']:
        raise ValueError('Preparation reuse receipt changed')
    for name,fp in p['code'].items():
        if sha256(ROOT/name)!=fp:raise ValueError('Frozen code changed: '+name)
    for path,descriptor in {**p['models'],**p['video_stats']}.items():
        st=Path(path).stat()
        if (st.st_size,st.st_mtime_ns)!=(descriptor['bytes'],descriptor['mtime_ns']):raise ValueError('Source/model stat changed')
        if full:
            expected=descriptor.get('sha256',p['video_hashes'].get(path))
            if sha256(path)!=expected:raise ValueError('Source/model bytes changed')
    if framework_identity()!=p['framework'] or sha256(LMMS/'lmms_eval/tasks/nextqa/utils.py')!=p['task_formatter_sha256']:
        raise ValueError('Frozen framework/task changed')
    if full:verify_prepared_assets(root)
    return p


def compare_parent_preparation(root):
    """接口：技术读帧修正必须保持父准备的源选择/RGB/表示；失败在首个教师前暂停。"""
    root=Path(root)
    if not (root/'preparation_reuse.json').exists():return
    receipt=read_json(root/'preparation_reuse.json');parent=Path(receipt['parent_run']);checked=[];error=0.
    for q,fp in receipt['parent_completed_selections'].items():
        if sha256(parent/'selection_checkpoints'/(q+'.json'))!=fp:raise ValueError('Parent checkpoint changed')
        for m in METHODS:
            a=read_json(parent/'selections'/m/(q+'.json'));b=read_json(root/'selections'/m/(q+'.json'))
            keys=lambda item:[(f['source_pts'],f['source_frame_index'],f['rgb_sha256']) for f in item['frames']]
            if keys(a)!=keys(b):raise ValueError('Sequential decoder changes an existing source/RGB input: '+q)
        with np.load(parent/'value_features'/(q+'.npz'),allow_pickle=False) as a,np.load(root/'value_features'/(q+'.npz'),allow_pickle=False) as b:
            error=max(error,max(float(np.max(np.abs(a[m]-b[m]))) for m in METHODS))
        if error>1e-6:raise ValueError('Parent frozen representation differs')
        checked.append(q)
    durable(root/'parent_preparation_control.json',dict(questions=len(checked),method_inputs=len(checked)*4,
        source_order_and_rgb_exact=True,maximum_feature_abs_error=error,new_teacher_generations=0,utc=utc()))


def verify_prepared_assets(root):
    """接口：最终审计与完成恢复核对选帧/特征/图像/实际输入指纹，不重新运行模型。"""
    root=Path(root);prepared=read_json(root/'selections_complete.json')
    # 步骤1：全部派生资产由题目检查点绑定，不能只检查代码和原视频。
    for q,fp in prepared['checkpoints'].items():
        cp=root/'selection_checkpoints'/(q+'.json')
        if sha256(cp)!=fp:raise ValueError('Frozen selection checkpoint changed')
        for name,expected in read_json(cp)['files'].items():
            if sha256(root/name)!=expected:raise ValueError('Frozen prepared asset changed: '+name)
    for q,expected in read_json(root/'inputs_complete.json')['exports'].items():
        if sha256(root/'inputs'/(q+'.json'))!=expected:raise ValueError('Frozen actual input changed')
    # 步骤2：验证预测必须对应同一个特征快照和最后一轮小头。
    if (root/'validation_predictions.json').exists():
        frozen=read_json(root/'validation_predictions.json')
        if frozen['checkpoint_sha256']!=sha256(root/'head/checkpoint.pt'):raise ValueError('Frozen learned checkpoint changed')
        for prediction in frozen['predictions']:
            if sha256(root/'value_features'/(prediction['question_id']+'.npz'))!=prediction['feature_sha256']:
                raise ValueError('Frozen validation feature changed')


def assign(root,stage,gpus):
    """接口：视频级固定分卡，按时长贪心平衡；同题四输入总在同卡同进程。"""
    root=Path(root);path=root/'assignments'/(stage+'.json')
    if path.exists():return read_json(path)
    rows=public_rows(root);groups={}
    for row in rows:groups.setdefault(row['video_id'],[]).append(row)
    load={g:0. for g in gpus};assignments={g:[] for g in gpus}
    ordered=sorted(groups.values(),key=lambda rs:(-read_json(rs[0]['index_path'])['video']['duration_seconds'],rs[0]['video_id']))
    for rs in ordered:
        gpu=min(gpus,key=lambda g:(load[g],g));assignments[gpu].extend(r['question_id'] for r in rs)
        load[gpu]+=read_json(rs[0]['index_path'])['video']['duration_seconds']*len(rs)
    value=dict(questions=assignments,gpu_uuids={g:subprocess.check_output(['nvidia-smi','-i',g,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip() for g in gpus},utc=utc())
    durable(path,value);return value


def idle_gpus(root,stage):
    """接口：只取启动时真实空闲卡，阶段内不扩卡，无空闲时后台等待。"""
    from videoqa_runtime.baseline_run import available_gpus
    while True:
        try:free,_=available_gpus(list(map(str,range(8))));return free
        except RuntimeError as error:
            if str(error)!='No requested GPU is currently idle':raise
            durable(Path(root)/'heartbeat.json',dict(stage=stage,status='waiting_for_idle_gpus',pid=os.getpid(),utc=utc()));time.sleep(5)


def selection_worker(root,gpu):
    """接口：一张卡常驻BLIP准备指定题；不运行教师答案或任何合成生成。"""
    root=Path(root);verify(root);offline_environment()
    assigned=set(read_json(root/'assignments/selection.json')['questions'][gpu]);rows=[r for r in public_rows(root) if r['question_id'] in assigned]
    pending=[r for r in rows if not (root/'selection_checkpoints'/(r['question_id']+'.json')).exists()]
    if not pending:return
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(2027)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter();scorer=ValueScorer(root)
    durable(root/'workers'/('selection_'+gpu+'_load.json'),dict(load_seconds=time.perf_counter()-started,synthetic_forwards=0,utc=utc()))
    for row in pending:
        if (root/'PAUSE.json').exists():return
        durable(root/'workers'/('selection_'+gpu+'.json'),dict(status='running',pid=os.getpid(),gpu=gpu,question_id=row['question_id'],utc=utc()))
        prepare_one(root,row,scorer)
    durable(root/'workers'/('selection_'+gpu+'.json'),dict(status='completed',pid=os.getpid(),questions=len(rows),utc=utc()))


def initialize_cpu():
    """接口：spawn进程各自加载CPU工具，避免首次懒导入和Torch fork线程死锁。"""
    global _TOOLS
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);_TOOLS=cpu_tools()


def load_images(record):
    """接口：只读取事前保存的真实RGB，并逐帧核对内容哈希。"""
    ims=[]
    try:
        for f in record['frames']:
            im=Image.open(f['image']).convert('RGB')
            if hashlib.sha256(np.ascontiguousarray(np.asarray(im)).tobytes()).hexdigest()!=f['rgb_sha256']:raise ValueError('Source RGB changed')
            ims.append(im)
        return ims
    except BaseException:
        for im in ims:im.close()
        raise


def freeze_one(job):
    """接口：同题四组的实际像素/提示/源身份完全一致才共享，答案不参与分组。"""
    root,row=Path(job[0]),job[1];q=row['question_id'];dest=root/'inputs'/(q+'.json')
    if dest.exists():return read_json(dest)
    methods={};groups=[];seen={}
    for i,m in enumerate(METHODS):
        path=root/'selections'/m/(q+'.json');record=read_json(path);ims=load_images(record)
        try:identity=cpu_identity(_TOOLS,ims,row)
        finally:
            for im in ims:im.close()
        source=[[f['source_pts'],f['source_frame_index']] for f in record['frames']]
        key=digest(dict(source_order=source,rgb=[f['rgb_sha256'] for f in record['frames']],input=identity))
        methods[m]=dict(selection_path=str(path),selection_sha256=sha256(path),identity=identity,source_order=source,input_key=key)
        if key not in seen:
            seen[key]=len(groups);groups.append(dict(input_key=key,representative_method_index=i,method_indices=[i]))
        else:groups[seen[key]]['method_indices'].append(i)
    value=dict(question_id=q,split=row['split'],methods=methods,groups=groups,uses_answer_for_partition=False,utc=utc())
    durable(dest,value);return value


def freeze_inputs(root):
    """接口：正式教师前冻结800份输入及两个调用额度；无模型前向。"""
    root=Path(root);verify(root)
    if (root/'inputs_complete.json').exists():return read_json(root/'inputs_complete.json')
    rows=public_rows(root);exports={};budget=dict(train=0,val=0)
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=initialize_cpu) as pool:
        for item in pool.map(freeze_one,[(str(root),r) for r in rows]):
            exports[item['question_id']]=sha256(root/'inputs'/(item['question_id']+'.json'));budget[item['split']]+=len(item['groups'])
            durable(root/'freeze_progress.json',dict(completed=len(exports),total=800,utc=utc()))
    if len(exports)!=800 or budget['train']>2400 or budget['val']>800:raise ValueError('Input/call budgets do not close')
    value=dict(questions=800,method_results=3200,exports=exports,real_call_budgets=budget,maximum_synthetic_calls=8,utc=utc())
    durable(root/'inputs_complete.json',value);return value


def phase_protocol(root,split):
    """接口：教师两阶段独立调用ID；验证协议绑定事前已冻结预测。"""
    root=Path(root);stage=root/'qa'/split;path=stage/'protocol.json'
    predictions=sha256(root/'validation_predictions.json') if split=='val' else None
    if split=='val' and read_json(root/'validation_predictions.json')['checkpoint_sha256']!=sha256(root/'head/checkpoint.pt'):
        raise ValueError('Validation prediction head changed')
    value=dict(parent_protocol_sha256=sha256(root/'protocol.json'),inputs_sha256=sha256(root/'inputs_complete.json'),
        split=split,max_real_calls=read_json(root/'inputs_complete.json')['real_call_budgets'][split],validation_predictions_sha256=predictions)
    if path.exists():
        if read_json(path)!=value:raise ValueError('Teacher phase fingerprint changed')
    else:durable(path,value)
    return stage


def restore_raw(stage,q,method,row,gpu):
    """接口：完整原文只恢复解析，不确定生成/换卡/指纹变化禁止重发。"""
    stage=Path(stage);cp=stage/'calls'/method/(q+'.json')
    if not cp.exists():return None
    call=read_json(cp)
    if call['protocol_sha256']!=sha256(stage/'protocol.json') or call['gpu']!=gpu:raise ValueError('Physical call fingerprint/GPU changed')
    if call['status']=='completed':return call['answer']
    path=stage/'raw_generations'/method/(q+'.json')
    if not path.exists():raise RuntimeError('Uncertain teacher generation; no retry')
    raw=read_json(path)
    if raw['generation_state']!={'started':True,'returned':True}:raise RuntimeError('Incomplete teacher return; no retry')
    if raw['identity']!=dict(question_id=q,method=method,gpu=gpu,protocol_sha256=call['protocol_sha256']):raise ValueError('Raw generation identity differs')
    answer=answer_from_raw(raw,row);call.update(status='completed',answer=answer,recovered_postprocessing=True,generation_state=raw['generation_state'])
    durable(cp,call);return answer


def process_question(root,row,backend,adapter,gpu):
    """接口：同题共享划分先冻结；真实generate先落盘原文，随后发布四方法结果。"""
    root=Path(root);q=row['question_id'];stage=root/'qa'/row['split'];plan=read_json(root/'inputs'/(q+'.json'))
    if sha256(root/'inputs'/(q+'.json'))!=read_json(root/'inputs_complete.json')['exports'][q]:raise ValueError('Frozen input changed')
    groups=plan['groups'];ordinal=next(i for i,r in enumerate(public_rows(root)) if r['question_id']==q)
    shift=ordinal%len(groups);order=groups[shift:]+groups[:shift]
    for group in order:
        method=METHODS[group['representative_method_index']];entry=plan['methods'][method]
        if all((stage/'results'/METHODS[i]/(q+'.json')).exists() for i in group['method_indices']):continue
        answer=restore_raw(stage,q,method,row,gpu)
        if answer is None:
            if (root/'PAUSE.json').exists():return
            record=read_json(entry['selection_path']);ims=load_images(record)
            try:
                identity=dict(question_id=q,method=method,gpu=gpu,protocol_sha256=sha256(stage/'protocol.json'))
                def generate(state):
                    return observe(backend,adapter,ims,record['selection'],row,state,stage/'raw_generations'/method/(q+'.json'),identity)
                answer,_,_=invoke_once(stage,q,method,identity['protocol_sha256'],gpu,generate)
            finally:
                for im in ims:im.close()
        expected=entry['identity'];observed=answer['observed']
        if any(observed[k]!=expected[k] for k in ('prompt','input_token_ids','pixel_sha256','source_frames','visual_tokens')):
            raise ValueError('Actual teacher input differs from frozen CPU input')
        if observed['visual_tokens']!=1680 or observed['max_new_tokens']!=16:raise ValueError('Teacher token budget changed')
        score=metric(row,answer['raw_output'])
        for i in group['method_indices']:
            m=METHODS[i];dest=stage/'results'/m/(q+'.json')
            value=dict(question_id=q,method=m,split=row['split'],answer=answer,correct=score['correct'],
                input_key=entry['input_key'],selection_sha256=plan['methods'][m]['selection_sha256'],
                call_method=method,call_path=str(stage/'calls'/method/(q+'.json')),shared=(m!=method),gpu=gpu)
            if dest.exists():
                if read_json(dest)!=value:raise ValueError('Completed method result changed')
            else:durable(dest,value)


def teacher_worker(root,gpu):
    """接口：同一卡教师常驻跨训练/验证阶段，仅一次合成；等待小头冻结预测后再验证。"""
    root=Path(root);verify(root);offline_environment();assignment=read_json(root/'assignments/teacher.json')
    qids=set(assignment['questions'][gpu]);rows=[r for r in public_rows(root) if r['question_id'] in qids]
    session=root/'workers'/('teacher_'+gpu+'.json')
    if session.exists():raise RuntimeError('Existing teacher session requires safe state audit, not automatic restart')
    import torch
    from videoqa_runtime.llava_backend import LlavaBackend
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    durable(session,dict(status='loading',pid=os.getpid(),gpu=gpu,utc=utc()));backend=LlavaBackend();adapter=make_adapter(backend)
    durable(root/'load_reports'/('teacher_'+gpu+'.json'),backend.load_report)
    # 步骤1：每个物理教师进程一次合成，两阶段共享同进程，合成总量≤8。
    synthetic=dict(question_id='synthetic-'+gpu,question='Which synthetic color is visible?',options=['Gray','Red','Blue','Green','Black'],answer_index=0)
    ims=[Image.new('RGB',(384,384),'gray') for _ in range(8)]
    selection=dict(video=dict(path='synthetic-nextqa.mp4',duration_seconds=8.),selected_frames=[dict(timestamp_seconds=float(i)) for i in range(8)])
    stage=root/'qa/train';method='synthetic';q=synthetic['question_id'];identity=dict(question_id=q,method=method,gpu=gpu,protocol_sha256=sha256(stage/'protocol.json'))
    try:
        invoke_once(stage,q,method,identity['protocol_sha256'],gpu,
            lambda state:observe(backend,adapter,ims,selection,synthetic,state,stage/'raw_generations'/method/(q+'.json'),identity))
    finally:
        for im in ims:im.close()
    for split in ('train','val'):
        # 步骤2：验证门只读取已冻结预测发布信号，不能提前启动验证答案。
        if split=='val':
            while not (root/'release_validation.json').exists():
                if (root/'PAUSE.json').exists() or (root/'STOP.json').exists():return
                durable(session,dict(status='waiting_for_validation_predictions',pid=os.getpid(),gpu=gpu,utc=utc()));time.sleep(2)
            release=read_json(root/'release_validation.json')
            if release['predictions_sha256']!=sha256(root/'validation_predictions.json'):raise ValueError('Frozen validation prediction changed')
        for row in rows:
            if row['split']!=split:continue
            if (root/'PAUSE.json').exists():return
            durable(session,dict(status='generating_'+split,pid=os.getpid(),gpu=gpu,question_id=row['question_id'],utc=utc()))
            process_question(root,row,backend,adapter,gpu)
    durable(session,dict(status='completed',pid=os.getpid(),gpu=gpu,utc=utc()))


def audit_phase(root,split):
    """接口：每阶段四方法结果/物理调用闭合，严格复算标签；已有审计恢复保持字节。"""
    root=Path(root);stage=root/'qa'/split;rows=public_rows(root,split);expected={r['question_id'] for r in rows}
    artifacts={};calls=set();correct=dict.fromkeys(METHODS,0);parse_failures=0
    for m in METHODS:
        paths=list((stage/'results'/m).glob('*.json'))
        if {p.stem for p in paths}!=expected:raise ValueError('Method question pairing incomplete')
        for row in rows:
            q=row['question_id'];p=stage/'results'/m/(q+'.json');r=read_json(p);a=r['answer'];score=metric(row,a['raw_output'])
            if r['correct']!=score['correct'] or a['parsed_answer']!=score['pred_answer']:raise ValueError('Independent label recount differs')
            correct[m]+=int(r['correct']);calls.add(r['call_path']);artifacts[str(p.relative_to(stage))]=sha256(p)
    raw_calls=list((stage/'calls').glob('*/*.json'));real=[p for p in raw_calls if p.parent.name in METHODS]
    if len(real)!=read_json(root/'inputs_complete.json')['real_call_budgets'][split] or {str(p) for p in real}!=calls:raise ValueError('Real physical call budget differs')
    for p in raw_calls:
        call=read_json(p);rawpath=stage/'raw_generations'/p.parent.name/p.name
        if call['status']!='completed':raise ValueError('Uncertain call cannot close phase')
        raw=read_json(rawpath)
        if raw['generation_state']!={'started':True,'returned':True} or raw['raw_generated_text'].strip()!=call['answer']['raw_output']:raise ValueError('Raw generation mismatch')
        if p.parent.name in METHODS:parse_failures+=call['answer']['parsed_answer'] is None
        artifacts[str(p.relative_to(stage))]=sha256(p);artifacts[str(rawpath.relative_to(stage))]=sha256(rawpath)
    payload=dict(split=split,method_results=len(rows)*4,physical_real_calls=len(real),correct=correct,
        physical_parse_failures=parse_failures,artifacts=artifacts,protocol_sha256=sha256(stage/'protocol.json'))
    dest=stage/'completion_audit.json'
    if dest.exists():
        old=read_json(dest)
        if {k:v for k,v in old.items() if k!='utc'}!=payload:raise ValueError('Completed teacher phase changed')
        return old
    result=dict(payload,utc=utc());durable(dest,result);return result


def launch_workers(root,kind,gpus):
    """接口：阶段启动时冻结分卡与长期进程，完整日志，子进程不依赖SSH。"""
    root=Path(root);assign(root,kind,gpus);children=[]
    for gpu in gpus:
        log=(root/('worker_'+kind+'_'+gpu+'.log')).open('a',buffering=1)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        python=PREP_PYTHON if kind=='selection' else QA_PYTHON
        process=subprocess.Popen([python,str(ROOT/'scripts/run_set_value_learning.py'),'--run-id',root.name,'--stage',kind+'_worker','--gpu',gpu],
            cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        children.append((gpu,process,log))
    return children


def wait_workers(root,children,stage,condition=None):
    """接口：只观察具体进程；超时不重启；失败写暂停并等待正在返回的调用落盘。"""
    root=Path(root);began=time.monotonic()
    while True:
        if any(p.poll() not in (None,0) for _,p,_ in children):raise RuntimeError(stage+' worker failed; no automatic retry')
        if condition is not None and condition():return
        if not any(p.poll() is None for _,p,_ in children):
            if condition is not None and not condition():raise RuntimeError('Workers ended before '+stage+' completed')
            return
        # 步骤1：仅记录进度与吞吐，不按中途正确率决定继续；观察超时不触发重启。
        if stage=='selection':
            completed=len(list((root/'selection_checkpoints').glob('*.json')));total=800
        else:
            split='train' if stage=='teacher_train' else 'val'
            completed=sum(len(list((root/'qa'/split/'results'/m).glob('*.json'))) for m in METHODS)
            total=2400 if split=='train' else 800
        elapsed=time.monotonic()-began
        durable(root/'heartbeat.json',dict(stage=stage,pid=os.getpid(),sid=os.getsid(0),completed=completed,total=total,
            stage_elapsed_seconds=elapsed,eta_seconds=elapsed*(total-completed)/completed if completed>=100 else None,
            workers={g:p.pid for g,p,_ in children},utc=utc()));time.sleep(3)


def run(root):
    """接口：后台自动衔接准备→2400上限监督→一次训练→先预测后800上限验证→报告。"""
    root=Path(root);children=[]
    with exclusive(root/'coordinator.lock'):
        try:
            durable(root/'coordinator_session.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            preflight(root)
            if (root/'report.json').exists():
                verify(root,full=True);audit_phase(root,'train');audit_phase(root,'val');return read_json(root/'report.json')
            if (root/'PAUSE.json').exists():raise RuntimeError('Paused pipeline requires inspection')
            if not (root/'selections_complete.json').exists():
                children=launch_workers(root,'selection',idle_gpus(root,'selection'));wait_workers(root,children,'selection')
                for _,p,log in children:p.wait();log.close()
                children=[]
                checkpoints={r['question_id']:sha256(root/'selection_checkpoints'/(r['question_id']+'.json')) for r in public_rows(root)}
                compare_parent_preparation(root)
                durable(root/'selections_complete.json',dict(checkpoints=checkpoints,utc=utc()))
            if not (root/'inputs_complete.json').exists():
                with (root/'freeze_inputs.log').open('a') as log:
                    subprocess.run([QA_PYTHON,str(ROOT/'scripts/run_set_value_learning.py'),'--run-id',root.name,'--stage','freeze_inputs'],
                        cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
            phase_protocol(root,'train')
            children=launch_workers(root,'teacher',idle_gpus(root,'teacher'))
            train_done=lambda:all((root/'qa/train/results'/m/(r['question_id']+'.json')).exists() for r in public_rows(root,'train') for m in METHODS)
            wait_workers(root,children,'teacher_train',train_done);audit_phase(root,'train')
            durable(root/'heartbeat.json',dict(stage='training_small_head',pid=os.getpid(),utc=utc()))
            trained=fit(root)
            if trained['status']!='trained':
                durable(root/'STOP.json',dict(reason='No within-question preference signal',utc=utc()))
                wait_workers(root,children,'teacher_stopped');return trained
            predict_validation(root);phase_protocol(root,'val')
            durable(root/'release_validation.json',dict(predictions_sha256=sha256(root/'validation_predictions.json'),utc=utc()))
            wait_workers(root,children,'teacher_validation');audit_phase(root,'val')
            from videoqa_set_value.report import report
            result=report(root);durable(root/'heartbeat.json',dict(stage='completed',pid=os.getpid(),utc=utc()));return result
        except BaseException as error:
            durable(root/'PAUSE.json',dict(error=repr(error),traceback=traceback.format_exc(),utc=utc()));raise
        finally:
            for _,p,log in children:
                if p.poll() is None:p.wait()
                log.close()


def wait_data_and_run(root,data_pid):
    """接口：仅观察已确认的数据准备进程，数据完整后接续正式调度，不自动重启准备。"""
    root=Path(root)
    with exclusive(root/'data_transition.lock'):
        while not (root/'manifest.json').exists():
            # 步骤1：进程句柄和命令身份同时核对；不能以旧心跳代替实际存活。
            command=Path('/proc')/str(data_pid)/'cmdline'
            if not command.exists():raise RuntimeError('Data process ended before manifest; inspect data logs, do not restart automatically')
            content=command.read_bytes().decode().replace('\0',' ')
            if 'scripts/prepare_set_value_data.py' not in content or root.name not in content:
                raise RuntimeError('Data process identity changed')
            durable(root/'transition_heartbeat.json',dict(stage='waiting_for_data',pid=os.getpid(),sid=os.getsid(0),
                data_pid=data_pid,downloads=read_json(root/'acquisition_status.json'),utc=utc()))
            time.sleep(5)
        # 步骤2：清单闭合与代码冻结仍由正式preflight执行，下载结束不绕过验收。
        return run(root)
