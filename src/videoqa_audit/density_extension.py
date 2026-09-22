"""冻结开发200题的增量执行：旧结果只读复用，新150题两方法独立处理。"""
import contextlib
import fcntl
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import shutil
import time
import traceback
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256,offline_environment
from videoqa_runtime.baseline_run import preflight,available_gpus
from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_runtime.protocol import question_text,parse_answer
from videoqa_methods.followups import CallLedger,durable,utc,save_features
from videoqa_methods.density_pipeline import DensityScorer,prepare_density as prepare_v1
from videoqa_density_soft.density_pipeline import prepare_density as prepare_soft
from videoqa_methods.density_run import verify_selection as verify_v1
from videoqa_density_soft.density_run import verify_selection as verify_soft
from videoqa_methods.region_context import read_exact_frames
from videoqa_methods.density_report import render_board

MANIFEST=ROOT/'data/manifests/video_mme_development_200_20260920.json'
LINKS=ROOT/'data/manifests/video_mme_development_200_baseline_links_20260920.json'
ORIGINALS={'v1':ROOT/'outputs/methods/retrieval_density_v1_videomme50_20260920_r1',
           'soft':ROOT/'outputs/methods/density_soft_videomme50_20260920_r1'}
VARIANTS=('v1','soft')


def protocol():
    """冻结200题身份、500条复用结果和新增执行代码；明确300次生成上限。"""
    manifest=read_json(MANIFEST);links=read_json(LINKS)
    assert sha256(MANIFEST)==links['manifest_sha256']
    sources={v['path']:v['sha256'] for item in links['references'].values() for v in item.values()}
    for base in ORIGINALS.values():
        for q in manifest['old_question_ids']:
            path=base/'results'/f'{q}.json';sources[str(path.relative_to(ROOT))]=sha256(path)
    # 步骤1：核对冻结版本的代码，新增执行器不替换任何旧文件。
    frozen=read_json(ORIGINALS['soft']/'protocol.json')
    for name,digest in {**frozen['code_sha256'],**sources}.items():
        if sha256(ROOT/name)!=digest:raise ProtocolError('Frozen dependency changed: '+name)
    files=[MANIFEST,LINKS,ROOT/'src/videoqa_audit/density_extension.py',ROOT/'src/videoqa_audit/density_extension_report.py',
           ROOT/'scripts/run_density_extension.py',ROOT/'scripts/prepare_density_extension.py']
    return dict(version='density-extension-200-v1',question_count=200,new_question_count=150,
        maximum_answer_calls=300,variants=list(VARIANTS),scope_calls=0,query_calls=0,
        config=frozen['config'],inference=frozen['inference'],
        sampling='Original50 plus150 disjoint videos; no original reserve videos; no correctness-based selection',
        comparison='Report old50, new150 and combined200 separately; already-observed full baselines',
        source_results=sources,code_sha256={**frozen['code_sha256'],**{str(p.relative_to(ROOT)):sha256(p) for p in files}},
        method_order='new ordinal even v1 then soft, odd soft then v1; same GPU per question',
        timing='independent raw scan per method; selected-input checks/checkpoints included; model load/warmup/answer persistence excluded')


def source_baseline(qid):
    """只读解析已冻结的全量Top-K来源，供身份/评分核验，不影响选择。"""
    link=read_json(LINKS)['references'][qid]['topk']
    path=ROOT/link['path']
    if sha256(path)!=link['sha256']:raise ProtocolError('Baseline source changed')
    return read_json(path)


def validate(run,record,row,fingerprint):
    """核对已生成结果、精确选帧重放、输入协议与调用账本，不调用模型。"""
    variant=record['variant'];assert variant in VARIANTS and record['protocol_sha256']==fingerprint
    assert record['status']=='completed'
    for key in ('question_id','video_id','question','options','stratum'):assert record[key]==row[key]
    assert sha256(run/record['feature_file'])==record['feature_sha256']
    features=np.load(run/record['feature_file'],allow_pickle=False);s=record['selection']
    base=source_baseline(row['question_id'])
    (verify_v1 if variant=='v1' else verify_soft)(s,features,read_json(run/'protocol.json')['config'],base['selection'])
    assert record['source_video_sha256']==base['source_video_sha256']
    for asset in s['new_assets']:assert sha256(Path(asset['path']))==asset['sha256']
    a=record['answer'];n=len(s['selected_frames'])
    assert n==16 and a['pixel_shape']==[n,3,384,384] and a['visual_tokens']==210*n
    assert a['prefill_tokens']==a['text_input_tokens']-1+210*n
    assert question_text(row['question'],row['options'],s['video']['duration_seconds'],[f['timestamp_seconds'] for f in s['selected_frames']]) in a['prompt']
    assert len(a['generated_token_ids'])<=8 and a['parsed_answer']==parse_answer(a['raw_output'])
    assert record['reference_answer']=='ABCD'[row['answer_index']]
    assert record['correct']==(a['parsed_answer']==record['reference_answer'])
    call=read_json(run/'calls/answer'/f'{row["question_id"]}__{variant}.json')
    assert call['status']=='completed' and call['result']==a and call['protocol_sha256']==fingerprint
    assert call['generation_state']=={'started':True,'returned':True}


def verify_pair(run,qid):
    """同题两个独立扫描必须得到相同候选与区域；差异只允许来自选择规则。"""
    a,b=[read_json(run/'results'/v/f'{qid}.json') for v in VARIANTS]
    for key in ('video','candidates','initial_count','plan','refinement','new_candidate_count'):
        assert a['selection'][key]==b['selection'][key],(qid,key)
    score_error=float(np.max(np.abs(np.asarray(a['selection']['initial_scores'])-np.asarray(b['selection']['initial_scores']))))
    fa,fb=[np.load(run/r['feature_file'],allow_pickle=False) for r in (a,b)]
    assert fa.shape==fb.shape
    feature_error=float(np.max(np.abs(fa-fb)))
    assert score_error<=1e-6 and feature_error<=1e-6
    return dict(question_id=qid,status='passed',score_max_abs_error=score_error,feature_max_abs_error=feature_error)


def run_variant(run,row,variant,gpu,backend,scorer,fingerprint):
    """一次正式问答：原片扫描→选择→检查点→持久生成账本→计分与图板。"""
    import torch
    qid=row['question_id'];key=f'{qid}__{variant}';destination=run/'results'/variant/f'{qid}.json'
    if destination.exists():validate(run,read_json(destination),row,fingerprint);return
    cfg=read_json(run/'protocol.json')['config'];checkpoint=run/'checkpoints'/f'{key}.json'
    ledger=CallLedger(run,key,fingerprint,gpu);before=dict(scorer.counts)
    torch.cuda.synchronize();resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    resumed=checkpoint.exists()
    # 步骤1：首次执行不读取另一方法的候选/特征缓存；恢复身份一致才复用。
    if resumed:
        saved=read_json(checkpoint);assert saved['protocol_sha256']==fingerprint
        assert sha256(run/saved['feature_file'])==saved['feature_sha256']
        selection=saved['selection'];features=np.load(run/saved['feature_file'],allow_pickle=False)
        asset={k:saved[k] for k in ('feature_file','feature_sha256')};timing={k:0. for k in saved['timings']}
    else:
        prepare=prepare_v1 if variant=='v1' else prepare_soft
        selection,features,timing=prepare(ROOT/'data'/row['video_relative_path'],row['question'],scorer,cfg,run/'fine_frames'/key)
        t=time.perf_counter();asset=save_features(run,key+'.npy',features);timing['feature_persist_seconds']=time.perf_counter()-t
    t=time.perf_counter();base=source_baseline(qid)
    (verify_v1 if variant=='v1' else verify_soft)(selection,features,cfg,base['selection'])
    assert len(selection['selected_frames'])==16
    timing['selection_validation_seconds']=time.perf_counter()-t
    t=time.perf_counter()
    if not resumed:durable(checkpoint,dict(protocol_sha256=fingerprint,selection=selection,timings=timing,**asset))
    timing['selection_checkpoint_seconds']=time.perf_counter()-t
    t=time.perf_counter();images=read_exact_frames(selection['video'],selection['selected_frames']);timing['final_decode_seconds']=time.perf_counter()-t
    torch.cuda.synchronize();selection_end=time.perf_counter();peak=torch.cuda.max_memory_allocated();ended=[]
    # 步骤2：生成前落盘，失败占用调用额度，禁止按对错自动重试。
    def answer_once(state):
        result=backend.answer(images,[r['timestamp_seconds'] for r in selection['selected_frames']],selection['video']['duration_seconds'],row['question'],row['options'],generation_state=state)
        ended.append(time.perf_counter());return result
    answer=ledger.invoke('answer',answer_once);returned=time.perf_counter();finish=ended[0] if ended else returned
    qa=finish-selection_end;components=sum(timing.values())
    timing.update(selection_total_seconds=selection_end-start,qa_total_seconds=qa,end_to_end_seconds=finish-start,
        stage_sum_seconds=components+qa,remaining_seconds=finish-start-components-qa,
        selection_core_seconds=sum(v for k,v in timing.items() if k in ('coarse_rgb_seconds','coarse_preprocess_seconds','coarse_forward_seconds','coarse_misc_seconds','region_build_seconds','fine_rgb_seconds','fine_visual_preprocess_seconds','fine_visual_forward_seconds','selection_compete_seconds')))
    result=dict(status='completed',variant=variant,question_id=qid,video_id=row['video_id'],stratum=row['stratum'],question=row['question'],options=row['options'],
        protocol_sha256=fingerprint,source_video_sha256=read_json(run/'preflight.json')['video_hashes'][row['video_id']],
        physical_gpu=gpu,selection=selection,answer=answer,reference_answer='ABCD'[row['answer_index']],correct=answer['parsed_answer']=='ABCD'[row['answer_index']],
        **asset,timings=timing,timing_kind='resumed_not_primary' if resumed or ledger.reused else 'direct_no_application_cache',
        answer_completion_checkpoint_seconds=returned-finish,blip_calls={k:scorer.counts[k]-before[k] for k in before},
        memory=dict(resident_allocated_gib=resident/2**30,selector_incremental_peak_gib=max(0,peak-resident)/2**30,
                    worker_task_peak_allocated_gib=max(peak/2**30,answer['peak_allocated_gib'])))
    # 步骤3：落盘完成结果后再生成展示；恢复不重答已完成调用。
    durable(destination,result);validate(run,result,row,fingerprint)
    render_board(run/'boards'/variant/f'{qid}.png',images,selection['selected_frames'],backend.processor,qid+' '+variant)
    for im in images:im.close()


def worker(gpu,tasks,events,stop,run,fingerprint):
    """每卡长期进程，同题两个方法在同卡交替先后；异常暂停整轮。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    offline_environment();run=Path(run);scorer=None
    with (run/'logs'/f'gpu_{gpu}.log').open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import torch
            from videoqa_runtime.llava_backend import LlavaBackend
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            t=time.perf_counter();backend=LlavaBackend();scorer=DensityScorer(read_json(ROOT/'configs/local_model_inventory.json')['models']['blip']['path'])
            loading=time.perf_counter()-t;t=time.perf_counter();backend.warmup();check=scorer.synthetic_acceptance()
            durable(run/'workers'/f'{gpu}_{os.getpid()}.json',dict(gpu=gpu,pid=os.getpid(),load_seconds=loading,warmup_seconds=time.perf_counter()-t,
                feature_acceptance=check,synthetic_blip_counts=dict(scorer.counts),llava_synthetic_forwards=1,load_report=backend.load_report))
            events.put(dict(type='ready',gpu=gpu))
            while not stop.is_set():
                job=tasks.get()
                if job is None:break
                row=job['row'];order=VARIANTS if job['ordinal']%2==0 else VARIANTS[::-1]
                for variant in order:
                    if stop.is_set():break
                    run_variant(run,row,variant,gpu,backend,scorer,fingerprint)
                if stop.is_set():break
                durable(run/'pairs'/f'{row["question_id"]}.json',verify_pair(run,row['question_id']))
                events.put(dict(type='completed',gpu=gpu,question_id=row['question_id']))
        except BaseException as error:
            stop.set();traceback.print_exc()
            durable(run/'failures'/f'{gpu}_{time.time_ns()}.json',dict(gpu=gpu,error=repr(error),traceback=traceback.format_exc(),utc=utc()))
            events.put(dict(type='failed',gpu=gpu,error=repr(error)))
        finally:
            if scorer is not None:durable(run/'workers'/f'{gpu}_{os.getpid()}_final.json',dict(blip_counts=scorer.counts,utc=utc()))


def check_recovery(run,rows,fingerprint):
    """300次固定调用ID检查；完成结果重放，不确定生成一律暂停。"""
    ids={f'{r["question_id"]}__{v}' for r in rows for v in VARIANTS}
    calls=list((run/'calls/answer').glob('*.json'));assert len(calls)<=300
    for p in calls:
        call=read_json(p);assert p.stem in ids and call['status']=='completed' and call['protocol_sha256']==fingerprint
    complete=[]
    for row in rows:
        q=row['question_id'];found=0
        for variant in VARIANTS:
            p=run/'results'/variant/f'{q}.json'
            if p.exists():validate(run,read_json(p),row,fingerprint);found+=1
        if found==2:
            pair=verify_pair(run,q);path=run/'pairs'/f'{q}.json'
            if path.exists():assert read_json(path)==pair
            else:durable(path,pair)
            complete.append(q)
    assert all(p.stem in {r['question_id'] for r in rows} for v in VARIANTS for p in (run/'results'/v).glob('*.json'))
    return complete


def execute(run_id, requested_gpus, resume=False, prepare_only=False):
    """后台批量接口：冻结指纹、排他锁、前5对验收后继续145对，完成条目不重答。"""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',run_id):
        raise ProtocolError('Invalid run ID')
    run=ROOT/'outputs/methods'/run_id;run.mkdir(parents=True,exist_ok=resume)
    with (run/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ('results/v1','results/soft','pairs','calls/answer','checkpoints','features','fine_frames','workers','failures','logs','persistence','code_snapshot'):
            (run/name).mkdir(parents=True,exist_ok=True)
        spec=protocol()
        if resume:
            if read_json(run/'protocol.json')!=spec:
                raise ProtocolError('Code/config/history fingerprint changed')
        else:
            durable(run/'protocol.json',spec)
            for name in spec['code_sha256']:
                destination=run/'code_snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
        fingerprint=sha256(run/'protocol.json');manifest=read_json(MANIFEST)
        wanted=set(manifest['new_question_ids']);rows=[r for r in manifest['rows'] if r['question_id'] in wanted]
        ordinals={r['question_id']:i for i,r in enumerate(rows)}
        if len(rows)!=150 or len({r['question_id'] for r in rows})!=150:
            raise ProtocolError('Expected frozen new150 questions')
        completed=len(check_recovery(run,rows,fingerprint))
        if completed==150:
            print('Verified 150 completed pairs (300 answers); no models loaded and no generation repeated.',flush=True);return run
        if prepare_only:
            durable(run/'status.json',dict(status='prepared_not_started',new_questions=150,maximum_answer_calls=300,utc=utc()))
            return run
        # GPU占用时拒绝启动，不抢占其他服务，也不开始任何生成。
        available_gpus(list(dict.fromkeys(requested_gpus))[:8])
        start=time.perf_counter();phase='preflight';active={};processes={};stop=None;queues={};failure=None
        durable(run/'pid.json',dict(pid=os.getpid(),session_id=os.getsid(0),utc=utc(),run_id=run_id))
        def publish():
            """进度落盘供外部观察；观察超时不被当作任务失败。"""
            durable(run/'status.json',dict(status=phase,completed=len(list((run/'pairs').glob('*.json'))),active=active,pid=os.getpid(),elapsed_seconds=time.perf_counter()-start,utc=utc()))
        try:
            publish();verification=preflight(rows,dict(question_count=150,unique_video_count=150,manifest_path=str(MANIFEST.relative_to(ROOT)),manifest_sha256=sha256(MANIFEST),asset_audit_path='data/manifests/video_mme_full_asset_audit.json'));durable(run/'preflight.json',verification)
            gpus,states=available_gpus(list(dict.fromkeys(requested_gpus))[:8]);durable(run/'gpu_allocation.json',dict(selected=gpus,states=states))
            os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            ctx=mp.get_context('spawn');events=ctx.Queue();stop=ctx.Event();queues={g:ctx.Queue() for g in gpus}
            processes={g:ctx.Process(target=worker,args=(g,queues[g],events,stop,str(run),fingerprint)) for g in gpus}
            idle=set()
            def event():
                """轮询真实进程状态；超时刷新同一任务，不重启。"""
                while True:
                    publish()
                    try:
                        item=events.get(timeout=10)
                    except queue.Empty:
                        if any(p.exitcode is not None for p in processes.values()):
                            raise RuntimeError('Worker exited unexpectedly; inspect persisted ledger')
                        continue
                    if item['type']=='failed':
                        raise RuntimeError(str(item))
                    return item
            def schedule(batch):
                """共享待办按顺序分给空闲GPU，已完成结果跳过。"""
                pending=[r for r in batch if not (run/'pairs'/f'{r["question_id"]}.json').exists()]
                while pending or active:
                    for gpu in sorted(idle):
                        if not pending:
                            break
                        row=pending.pop(0);idle.remove(gpu);active[gpu]=row['question_id'];queues[gpu].put(dict(row=row,ordinal=ordinals[row['question_id']]))
                    item=event()
                    if item['type']=='completed':
                        active.pop(item['gpu']);idle.add(item['gpu'])
            phase='model_startup'
            for process in processes.values():
                process.start()
            while len(idle)<len(gpus):
                idle.add(event()['gpu'])
            phase='pilot';pilot_start=time.perf_counter();schedule(rows[:5])
            for row in rows[:5]:
                for variant in VARIANTS:
                    validate(run,read_json(run/'results'/variant/f'{row["question_id"]}.json'),row,fingerprint)
            durable(run/'pilot_acceptance.json',dict(status='passed',question_ids=[r['question_id'] for r in rows[:5]],
                    wall_seconds=time.perf_counter()-pilot_start,gate='identity/frame/token/timing; independent of accuracy'))
            phase='full'
            durations={v['video_id']:v['duration_seconds'] for v in read_json(ROOT/'data/manifests/video_mme_full_asset_audit.json')['videos']}
            schedule(sorted(rows[5:],key=lambda r:-durations[r['video_id']]))
            if len(check_recovery(run,rows,fingerprint))!=150:
                raise ProtocolError('Final set incomplete')
            phase='completed'
        except BaseException as error:
            phase='paused';failure=repr(error);traceback.print_exc()
        finally:
            if stop is not None:
                stop.set()
            for task_queue in queues.values():
                task_queue.put(None)
            while any(p.is_alive() for p in processes.values()):
                for process in processes.values():
                    if process.pid is not None:
                        process.join(timeout=1)
            previous=read_json(run/'execution.json').get('attempts',[]) if (run/'execution.json').exists() else []
            previous.append(dict(status=phase,error=failure,wall_seconds=time.perf_counter()-start,utc=utc()))
            durable(run/'execution.json',dict(status=phase,attempts=previous,total_active_wall_seconds=sum(a['wall_seconds'] for a in previous)))
            publish()
        if failure:
            raise RuntimeError(failure)
        return run
