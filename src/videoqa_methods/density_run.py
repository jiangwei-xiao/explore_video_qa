"""密度框架独立执行器：一次正式50题、持久调用账本、后台可恢复。"""
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
from videoqa_runtime.common import ROOT, DEVELOPMENT_SHA256, SOURCE_COMMIT, read_json, sha256, offline_environment
from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_runtime.baseline_run import available_gpus, preflight
from videoqa_runtime.protocol import question_text, parse_answer
from .followups import CallLedger, durable, utc, save_features
from .density_pipeline import DensityScorer, prepare_density
from .density_selection import build_regions, select_joint
from .region_context import read_exact_frames

BASELINE = ROOT / 'outputs/baselines/videomme50_uniform_topk_20260914_r1'
CONFIG = ROOT / 'configs/retrieval_density_v1.json'
MANIFEST = ROOT / 'data/manifests/video_mme_development.json'
HISTORY_PATHS = dict(TopK=BASELINE/'results/topk', Uniform=BASELINE/'results/uniform',
                    B=ROOT/'outputs/methods/method_v1_videomme50_20260914_r1/results/B',
                    C_local=ROOT/'outputs/methods/followups_clocal_query_videomme50_20260915_r1/results/E',
                    R=ROOT/'outputs/methods/region_boundary_v1_videomme50_20260917_r1/results')


def execution_protocol():
    """接口：冻结运行代码、配置、清单、依赖、模型身份及历史结果哈希。"""
    if sha256(MANIFEST) != DEVELOPMENT_SHA256:
        raise ProtocolError('Frozen development50 manifest changed')
    files = sorted((ROOT/'src/videoqa_runtime').glob('*.py')) + sorted((ROOT/'src/videoqa_methods').glob('*.py'))
    files += [ROOT/'scripts/run_retrieval_density.py', CONFIG, ROOT/'configs/local_model_inventory.json',
              ROOT/'configs/runtime_environment.json',ROOT/'configs/llava_source_manifest.json',
              ROOT/'requirements-runtime.lock.txt',ROOT/'requirements-method.lock.txt',
              ROOT/'docs/research/检索密度引导的条件补查与选帧框架.md']
    history = [p for directory in HISTORY_PATHS.values() for p in sorted(directory.glob('*.json'))]
    return dict(version='retrieval-density-v1',config=read_json(CONFIG),question_count=50,
                dataset_sha256=sha256(MANIFEST),llava_source_commit=SOURCE_COMMIT,
                maximum_answer_calls=50,scope_calls=0,query_calls=0,automatic_generation_retries=0,
                inference=dict(dtype='bfloat16',attention='sdpa',template='qwen_1_5',max_new_tokens=8,
                               seed=2027,visual_tokens_per_frame=210,no_subtitles=True,no_audio=True),
                execution=dict(pilot_count=5,cpu_threads=2,decode_threads=2,maximum_gpus=8,
                               order_after_pilot='duration descending',asset_mode='independent raw scan',
                               synthetic_warmup='one LLaVA forward plus two BLIP ITM and one visual forward per worker'),
                timing_scope='models ready; raw processing to answer parsed; selection assets/checkpoints included; '
                             'answer-completion ledger/result persistence excluded and separately recorded; OS cache uncontrolled',
                code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in files},
                history_sha256={str(p.relative_to(ROOT)):sha256(p) for p in history})


def verify_selection(selection, features, cfg, historical=None):
    """独立重放区域与选择；正式QA前核对原候选、保护、补查网格和最终预算。"""
    # 步骤1：核对原片扫描是否遵循历史相同输入协议，不读取历史答案用于选择。
    n = selection['initial_count']; rows = selection['candidates']; video=selection['video']
    if historical is not None:
        if rows[:n] != historical['candidates'] or not np.allclose(selection['initial_scores'],historical['scores'],atol=1e-6,rtol=0):
            raise ProtocolError('Independent coarse candidates/scores differ from baseline')
    expected = build_regions(rows[:n],selection['initial_scores'],video,cfg)
    if expected != selection['plan']:
        raise ProtocolError('Region plan cannot replay')
    # 步骤2：补查仅发生在获准区域，逐新增源帧能回溯到一个目标映射。
    new_ids=[]
    from fractions import Fraction
    from .density_selection import refinement_targets, source_time
    for region, report in zip(expected['regions'],selection['refinement']):
        if report['region_id']!=region['region_id'] or report['enabled']!=region['refine']:
            raise ProtocolError('Refinement region mismatch')
        targets=refinement_targets(region,video,cfg) if region['refine'] else []
        if report['targets']!=list(map(str,targets)) or len(report['mappings'])!=len(targets):
            raise ProtocolError('Refinement targets/mappings incomplete')
        for mapping,target in zip(report['mappings'],targets):
            if Fraction(mapping['target_fraction']) != target:
                raise ProtocolError('Target mapping order changed')
            if mapping['status']=='added':
                i=mapping['candidate_index'];new_ids.append(i)
                if i<n or rows[i]['source_pts']!=mapping['source_pts'] or rows[i]['source_frame_index']!=mapping['source_frame_index']:
                    raise ProtocolError('New source identity mismatch')
                if not Fraction(region['left_fraction'])<=source_time(rows[i],video)<=Fraction(region['right_fraction']):
                    raise ProtocolError('New source outside region')
                if source_time(rows[i],video)<target:
                    raise ProtocolError('Fine source earlier than target')
    if len(selection['refinement'])!=len(expected['regions']) or sorted(new_ids)!=list(range(n,len(rows))):
        raise ProtocolError('Missing or duplicate new-source lineage')
    if selection['new_candidate_count']!=len(rows)-n or selection['new_frames_have_itm_scores'] is not False:
        raise ProtocolError('New candidate count/scoring mismatch')
    if [a['candidate_index'] for a in selection['new_assets']]!=list(range(n,len(rows))):
        raise ProtocolError('New asset inventory mismatch')
    # 步骤3：利用保存特征重放全部选择；离散输入必须完全一致。
    indices, trace=select_joint(rows,features,expected,video,cfg)
    if indices!=selection['selected_indices'] or trace!=selection['selection_trace']:
        raise ProtocolError('Selection/trace cannot replay')
    if selection['selected_frames']!=[rows[i] for i in indices]:
        raise ProtocolError('Final selected metadata mismatch')


def validate_result(run, record, row, fingerprint, verify_assets=False):
    """接口：验证结果身份、特征重放、生成协议和计分；不会调用模型。"""
    if record['status']!='completed' or record['method']!='density_v1' or record['protocol_sha256']!=fingerprint:
        raise ProtocolError('Result identity changed')
    if any(record[k]!=row[k] for k in ('question_id','video_id','question','options','stratum')):
        raise ProtocolError('Question/options changed')
    path=run/record['feature_file']
    if sha256(path)!=record['feature_sha256']:
        raise ProtocolError('Feature bytes changed')
    reference=read_json(BASELINE/'results/topk'/f'{row["question_id"]}.json')
    if record['source_video_sha256']!=reference['source_video_sha256']:
        raise ProtocolError('Video identity differs')
    selection=record['selection']; features=np.load(path,allow_pickle=False)
    verify_selection(selection,features,read_json(run/'protocol.json')['config'],reference['selection'])
    if verify_assets:
        for asset in selection['new_assets']:
            if sha256(Path(asset['path']))!=asset['sha256']:
                raise ProtocolError('Fine image asset changed')
    a=record['answer'];count=len(selection['selected_frames'])
    if a['pixel_shape']!=[count,3,384,384] or a['visual_tokens']!=210*count or a['prefill_tokens']!=a['text_input_tokens']-1+210*count:
        raise ProtocolError('Visual input protocol mismatch')
    message=question_text(row['question'],row['options'],selection['video']['duration_seconds'],[r['timestamp_seconds'] for r in selection['selected_frames']])
    if message not in a['prompt'] or len(a['generated_token_ids'])>8 or a['parsed_answer']!=parse_answer(a['raw_output']):
        raise ProtocolError('Prompt/generation/parser mismatch')
    if record['reference_answer']!='ABCD'[row['answer_index']] or record['correct']!=(a['parsed_answer']==record['reference_answer']):
        raise ProtocolError('Scoring mismatch')
    call=read_json(run/'calls/answer'/f'{row["question_id"]}.json')
    if call['status']!='completed' or call['protocol_sha256']!=fingerprint or call['result']!=a:
        raise ProtocolError('Answer ledger/result mismatch')
    if call.get('generation_state')!={'started':True,'returned':True}:
        raise ProtocolError('Generation completion not confirmed')


def run_question(run, row, gpu, backend, scorer, fingerprint):
    """一次完整任务；持久检查点与答案账本保证恢复不重复已完成生成。"""
    import torch
    qid=row['question_id'];destination=run/'results'/f'{qid}.json'
    if destination.exists():
        validate_result(run,read_json(destination),row,fingerprint);return
    cfg=read_json(run/'protocol.json')['config'];checkpoint=run/'checkpoints'/f'{qid}.json'
    ledger=CallLedger(run,qid,fingerprint,gpu);counter_before=dict(scorer.counts)
    torch.cuda.synchronize();resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    resumed=checkpoint.exists()
    # 步骤1：只允许同指纹输入检查点复用，正式首次运行始终从原片开始。
    if resumed:
        saved=read_json(checkpoint)
        if saved['protocol_sha256']!=fingerprint or sha256(run/saved['feature_file'])!=saved['feature_sha256']:
            raise ProtocolError('Selection checkpoint changed')
        selection=saved['selection'];features=np.load(run/saved['feature_file'],allow_pickle=False)
        feature_record={k:saved[k] for k in ('feature_file','feature_sha256')}
        timing={k:0. for k in saved['selection_timings']}
    else:
        selection,features,timing=prepare_density(ROOT/'data'/row['video_relative_path'],row['question'],scorer,cfg,run/'fine_frames'/qid)
        t=time.perf_counter();feature_record=save_features(run,f'{qid}.npy',features)
        timing['feature_persist_seconds']=time.perf_counter()-t
    t=time.perf_counter();historical=read_json(BASELINE/'results/topk'/f'{qid}.json')['selection']
    verify_selection(selection,features,cfg,historical)
    timing['selection_validation_seconds']=time.perf_counter()-t
    t=time.perf_counter()
    if not resumed:
        durable(checkpoint,dict(protocol_sha256=fingerprint,selection=selection,selection_timings=dict(timing),**feature_record))
    timing['selection_checkpoint_seconds']=time.perf_counter()-t
    t=time.perf_counter();images=read_exact_frames(selection['video'],selection['selected_frames'])
    timing['final_decode_seconds']=time.perf_counter()-t
    torch.cuda.synchronize();selector_end=time.perf_counter();selector_peak=torch.cuda.max_memory_allocated()
    qa_start=time.perf_counter();ended=[];call_start_checkpoint=[]
    # 步骤2：先持久化调用开始，再使用原提示词回答一次；标准答案仅在生成后评分。
    def answer_once(state):
        """实际解析结束时记录E2E边界，答案落盘不伪装成生成耗时。"""
        call_start_checkpoint.append(ledger.checkpoint_seconds)
        answer=backend.answer(images,[r['timestamp_seconds'] for r in selection['selected_frames']],selection['video']['duration_seconds'],row['question'],row['options'],generation_state=state)
        ended.append(time.perf_counter());return answer
    answer=ledger.invoke('answer',answer_once);returned=time.perf_counter();finish=ended[0] if ended else returned
    # 步骤3：记录直接E2E和不重叠阶段之和；恢复性能与首次性能分开。
    components=sum(timing.values());qa_seconds=finish-qa_start
    timing.update(selection_total_seconds=selector_end-start,qa_total_seconds=qa_seconds,end_to_end_seconds=finish-start,
                  stage_sum_seconds=components+qa_seconds,remaining_seconds=finish-start-components-qa_seconds,
                  selection_core_seconds=sum(v for k,v in timing.items() if k in ('coarse_rgb_seconds','coarse_preprocess_seconds','coarse_forward_seconds','coarse_misc_seconds','region_build_seconds','fine_rgb_seconds','fine_visual_preprocess_seconds','fine_visual_forward_seconds','selection_compete_seconds')),
                  generation_seconds=answer['generation_seconds'])
    result=dict(status='completed',method='density_v1',question_id=qid,video_id=row['video_id'],stratum=row['stratum'],question=row['question'],options=row['options'],
                source_video_sha256=read_json(run/'preflight.json')['video_hashes'][row['video_id']],protocol_sha256=fingerprint,
                physical_gpu=gpu,selection=selection,answer=answer,reference_answer='ABCD'[row['answer_index']],
                correct=answer['parsed_answer']=='ABCD'[row['answer_index']],**feature_record,timings=timing,
                timing_kind='resumed_not_primary' if resumed or ledger.reused else 'direct_no_application_cache',
                blip_calls={k:scorer.counts[k]-counter_before[k] for k in scorer.counts},
                call_start_checkpoint_seconds=call_start_checkpoint[0] if call_start_checkpoint else 0.,
                answer_completion_checkpoint_seconds=returned-finish,memory=dict(resident_allocated_gib=resident/2**30,
                    selector_incremental_peak_gib=max(0,selector_peak-resident)/2**30,
                    qa_incremental_peak_gib=max(0,answer['peak_allocated_gib']-resident/2**30),
                    worker_task_peak_allocated_gib=max(selector_peak/2**30,answer['peak_allocated_gib'])))
    t=time.perf_counter();durable(destination,result)
    durable(run/'persistence'/f'{qid}.json',dict(result_save_seconds=time.perf_counter()-t,utc=utc()))
    validate_result(run,result,row,fingerprint)
    from .density_report import render_board
    t=time.perf_counter();render_board(run/'boards'/f'{qid}.png',images,selection['selected_frames'],backend.processor,qid)
    durable(run/'persistence'/f'{qid}_board.json',dict(board_seconds=time.perf_counter()-t,utc=utc()))
    for image in images:
        image.close()


def worker(gpu, tasks, events, stop, run, fingerprint):
    """一卡常驻两个模型，先合成验收；任何异常暂停，不自动再次生成。"""
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
                row=tasks.get()
                if row is None:
                    break
                run_question(run,row,gpu,backend,scorer,fingerprint)
                events.put(dict(type='completed',gpu=gpu,question_id=row['question_id']))
        except BaseException as error:
            stop.set();traceback.print_exc()
            durable(run/'failures'/f'{gpu}_{time.time_ns()}.json',dict(gpu=gpu,error=repr(error),traceback=traceback.format_exc(),utc=utc(),
                    blip_counts=None if scorer is None else scorer.counts))
            events.put(dict(type='failed',gpu=gpu,error=repr(error)))
        finally:
            if scorer is not None:
                durable(run/'workers'/f'{gpu}_{os.getpid()}_final.json',dict(gpu=gpu,pid=os.getpid(),blip_counts=scorer.counts,utc=utc()))


def recovery_check(run, rows, fingerprint):
    """固定50题与调用ID恢复检查；失败或不确定生成禁止自动重试。"""
    ids={r['question_id'] for r in rows};calls=list((run/'calls/answer').glob('*.json'))
    if len(calls)>50 or any(p.stem not in ids for p in calls):
        raise ProtocolError('Call budget/scope exceeded')
    for path in calls:
        call=read_json(path)
        if call['status']!='completed' or call['protocol_sha256']!=fingerprint or call['question_id']!=path.stem:
            raise ProtocolError('Uncertain/failed answer call; automatic repeat forbidden')
    results=list((run/'results').glob('*.json'))
    if any(p.stem not in ids for p in results):
        raise ProtocolError('Out-of-scope result')
    for row in rows:
        path=run/'results'/f'{row["question_id"]}.json'
        if path.exists():
            validate_result(run,read_json(path),row,fingerprint)
    return len(results)


def execute(run_id, requested_gpus, resume=False):
    """后台批量接口：冻结指纹、排他锁、前5题验收后继续45题，完成条目不重答。"""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',run_id):
        raise ProtocolError('Invalid run ID')
    run=ROOT/'outputs/methods'/run_id;run.mkdir(parents=True,exist_ok=resume)
    with (run/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ('results','calls/answer','checkpoints','features','fine_frames','workers','failures','logs','persistence','code_snapshot'):
            (run/name).mkdir(parents=True,exist_ok=True)
        spec=execution_protocol()
        if resume:
            if read_json(run/'protocol.json')!=spec:
                raise ProtocolError('Code/config/history fingerprint changed')
        else:
            durable(run/'protocol.json',spec)
            for name in spec['code_sha256']:
                destination=run/'code_snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
        fingerprint=sha256(run/'protocol.json');rows=read_json(MANIFEST)['rows']
        if len(rows)!=50 or len({r['question_id'] for r in rows})!=50:
            raise ProtocolError('Expected frozen 50 questions')
        completed=recovery_check(run,rows,fingerprint)
        if completed==50:
            print('Verified 50 completed results; no models loaded and no generation repeated.',flush=True);return run
        start=time.perf_counter();phase='preflight';active={};processes={};stop=None;queues={};failure=None
        durable(run/'pid.json',dict(pid=os.getpid(),session_id=os.getsid(0),utc=utc(),run_id=run_id))
        def publish():
            """进度落盘供外部观察；观察超时不被当作任务失败。"""
            durable(run/'status.json',dict(status=phase,completed=len(list((run/'results').glob('*.json'))),active=active,pid=os.getpid(),elapsed_seconds=time.perf_counter()-start,utc=utc()))
        try:
            publish();verification=preflight(rows);durable(run/'preflight.json',verification)
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
                pending=[r for r in batch if not (run/'results'/f'{r["question_id"]}.json').exists()]
                while pending or active:
                    for gpu in sorted(idle):
                        if not pending:
                            break
                        row=pending.pop(0);idle.remove(gpu);active[gpu]=row['question_id'];queues[gpu].put(row)
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
                validate_result(run,read_json(run/'results'/f'{row["question_id"]}.json'),row,fingerprint,verify_assets=True)
            durable(run/'pilot_acceptance.json',dict(status='passed',question_ids=[r['question_id'] for r in rows[:5]],
                    wall_seconds=time.perf_counter()-pilot_start,gate='identity/frame/token/timing; independent of accuracy'))
            phase='full'
            durations={v['video_id']:v['duration_seconds'] for v in read_json(ROOT/'data/manifests/video_mme_asset_validation.json')['videos']}
            schedule(sorted(rows[5:],key=lambda r:-durations[r['video_id']]))
            if recovery_check(run,rows,fingerprint)!=50:
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
