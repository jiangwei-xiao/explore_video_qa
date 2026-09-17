"""R的离线门槛与独立50题执行；保持旧方法和公共运行层原样。"""
import contextlib
import fcntl
import html
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import shutil
import time
import traceback
import numpy as np
from videoqa_runtime.common import ROOT,read_json,write_json,sha256,offline_environment
from videoqa_runtime.baseline_selection import ProtocolError,BlipItmScorer,iter_candidate_frames,select_topk
from videoqa_runtime.baseline_run import available_gpus,preflight
from videoqa_methods.followups import CallLedger,durable,utc
from .region_context import CONFIG,read_neighborhoods_and_select,read_exact_frames

BASELINE=ROOT/'outputs/baselines/videomme50_uniform_topk_20260914_r1'
AUDIT=ROOT/'outputs/analysis/evidence_videomme50_20260915_r1'
KEY_CASES=['103-1','306-2','395-2','383-2','260-2','170-2','443-1','004-3','544-1']


def execution_protocol():
    """冻结执行文件、配置与旧结果/事实目录身份，拒绝跨版本恢复，不加载模型。"""
    files=list((ROOT/'src/videoqa_runtime').glob('*.py'))
    files += [ROOT/'src/videoqa_methods'/n for n in ('region_context.py','region_run.py','refinement.py','followups.py')]
    files += [ROOT/'scripts/run_region_context.py',ROOT/'requirements-runtime.lock.txt',ROOT/'configs/local_model_inventory.json']
    sources=list((BASELINE/'results/topk').glob('*.json'))+list((AUDIT/'full_review/witnesses_r6').glob('*.json'))
    sources += list((AUDIT/'annotations').glob('*.json'))+[ROOT/'data/manifests/video_mme_development.json']
    return dict(version='region-boundary-v1',config=CONFIG,maximum_answer_calls=50,scope_calls=0,query_calls=0,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sources},
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in files},
        review_time_limit_seconds=1800,key_cases=KEY_CASES,formal_requires_passed_gate=True)


def verify_selection(result,historical=None):
    """检查候选、保护、替换与16帧上限；已有分数只对应初始候选，新帧没有伪造分数。"""
    rows=result['candidates'];n=result['initial_count'];selected=result['selected_indices'];plan=result['plan']
    if not (len(selected)==min(16,n) and len(set(rows[i]['source_pts'] for i in selected))==len(selected)):
        raise ProtocolError('R final unique budget mismatch')
    if result['selected_frames']!=[rows[i] for i in selected] or [rows[i]['source_pts'] for i in selected]!=sorted(rows[i]['source_pts'] for i in selected):
        raise ProtocolError('R selected rows/order mismatch')
    if not set(plan['protected_indices'])<=set(selected) or len(result['replacements'])>4 or len(rows)-n>24:
        raise ProtocolError('R protection/cap failure')
    if len(result['initial_scores'])!=n:raise ProtocolError('Fine candidates must not have BLIP scores')
    if historical is not None:
        if rows[:n]!=historical['candidates'] or not np.allclose(result['initial_scores'],historical['scores'],atol=1e-6,rtol=0):
            raise ProtocolError('Independent initial pool differs from frozen Top-K')
    return True


def render_offline(run,qid,result,historical,processor):
    """输出实际384输入及变化图板；只做图像预处理，不运行视觉编码或问答。"""
    from PIL import Image,ImageDraw
    directory=run/'review'/qid;directory.mkdir(parents=True,exist_ok=True)
    before=historical['selected_frames'];after=result['selected_frames']
    union={r['source_pts']:r for r in before+after};ordered=sorted(union.values(),key=lambda r:r['source_pts'])
    images=read_exact_frames(result['video'],ordered);views={}
    for row,im in zip(ordered,images):
        pts=row['source_pts'];pixel=processor.preprocess(im,return_tensors='np')['pixel_values'][0].transpose(1,2,0)
        rgb=pixel*np.asarray(processor.image_std)+np.asarray(processor.image_mean)
        view=Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype('uint8'));views[pts]=view
        view.save(directory/f'{pts}.input.png');im.save(directory/f'{pts}.original.jpg',quality=95)
    for label,rows in [('TopK',before),('R',after)]:
        canvas=Image.new('RGB',(1536,40+4*410),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,8),f'{qid} {label}: actual ordered input',fill='black')
        for i,row in enumerate(rows):
            x=i%4*384;y=i//4*410+40;canvas.paste(views[row['source_pts']],(x,y))
            draw.text((x+3,y+386),f'#{row["candidate_index"]} {row["timestamp_seconds"]:.5f}s PTS {row["source_pts"]}',fill='black')
        canvas.save(directory/f'{label}.png')
    swaps=result['replacements']
    if swaps:
        canvas=Image.new('RGB',(768,40+len(swaps)*410),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,8),f'{qid} LEFT removed / RIGHT added',fill='black')
        for j,swap in enumerate(swaps):
            for k,key in enumerate(('removed_index','added_index')):
                row=result['candidates'][swap[key]];canvas.paste(views[row['source_pts']],(k*384,40+j*410))
                draw.text((k*384+3,426+j*410),f'#{row["candidate_index"]} {row["timestamp_seconds"]:.5f}s PTS {row["source_pts"]}',fill='black')
        canvas.save(directory/'changes.png')


def offline(run,rows,fingerprint):
    """全50题离线检查，缓存只用于此阶段；结果及可能事实损失全部保存，不自动放行问答。"""
    from transformers import SiglipImageProcessor
    processor=SiglipImageProcessor.from_pretrained(read_json(ROOT/'configs/local_model_inventory.json')['models']['vision']['path'],local_files_only=True)
    issues=[];summaries=[];start=time.perf_counter()
    for ordinal,row in enumerate(rows):
        qid=row['question_id'];destination=run/'offline'/f'{qid}.json'
        historical=read_json(BASELINE/'results/topk'/f'{qid}.json');selection=historical['selection']
        if sha256(Path(selection['video']['path']))!=historical['source_video_sha256']:
            raise ProtocolError('Source video differs from frozen identity')
        if destination.exists():
            record=read_json(destination)
            if record['protocol_sha256']!=fingerprint:raise ProtocolError('Offline fingerprint changed')
            result=record['selection']
        else:
            pool={k:selection[k] for k in ('video','candidates','scores')}
            t=time.perf_counter();result,times=read_neighborhoods_and_select(pool)
            record=dict(question_id=qid,protocol_sha256=fingerprint,source_result_sha256=sha256(BASELINE/'results/topk'/f'{qid}.json'),
                        source_video_sha256=historical['source_video_sha256'],selection=result,timings=times,
                        offline_seconds=time.perf_counter()-t,application_cache_hits=1,qa_calls=0,blip_calls=0)
            durable(destination,record)
        verify_selection(result,selection)
        if not (run/'review'/qid/'R.png').exists():render_offline(run,qid,result,selection,processor)
        old_pts={r['source_pts'] for r in selection['selected_frames']};new_pts={r['source_pts'] for r in result['selected_frames']}
        facts=read_json(AUDIT/'full_review/witnesses_r6'/f'{qid}.json')['facts']
        affected=[]
        for fact in facts:
            witnesses={r['source_pts'] for r in fact['witnesses']}
            if fact['role'] in ('required','partial') and witnesses&old_pts and not witnesses&new_pts:
                issue=dict(question_id=qid,kind='listed_fact_witness_lost',fact_id=fact['id'],role=fact['role'],
                           description=fact['description'],old_witness_pts=sorted(witnesses&old_pts),topk_correct=historical['correct'])
                issues.append(issue);affected.append(fact['id'])
        annotation=read_json(AUDIT/'annotations'/f'{qid}.json')
        for pts in old_pts-new_pts:
            label=annotation['frames'][str(pts)]
            mapped=any(pts in {w['source_pts'] for w in f['witnesses']} for f in facts)
            if historical['correct'] and label['label']=='direct' and not mapped:
                issues.append(dict(question_id=qid,kind='uncatalogued_direct_frame_removed',source_pts=pts,reason=label['reason'],topk_correct=True))
        summaries.append(dict(question_id=qid,topk_correct=historical['correct'],potential_slots=len(result['plan']['potential_removals']),
                             replacements=len(result['replacements']),new_candidates=result['new_candidate_count'],lost_listed_facts=affected,
                             removed_pts=sorted(old_pts-new_pts),added_pts=sorted(new_pts-old_pts)))
        durable(run/'status.json',dict(status='offline',completed=ordinal+1,total=50,qa_calls=0,utc=utc()))
        print('offline',ordinal+1,qid,'swaps',len(result['replacements']),flush=True)
    hashes={r['question_id']:sha256(run/'offline'/f'{r["question_id"]}.json') for r in rows}
    durable(run/'offline_manifest.json',dict(protocol_sha256=fingerprint,result_sha256=hashes))
    durable(run/'offline_summary.json',dict(questions=summaries,issues=issues,qa_calls=0,blip_calls=0,wall_seconds=time.perf_counter()-start))
    blocks=['<!doctype html><meta charset="utf-8"><style>body{font:16px/1.7 sans-serif;max-width:1100px;margin:auto}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>R：离线变化review</h1><p>仅事后AI初审，不是用户金标准；新帧无BLIP评分，旧事实目录未覆盖新帧不等于事实丢失。</p>']
    for row,s in zip(rows,summaries):
        qid=row['question_id'];blocks.append(f'<details id="{qid}"><summary>{qid}：替换{s["replacements"]}；原Top-K正确={s["topk_correct"]}；目录疑点={s["lost_listed_facts"]}</summary><pre>{html.escape(row["question"])}</pre>')
        if s['replacements']:blocks.append(f'<a href="review/{qid}/changes.png"><img loading="lazy" src="review/{qid}/changes.png"></a>')
        blocks.append(f'<p><a href="review/{qid}/TopK.png">原Top-K</a> · <a href="review/{qid}/R.png">R完整输入</a> · <a href="offline/{qid}.json">精确计划/距离/PTS</a></p></details>')
    (run/'index.html').write_text(''.join(blocks))
    durable(run/'status.json',dict(status='awaiting_evidence_gate',completed=50,qa_calls=0,utc=utc()))


def check_gate(run,fingerprint):
    """只有可追溯的已通过门槛才能启动正式问答；未决/失败均停止且不产生调用。"""
    gate=read_json(run/'gate.json');manifest=read_json(run/'offline_manifest.json')
    if gate.get('status')!='passed' or gate.get('protocol_sha256')!=fingerprint or gate.get('offline_manifest_sha256')!=sha256(run/'offline_manifest.json'):
        raise ProtocolError('Evidence gate not passed or identity changed')
    if not gate.get('visible_evidence_added') or gate.get('confirmed_critical_losses') or gate.get('unresolved_critical_issues'):
        raise ProtocolError('Evidence gate conditions not met')
    if gate.get('review_seconds',1801)>1800:raise ProtocolError('Review exceeded fixed timebox')
    if len(manifest['result_sha256'])!=50:raise ProtocolError('Offline set incomplete')
    for qid,digest in manifest['result_sha256'].items():
        if sha256(run/'offline'/f'{qid}.json')!=digest:raise ProtocolError('Offline evidence changed')
    return gate


def scan_original(path,question,scorer):
    """正式阶段独立全片扫描评分，最多16张RGB驻留；不读取离线候选或评分缓存。"""
    times=dict(candidate_decode_seconds=0.,rgb_seconds=0.,blip_preprocess_seconds=0.,blip_forward_seconds=0.)
    t=time.perf_counter();encoded=scorer.tokenize(question);times['blip_preprocess_seconds']+=time.perf_counter()-t
    candidates=[];scores=[];batch=[];video={};iterator=iter_candidate_frames(path,video)
    def flush():
        """执行当前评分批次，记录同步计时后释放RGB。"""
        values,pre,forward=scorer.score_batch(encoded,batch);scores.extend(values)
        times['blip_preprocess_seconds']+=pre;times['blip_forward_seconds']+=forward;batch.clear()
    while True:
        t=time.perf_counter()
        try:row,frame=next(iterator)
        except StopIteration:times['candidate_decode_seconds']+=time.perf_counter()-t;break
        times['candidate_decode_seconds']+=time.perf_counter()-t;candidates.append(row)
        t=time.perf_counter();batch.append(frame.to_image());times['rgb_seconds']+=time.perf_counter()-t
        if len(batch)==16:flush()
    if batch:flush()
    return dict(video=video,candidates=candidates,scores=scores),times


def formal_job(run,row,gpu,backend,scorer,fingerprint):
    """一次R任务：原片扫描到解析直接计时，调用前用既有账本落盘，完成答案不重复生成。"""
    import torch
    qid=row['question_id'];dest=run/'results'/f'{qid}.json'
    if dest.exists():return
    ledger=CallLedger(run,qid,fingerprint,gpu);checkpoint=run/'checkpoints'/f'{qid}.json'
    torch.cuda.synchronize();resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    resumed=checkpoint.exists()
    if resumed:
        saved=read_json(checkpoint)
        if saved['protocol_sha256']!=fingerprint:raise ProtocolError('Checkpoint fingerprint changed')
        result=saved['selection'];times={k:0. for k in saved['timings']}
    else:
        pool,times=scan_original(ROOT/'data'/row['video_relative_path'],row['question'],scorer)
        result,local_times=read_neighborhoods_and_select(pool);times.update(local_times)
        # 必要输入检查点计入E2E；不先执行原Top-K问答或保存另一组结果。
        durable(checkpoint,dict(protocol_sha256=fingerprint,selection=result,timings=times))
    t=time.perf_counter();images=read_exact_frames(result['video'],result['selected_frames']);times['final_decode_seconds']=time.perf_counter()-t
    torch.cuda.synchronize();selected_end=time.perf_counter();selector_peak=torch.cuda.max_memory_allocated();qa_start=time.perf_counter();end=[]
    def answer_once(state):
        """不向问答模型传分类、分数或取证理由；解析完成立即记录墙钟边界。"""
        answer=backend.answer(images,[r['timestamp_seconds'] for r in result['selected_frames']],result['video']['duration_seconds'],row['question'],row['options'],generation_state=state)
        end.append(time.perf_counter());return answer
    answer=ledger.invoke('answer',answer_once);finished=end[0] if end else time.perf_counter()
    times.update(selection_total_seconds=selected_end-start,qa_total_seconds=finished-qa_start,end_to_end_seconds=finished-start,
                 selection_core_seconds=sum(v for k,v in times.items() if k not in ('candidate_decode_seconds','fine_decode_seconds','final_decode_seconds')),
                 generation_seconds=answer['generation_seconds'])
    times['stage_sum_seconds']=times['selection_total_seconds']+times['qa_total_seconds'];times['remaining_seconds']=times['end_to_end_seconds']-times['stage_sum_seconds']
    reference='ABCD'[row['answer_index']]
    record=dict(status='completed',method='R',question_id=qid,video_id=row['video_id'],stratum=row['stratum'],question=row['question'],options=row['options'],
        source_video_sha256=read_json(run/'formal_preflight.json')['video_hashes'][row['video_id']],
        protocol_sha256=fingerprint,physical_gpu=gpu,selection=result,answer=answer,reference_answer=reference,correct=answer['parsed_answer']==reference,
        timings=times,timing_kind='resumed_not_primary' if resumed or ledger.reused else 'direct_no_application_cache',
        memory=dict(resident_allocated_gib=resident/2**30,selector_incremental_peak_gib=(selector_peak-resident)/2**30,
                    worker_peak_allocated_gib=max(selector_peak/2**30,answer['peak_allocated_gib'])),call_checkpoint_seconds=ledger.checkpoint_seconds)
    durable(dest,record)
    validate_formal(run,record,row,fingerprint)


def validate_formal(run,record,row,fingerprint):
    """按独立正式输入与离线计划核对；发现不一致保留输出并暂停，禁止重答。"""
    if record['protocol_sha256']!=fingerprint or record['question']!=row['question'] or record['options']!=row['options']:
        raise ProtocolError('Formal identity mismatch')
    offline_record=read_json(run/'offline'/f'{row["question_id"]}.json');selection=record['selection']
    historical=read_json(BASELINE/'results/topk'/f'{row["question_id"]}.json')['selection'];verify_selection(selection,historical)
    for key in ('candidates','selected_frames','plan','replacements'):
        def agrees(a,b):
            """离散身份必须一致；浮点诊断值只允许预定1e-6绝对误差。"""
            if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(agrees(a[k],b[k]) for k in a)
            if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(agrees(x,y) for x,y in zip(a,b))
            if isinstance(a,float):return isinstance(b,(int,float)) and abs(a-b)<=1e-6
            return a==b
        if not agrees(selection[key],offline_record['selection'][key]):raise ProtocolError(f'Formal/offline {key} differs')
    n=len(selection['selected_frames']);a=record['answer']
    if a['pixel_shape']!=[n,3,384,384] or a['visual_tokens']!=n*210 or a['prefill_tokens']!=a['text_input_tokens']-1+n*210:
        raise ProtocolError('Formal visual input protocol failed')
    from videoqa_runtime.protocol import question_text,parse_answer
    message=question_text(row['question'],row['options'],selection['video']['duration_seconds'],[r['timestamp_seconds'] for r in selection['selected_frames']])
    if message not in a['prompt'] or a['parsed_answer']!=parse_answer(a['raw_output']) or len(a['generated_token_ids'])>8:
        raise ProtocolError('Formal prompt/parser/generation protocol failed')
    if record['reference_answer']!='ABCD'[row['answer_index']] or record['correct']!=(a['parsed_answer']==record['reference_answer']):
        raise ProtocolError('Formal scoring mismatch')


def worker(gpu,tasks,events,stop,run,fingerprint):
    """一卡一个常驻工作进程；错误持久化并暂停，不自动重试生成。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');offline_environment();run=Path(run)
    with (run/'logs'/f'gpu_{gpu}.log').open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import torch
            from videoqa_runtime.llava_backend import LlavaBackend
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            t=time.perf_counter();backend=LlavaBackend();scorer=BlipItmScorer(read_json(ROOT/'configs/local_model_inventory.json')['models']['blip']['path']);load=time.perf_counter()-t
            t=time.perf_counter();backend.warmup();scorer.warmup()
            durable(run/'workers'/f'{gpu}_{os.getpid()}.json',dict(load_seconds=load,warmup_seconds=time.perf_counter()-t,backend=backend.load_report,gpu=gpu,llava_synthetic_forwards=1,blip_synthetic_forwards=2))
            events.put(dict(type='ready',gpu=gpu))
            while not stop.is_set():
                row=tasks.get()
                if row is None:return
                formal_job(run,row,gpu,backend,scorer,fingerprint)
                events.put(dict(type='completed',gpu=gpu,question_id=row['question_id']))
        except BaseException as e:
            stop.set();traceback.print_exc();durable(run/'failures'/f'{gpu}_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc()))
            events.put(dict(type='failed',gpu=gpu,error=repr(e)))


def formal(run,rows,fingerprint,requested_gpus):
    """通过门槛后先5题再45题；后台协调器固定调用集合，最多50次，不根据正确率放行。"""
    check_gate(run,fingerprint)
    calls=list((run/'calls/answer').glob('*.json'))
    if len(calls)>50:raise ProtocolError('Answer call cap exceeded')
    ids={r['question_id'] for r in rows}
    for path in calls:
        c=read_json(path)
        if c['question_id'] not in ids or c['status']!='completed' or c['protocol_sha256']!=fingerprint:raise ProtocolError('Uncertain/invalid generation cannot resume')
    for row in rows:
        dest=run/'results'/f'{row["question_id"]}.json'
        if dest.exists():validate_formal(run,read_json(dest),row,fingerprint)
    if all((run/'results'/f'{r["question_id"]}.json').exists() for r in rows):return
    start=time.perf_counter();durable(run/'formal_preflight.json',preflight(rows));gpus,states=available_gpus(requested_gpus)
    durable(run/'gpu_allocation.json',dict(selected=gpus,states=states))
    ctx=mp.get_context('spawn');events=ctx.Queue();stop=ctx.Event();queues={g:ctx.Queue() for g in gpus}
    processes={g:ctx.Process(target=worker,args=(g,queues[g],events,stop,str(run),fingerprint)) for g in gpus}
    idle=set();active={};phase='startup';error=None
    def event():
        """观察超时只刷新状态，不重启工作进程；实际退出或失败才暂停。"""
        while True:
            durable(run/'status.json',dict(status=phase,completed=len(list((run/'results').glob('*.json'))),active=active,utc=utc()))
            try:item=events.get(timeout=10)
            except queue.Empty:
                if any(p.exitcode is not None for p in processes.values()):raise RuntimeError('Worker exited; inspect calls')
                continue
            if item['type']=='failed':raise RuntimeError(str(item))
            return item
    def schedule(batch):
        """空闲GPU领取未完成题，不重复已有结果。"""
        pending=[r for r in batch if not (run/'results'/f'{r["question_id"]}.json').exists()]
        while pending or active:
            for g in sorted(idle):
                if not pending:break
                row=pending.pop(0);idle.remove(g);active[g]=row['question_id'];queues[g].put(row)
            item=event()
            if item['type']=='completed':active.pop(item['gpu']);idle.add(item['gpu'])
    try:
        for p in processes.values():p.start()
        while len(idle)<len(gpus):idle.add(event()['gpu'])
        phase='formal_pilot';pilot=time.perf_counter();schedule(rows[:5])
        for row in rows[:5]:validate_formal(run,read_json(run/'results'/f'{row["question_id"]}.json'),row,fingerprint)
        durable(run/'pilot_acceptance.json',dict(status='passed',questions=[r['question_id'] for r in rows[:5]],seconds=time.perf_counter()-pilot))
        phase='formal_full';durations={r['question_id']:read_json(run/'offline'/f'{r["question_id"]}.json')['selection']['video']['duration_seconds'] for r in rows}
        schedule(sorted(rows[5:],key=lambda r:-durations[r['question_id']]))
        phase='formal_completed'
    except BaseException as e:error=repr(e);phase='paused';print(traceback.format_exc(),flush=True)
    finally:
        stop.set()
        for q in queues.values():q.put(None)
        while any(p.is_alive() for p in processes.values()):
            for p in processes.values():
                if p.pid is not None:p.join(timeout=1)
        durable(run/'execution.json',dict(status=phase,error=error,wall_seconds=time.perf_counter()-start,gpus=gpus,utc=utc()))
        durable(run/'status.json',dict(status=phase,completed=len(list((run/'results').glob('*.json'))),utc=utc()))
    if error:raise RuntimeError(error)


def execute(run_id,phase,gpus,resume=False):
    """批量入口：独立运行目录、排他锁、指纹冻结；不自动跨过人工证据门槛。"""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',run_id):raise ValueError('Invalid run ID')
    run=ROOT/'outputs/methods'/run_id;run.mkdir(parents=True,exist_ok=resume)
    with (run/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ('offline','review','results','calls/answer','checkpoints','logs','workers','failures','code_snapshot'):(run/name).mkdir(exist_ok=True,parents=True)
        spec=execution_protocol()
        if resume:
            if read_json(run/'protocol.json')!=spec:raise ProtocolError('Code/config/data fingerprint changed')
        else:
            durable(run/'protocol.json',spec)
            for name in spec['code_sha256']:
                dest=run/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        rows=read_json(ROOT/'data/manifests/video_mme_development.json')['rows']
        if len(rows)!=50 or len({r['question_id'] for r in rows})!=50:raise ProtocolError('Expected frozen development50')
        fingerprint=sha256(run/'protocol.json');durable(run/'pid.json',dict(pid=os.getpid(),session_id=os.getsid(0),phase=phase,utc=utc()))
        if phase=='offline':offline(run,rows,fingerprint)
        else:formal(run,rows,fingerprint,gpus)
    return run
