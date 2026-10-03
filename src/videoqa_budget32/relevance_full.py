"""exp12独立全量适配：不改冻结选择器，严格分离200条导入与2500次新回答。"""
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, rows, exclusive, utc, pilot_ids
from videoqa_full.report import compare
from videoqa_runtime.baseline_report import wilson, distribution
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from .frames import FrameProvider, validate_frames
from .run import verify as verify_runtime, validate, recover_call, process, PYTHON, critical_code
from .development import worker
from .prefix_full import partition

METHOD='RD-1.2-exp12'
BASE=ROOT/'outputs/full_videoqa/uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01'
REFERENCE=ROOT/'outputs/full_videoqa/rd12_exp07__qwen32__videomme2700__20260930_r1'
DEV=ROOT/'outputs/diagnostics/rd12_exp12__qwen32__dev200__20261002_r1'
DEV_MANIFEST=ROOT/'data/manifests/video_mme_development_200_20260920.json'
RELEASE=ROOT/'configs/releases/rd_1_2_exp12_qwen32_dev200_20261002.json'


def release_check():
    """接口：验证原开发协议、算法字节及冻结清单，不把新参数冒充旧候选。"""
    release=read_json(RELEASE)
    assert release['method_id']==METHOD
    assert sha256(DEV/'protocol.json')==release['protocol_sha256']
    assert sha256(DEV_MANIFEST)==release['dataset_manifest_sha256']
    assert sha256(ROOT/release['frozen_artifacts'])==release['frozen_artifacts_sha256']
    for name,h in release['method_code_sha256'].items():assert sha256(ROOT/name)==h,name
    assert sha256(ROOT/release['full_preflight'])==release['full_preflight_sha256']
    return release


def select_one(job):
    """接口：重算加权覆盖与DP，核验旧200题并真实解码冻结RGB；不执行模型。"""
    run,row,imported=job;q=row['question_id'];start=time.perf_counter()
    paths=[BASE/'pools'/(q+'.json'),BASE/'features'/(q+'.npy'),BASE/'selection_checkpoints'/(q+'.json'),REFERENCE/'traces'/(q+'.json'),RELEASE]
    sources={str(p):sha256(p) for p in paths};dest=run/'selections'/METHOD/(q+'.json')
    if dest.exists():
        value=read_json(dest)
        assert value['sources']==sources and sha256(run/value['trace_file'])==value['trace_sha256']
        return value
    # 步骤1：同一粗细池和原前缀，不重建补查或重算BLIP。
    check=read_json(paths[2])
    for p in paths[:2]:assert sources[str(p)]==check['files'][str(p.relative_to(BASE))]
    pool=read_json(paths[0]);reference=read_json(paths[3]);features=np.load(paths[1],allow_pickle=False);loaded=time.perf_counter()
    indices,trace=select_relevance_coverage(pool,features,reference)
    selected=time.perf_counter();selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in indices])
    validate_frames(selection)
    assert len(indices)==sum(reference['region_budgets'].values())
    assert set(reference['protected_indices'])<=set(indices)
    images,hashes=FrameProvider(selection).decode()
    for im in images:im.close()
    # 步骤2：旧题只允许原输入重现，答案尚不在此阶段导入。
    if imported:
        old=read_json(DEV/'selections'/METHOD/(q+'.json'))
        assert old['selection']==selection and old['rgb_sha256']==hashes,q
    tp=run/'traces'/(q+'.json');durable(tp,trace)
    value=dict(question_id=q,video_id=row['video_id'],method=METHOD,selection=selection,rgb_sha256=hashes,
        sources=sources,selected_indices=indices,trace_file=str(tp.relative_to(run)),trace_sha256=sha256(tp),
        imported_answer=imported,new_model_calls=0,
        timings=dict(asset_hash_read_seconds=loaded-start,weighted_curve_dp_seconds=selected-loaded,
                     final_decode_seconds=time.perf_counter()-selected))
    durable(dest,value);return value


def selection_all(run):
    """接口：全部2700题固定选择，四线程CPU准备；完成清单是生成的前置条件。"""
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'selection.lock'):
        release=release_check();data=rows();dev=[r['question_id'] for r in read_json(DEV_MANIFEST)['rows']]
        imported,new=partition(data,dev);assert (len(imported),len(new))==(200,2500)
        if (run/'selection_summary.json').exists():
            old=read_json(run/'selection_summary.json');assert old['release_sha256']==sha256(RELEASE)
            assert old['imported_ids']==imported and old['new_ids']==new and len(old['exports'])==2700
            for p,h in old['exports'].items():assert sha256(run/p)==h
            return
        assert shutil.disk_usage(run).free>30*1024**3
        start=time.perf_counter();exports={}
        # 步骤1：每条原子保存；中断只恢复尚未完成的选择，不开启新的运行编号。
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures=[ex.submit(select_one,(run,r,r['question_id'] in set(imported))) for r in data]
            for future in as_completed(futures):
                item=future.result();rel=f'selections/{METHOD}/{item["question_id"]}.json';exports[rel]=sha256(run/rel)
                durable(run/'selection_status.json',dict(status='running',completed=len(exports),total=2700,
                    pid=os.getpid(),sid=os.getsid(0),elapsed_seconds=time.perf_counter()-start,utc=utc()))
        # 步骤2：冻结完整名单，后续不依据分数重新选择。
        durable(run/'selection_summary.json',dict(status='passed',exports=exports,imported_ids=imported,new_ids=new,
            new_model_calls=0,wall_seconds=time.perf_counter()-start,release_sha256=sha256(RELEASE)))
        durable(run/'selection_status.json',dict(status='completed',completed=2700,pid=os.getpid(),sid=os.getsid(0),utc=utc()))


def prepare(run,allowed):
    """接口：模型协议相同才导入200条原始记录；新增代码快照与2500次调用额度单列。"""
    from videoqa_full.prepare import environment
    from videoqa_lmms_bridge.common import framework_identity
    run=Path(run)
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():return verify(run)
        release=release_check();base=verify_runtime(BASE);devspec=verify_runtime(DEV)
        sel=read_json(run/'selection_summary.json');syn=read_json(run/'synthetic_acceptance.json')
        assert sel['status']==syn['status']=='passed'
        assert syn['code_sha256']==critical_code() and syn['environment']==environment(PYTHON)
        assert syn['framework']==framework_identity()
        assert allowed and len(set(allowed))==len(allowed) and set(allowed)<=set(map(str,range(8)))
        assert syn['gpu'] in allowed
        for key in ['model','profile','framework','environment','model_files']:assert base[key]==devspec[key],key
        # 步骤1：预检已读满视频/权重；确认文件元数据仍与冻结版本一致，不重写预检。
        preflight=read_json(ROOT/release['full_preflight']);assert preflight['status']=='passed'
        for v,st in base['video_stats'].items():
            actual=Path(st['path']).stat();assert actual.st_size==st['bytes'] and actual.st_mtime_ns==st['mtime_ns']
            assert preflight['verified_video_model_bytes'][st['path']]==base['video_hashes'][v]
        for p,st in base['model_files'].items():
            actual=Path(p).stat();assert actual.st_size==st['bytes'] and actual.st_mtime_ns==st['mtime_ns']
            assert preflight['verified_video_model_bytes'][p]==st['sha256']
        sources={str(RELEASE):sha256(RELEASE),str(ROOT/release['full_preflight']):release['full_preflight_sha256']};imports={}
        for rel,h in sel['exports'].items():
            assert sha256(run/rel)==h;item=read_json(run/rel)
            for p,value in item['sources'].items():assert sha256(p)==value;sources[p]=value
            sources[str(run/item['trace_file'])]=item['trace_sha256']
        byid={r['question_id']:r for r in rows()}
        frozen=read_json(ROOT/release['frozen_artifacts'])['files']
        # 步骤2：原JSON逐字节复制，不能伪装成本轮生成或更换旧预测。
        for q in sel['imported_ids']:
            old=validate(DEV,byid[q],METHOD,devspec);current=read_json(run/'selections'/METHOD/(q+'.json'))
            assert old['selection']==current['selection'] and old['rgb_sha256']==current['rgb_sha256']
            files={}
            for folder in ['results','calls','raw_generations']:
                p=DEV/folder/METHOD/(q+'.json');h=sha256(p);assert h==frozen[str(p.relative_to(ROOT))]
                dest=run/'imported'/folder/METHOD/(q+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():assert sha256(dest)==h
                else:shutil.copy2(p,dest)
                files[str(dest.relative_to(run))]=h;sources[str(p)]=h
            imports[q]=dict(origin=str(DEV),files=files,protocol_sha256=sha256(DEV/'protocol.json'))
        assert sum(read_json(DEV/'results'/METHOD/(q+'.json'))['correct'] for q in sel['imported_ids'])==release['results']['correct']
        for row in byid.values():
            q=row['question_id']
            for m in ['BASE-Uniform','BASE-BLIP-TopK','RD-1.2']:
                p=BASE/'results'/m/(q+'.json');sources[str(p)]=sha256(p)
            p=REFERENCE/('imported/results' if q in imports else 'results')/'RD-1.2-exp07'/(q+'.json');sources[str(p)]=sha256(p)
        # 步骤3：只快照新适配层和原算法；不改原全量入口。
        code=dict(devspec['code_sha256'])
        for name in ['src/videoqa_budget32/relevance_full.py','src/videoqa_budget32/prefix_full.py',
                     'scripts/run_relevance_coverage_full.py',*release['method_code_sha256']]:code[name]=sha256(ROOT/name)
        for name,h in code.items():
            assert sha256(ROOT/name)==h
            dest=run/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        spec=dict(base,run_id=run.name,methods=[METHOD],question_count=2700,max_calls=2500,max_calls_per_method=2500,
            question_ids=list(byid),imported_ids=sel['imported_ids'],new_ids=sel['new_ids'],imports=imports,
            allowed_gpus=allowed,selection_exports=sel['exports'],input_sources=sources,code_sha256=code,
            development_run=str(DEV),release_sha256=sha256(RELEASE),
            synthetic_acceptance_sha256=sha256(run/'synthetic_acceptance.json'),pilot_question_ids=pilot_ids(list(byid.values())),
            configuration='RD-1.2-exp12 M32/K32; 200 imported + 2500 new',utc=utc())
        durable(run/'protocol.json',spec);return spec


def verify(run):
    """接口：启动与恢复严格核对代码、输入和导入来源，任何漂移暂停。"""
    run=Path(run);spec=verify_runtime(run);release_check()
    assert spec['methods']==[METHOD] and spec['max_calls']==2500 and spec['question_count']==2700
    old,new=partition(rows(),spec['imported_ids']);assert old==spec['imported_ids'] and new==spec['new_ids']
    assert len(old)==200 and len(new)==2500 and sha256(RELEASE)==spec['release_sha256']
    assert sha256(run/'synthetic_acceptance.json')==spec['synthetic_acceptance_sha256']
    for p,h in spec['input_sources'].items():assert sha256(p)==h,p
    for p,h in spec['selection_exports'].items():assert sha256(run/p)==h,p
    for item in spec['imports'].values():
        for p,h in item['files'].items():assert sha256(run/p)==h,p
    return spec


def execute(run):
    """接口：后台协调2500个新题，导入题不进队列；恢复已生成项仅完成CPU后处理。"""
    from videoqa_runtime.baseline_run import available_gpus
    run=Path(run)
    with exclusive(run/'coordinator.lock'):
        spec=verify(run);data={r['question_id']:r for r in rows()};new=set(spec['new_ids']);done=set()
        active={};gpus=[];phase='checking';failure=None;start=time.perf_counter()
        def publish():
            """步骤状态：新调用进度与旧200条分开，ETA只基于当前批次。"""
            elapsed=time.perf_counter()-start
            durable(run/'status.json',dict(status=phase,completed_new=len(done),total_new=2500,imported=200,
                completed_total=200+len(done),active=active,gpus=gpus,pid=os.getpid(),session_id=os.getsid(0),
                elapsed_seconds=elapsed,eta_seconds=(2500-len(done))*elapsed/len(done) if done else None,failure=failure,utc=utc()))
        # 步骤1：已返回调用用同一原始答案恢复，不加载模型、不消耗合成或真实生成。
        calls=list((run/'calls'/METHOD).glob('*.json'));assert len(calls)<=2500
        for p in calls:
            assert p.stem in new;recover_call(run,data[p.stem],METHOD,spec)
            c=read_json(p);assert c['status']=='completed' and c['gpu'] in spec['allowed_gpus']
            if not (run/'results'/METHOD/p.name).exists():process(run,data[p.stem],METHOD,None,None,c['gpu'],spec)
        for p in (run/'results'/METHOD).glob('*.json'):
            assert p.stem in new;validate(run,data[p.stem],METHOD,spec);done.add(p.stem)
        if done==new:
            if not (run/'execution.json').exists():durable(run/'execution.json',dict(status='completed',new_answer_calls=2500,wall_seconds=None,
                actual_gpus=sorted({read_json(p)['gpu'] for p in calls}),recovered_completion=True))
            durable(run/'recovery_status.json',dict(status='completed',completed_new=2500,imported=200,new_generations=0,utc=utc()))
            if not (run/'status.json').exists() or read_json(run/'status.json')['status']!='completed':
                durable(run/'status.json',dict(status='completed',completed_new=2500,total_new=2500,imported=200,
                    completed_total=2700,elapsed_seconds=None,recovered_completion=True,utc=utc()))
            return
        while True:
            try:gpus,states=available_gpus(spec['allowed_gpus'])
            except RuntimeError:gpus,states=[],{}
            if gpus:break
            phase='waiting_resources';publish();time.sleep(15)
        for name in ['logs','workers','failures']:(run/name).mkdir(exist_ok=True)
        durable(run/'gpu_preflight.json',dict(actual=gpus,states=states,utc=utc()))
        ctx=mp.get_context('spawn');jobs={g:ctx.Queue() for g in gpus};events=ctx.Queue();stop=ctx.Event()
        children={};ready=set();busy=set();pending=[];pilot=[q for q in spec['pilot_question_ids'] if q in new]
        def schedule():
            """先行只按协议放行，余题按时长排序；不使用正确率。"""
            nonlocal phase,pending
            if any(q not in done for q in pilot):phase='pilot';pending=[q for q in pilot if q not in done]
            else:
                for q in pilot:validate(run,data[q],METHOD,spec)
                durable(run/'pilot_acceptance.json',dict(status='passed',new_ids=pilot,imported_ids=[q for q in spec['pilot_question_ids'] if q not in new],accuracy_not_gate=True))
                phase='running';pending=sorted(new-done,key=lambda q:(-read_json(run/'selections'/METHOD/(q+'.json'))['selection']['video']['duration_seconds'],q))
        def refill():
            """单卡最多一个在途题；生成不确定的题已在恢复校验处拒绝，不会再次派发。"""
            for gpu in gpus:
                if gpu in busy or not pending:continue
                q=pending.pop(0);busy.add(gpu);jobs[gpu].put(data[q])
        # 步骤2：复用正式Qwen工作进程，阶段切换不依赖观察连接。
        try:
            phase='loading'
            for gpu in gpus:
                child=ctx.Process(target=worker,args=(str(run),gpu,jobs[gpu],events,stop));child.start();children[gpu]=child
            publish()
            while len(done)<2500:
                try:event=events.get(timeout=5)
                except queue.Empty:event=None
                if event:
                    if event['type']=='failed':raise RuntimeError(str(event))
                    if event['type']=='ready':
                        ready.add(event['gpu'])
                        if len(ready)==len(gpus):schedule();refill()
                    elif event['type']=='active':active[event['gpu']]=event['question_id']
                    elif event['type']=='completed':
                        assert event['question_id'] not in done
                        done.add(event['question_id']);busy.remove(event['gpu']);active.pop(event['gpu'],None)
                        if phase=='pilot' and not pending and not busy:schedule()
                        refill()
                for gpu,child in children.items():
                    if child.exitcode is not None:raise RuntimeError(f'Worker {gpu} exited {child.exitcode}')
                publish()
        except BaseException as exc:
            failure=repr(exc);phase='paused';stop.set();publish();raise
        finally:
            for channel in jobs.values():channel.put(None)
            while any(p.is_alive() for p in children.values()):
                for child in children.values():child.join(timeout=1)
                publish()
        phase='completed';publish()
        durable(run/'execution.json',dict(status='completed',new_answer_calls=2500,actual_gpus=gpus,wall_seconds=time.perf_counter()-start,utc=utc()))


def summarize(run):
    """接口：五组配对和分母分开报告；不把导入200题当新生成，不只展示有利案例。"""
    from collections import Counter
    run=Path(run);spec=verify(run);data=rows();devspec=read_json(DEV/'protocol.json');current={}
    for folder in ['results','calls','raw_generations']:
        assert {p.stem for p in (run/folder/METHOD).glob('*.json')}==set(spec['new_ids'])
    for row in data:
        q=row['question_id'];old=q in spec['imported_ids']
        value=validate(DEV if old else run,row,METHOD,devspec if old else spec)
        assert value['correct']==(value['answer']['parsed_answer']=='ABCD'[row['answer_index']]);current[q]=value
    records={METHOD:current}
    for m in ['BASE-Uniform','BASE-BLIP-TopK','RD-1.2','RD-1.2-exp07']:
        records[m]={}
        for row in data:
            q=row['question_id']
            p=(REFERENCE/('imported/results' if q in spec['imported_ids'] else 'results')/m/(q+'.json')) if m=='RD-1.2-exp07' else BASE/'results'/m/(q+'.json')
            value=read_json(p);assert value['correct']==(value['answer']['parsed_answer']=='ABCD'[row['answer_index']]);records[m][q]=value
    devq=set(spec['imported_ids']);devv={r['video_id'] for r in data if r['question_id'] in devq}
    subsets=dict(all2700=data,development200=[r for r in data if r['question_id'] in devq],
        other2500=[r for r in data if r['question_id'] not in devq],outside_development_videos2100=[r for r in data if r['video_id'] not in devv],
        common32=[r for r in data if all(len(g[r['question_id']]['selection']['selected_frames'])==32 for g in records.values())])
    assert [len(subsets[k]) for k in list(subsets)[:4]]==[2700,200,2500,2100]
    def metric(items):
        """按真实分母计算准确率和Wilson区间，不筛除答错/解析失败。"""
        c=sum(v['correct'] for v in items);return dict(n=len(items),correct=c,accuracy=c/len(items),wilson_95=wilson(c,len(items)))
    tables={m:{s:{layer:metric([g[r['question_id']] for r in rs if layer=='all' or r['stratum']==layer]) for layer in ['all','short','medium','long']} for s,rs in subsets.items()} for m,g in records.items()}
    pairs={m:{s:compare(rs,current,g) for s,rs in subsets.items()} for m,g in records.items() if m!=METHOD}
    types=sorted({r['task_type'] for r in data})
    type_stats={m:{t:metric([g[r['question_id']] for r in data if r['task_type']==t]) for t in types} for m,g in records.items()}
    flat=[dict(question_id=r['question_id'],video_id=r['video_id'],stratum=r['stratum'],task_type=r['task_type'],
        source='imported_development' if r['question_id'] in devq else 'new_full_batch',
        methods={m:dict(correct=g[r['question_id']]['correct'],answer=g[r['question_id']]['answer']['parsed_answer'],frames=len(g[r['question_id']]['selection']['selected_frames'])) for m,g in records.items()}) for r in data]
    durable(run/'per_question.json',flat)
    same={}
    for m,g in records.items():
        if m==METHOD:continue
        ids=[q for q in current if current[q]['rgb_sha256']==g[q]['rgb_sha256']]
        same[m]=dict(question_ids=ids,prediction_differences=[q for q in ids if current[q]['answer']['parsed_answer']!=g[q]['answer']['parsed_answer']])
    # 步骤1：主计时排除CPU后处理恢复条目；原始QA观测仍保留在逐题文件。
    fresh=[current[q] for q in spec['new_ids'] if not current[q]['reused_answer']]
    performance={k:distribution([r['timings'][k] for r in fresh]) if fresh else None for k in ['final_decode_seconds','cached_input_to_persisted_answer_seconds']}
    performance['fresh_timing_count']=len(fresh)
    performance['qa_seconds']=distribution([current[q]['answer']['observed'].get('qa_seconds',current[q]['answer']['observed']['qa_through_generation_seconds']) for q in spec['new_ids']])
    performance['peak_allocated_gib']=distribution([current[q]['answer']['observed']['peak_allocated_gib'] for q in spec['new_ids']])
    performance['peak_reserved_gib']=distribution([current[q]['answer']['observed']['peak_reserved_gib'] for q in spec['new_ids']])
    # 步骤2：结构变化相对exp07；不把窗口和补查未变解释为补查收益。
    changes=[]
    for row in data:
        q=row['question_id'];item=read_json(run/'selections'/METHOD/(q+'.json'));tr=read_json(run/item['trace_file'])
        ref=read_json(REFERENCE/'traces'/(q+'.json'));pool=read_json(BASE/'pools'/(q+'.json'))
        old={f['candidate_index'] for f in records['RD-1.2-exp07'][q]['selection']['selected_frames']};new=set(item['selected_indices'])
        assert all(a['prefix']==b['prefix'] for a,b in zip(tr['regions'],ref['regions']))
        assert len(tr['regions'])==len(ref['regions']) and set(ref['protected_indices'])<=new
        changes.append(dict(question_id=q,old_budgets=ref['region_budgets'],new_budgets=tr['region_budgets'],
            added=sorted(new-old),removed=sorted(old-new),selected_fine=sum(i>=pool['initial_count'] for i in new),local_prefixes_unchanged=True))
    durable(run/'stage_analysis.json',dict(rows=changes,scope='Budget reallocation only; not semantic evidence completeness'))
    cases=[]
    for m in ['RD-1.2-exp07','BASE-Uniform']:
        for layer in ['short','medium','long']:
            for direction,key in [('gain','left_only'),('loss','right_only')]:
                eligible=set(pairs[m]['all2700']['cells'][key]);q=next((r['question_id'] for r in data if r['stratum']==layer and r['question_id'] in eligible),None)
                if q:cases.append(dict(question_id=q,comparison=m,stratum=layer,direction=direction))
    durable(run/'case_manifest.json',dict(entries=cases,unique_ids=list(dict.fromkeys(c['question_id'] for c in cases)),rule='first frozen-order gain/loss per stratum; no backfill'))
    summary=dict(status='completed',method=METHOD,accuracy=tables,task_types=type_stats,comparisons=pairs,
        identical_rgb_checks=same,new_calls=2500,imported=200,performance=performance,
        resumed_postprocessing=[q for q in spec['new_ids'] if current[q]['reused_answer']],
        frame_counts={m:dict(Counter(len(x['selection']['selected_frames']) for x in g.values())) for m,g in records.items()},
        parse_failures={m:[q for q,x in g.items() if x['answer']['parsed_answer'] is None] for m,g in records.items()},
        execution=read_json(run/'execution.json'),selection_wall_seconds=read_json(run/'selection_summary.json')['wall_seconds'],
        synthetic_generations=read_json(run/'synthetic_acceptance.json')['synthetic_calls_started']+sum(read_json(p)['synthetic_generations'] for p in (run/'workers').glob('*.json')),
        failures=[str(p.relative_to(run)) for p in (run/'failures').glob('*.json')],
        limitations=['Repeated development and previously observed full dataset, not blind','200 imported plus2500 new; no best-of-runs','Cached upstream assets; not raw-video E2E'],utc=utc())
    durable(run/'summary.json',summary)
    # 步骤3：后台同时生成简明可读报告；不声称尚未完成的案例语义审核已经完成。
    lines=[f'# {METHOD} Qwen32全量结果','',
        '200条冻结开发答案＋2500条新回答；人工案例解释另行审核。','',
        '| 方法 | 全量正确数 | 准确率 | 短 | 中 | 长 | 开发200 | 其余2500 | 开发视频外2100 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m,groups in tables.items():
        a=groups['all2700']
        values=[str(a['all']['correct']),f"{100*a['all']['accuracy']:.3f}%"]
        values += [str(a[t]['correct']) for t in ['short','medium','long']]
        values += [str(groups[t]['all']['correct']) for t in ['development200','other2500','outside_development_videos2100']]
        lines.append('| '+m+' | '+' | '.join(values)+' |')
    lines += ['','## 配对结果','']
    for m,groups in pairs.items():
        p=groups['all2700'];lo,hi=p['stratified_video_cluster_bootstrap_95']
        lines.append(f"- 相对{m}：{p['difference_pp']:+.3f}个百分点；新增{len(p['cells']['left_only'])}、退化{len(p['cells']['right_only'])}；视频簇95%区间[{lo:+.3f},{hi:+.3f}]。")
    lines += ['','开发集已反复用于探索，其余数据也不称完全未见盲测。缓存上游资产的成本不是原片独立E2E。',
              '完整题型、帧数、配对题号和性能见summary.json；来源见per_question.json；恢复审计以completion_audit.json为准。']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def audit(run):
    """接口：真实恢复入口复验8100份新旧生成记录，不加载模型或新增生成。"""
    run=Path(run)
    def fingerprints():
        """枚举新2500与旧200的答案、账本、原文，排除易变心跳文件。"""
        return {str(p.relative_to(run)):sha256(p) for root in [run,run/'imported'] for folder in ['results','calls','raw_generations'] for p in (root/folder/METHOD).glob('*.json')}
    before=fingerprints();assert len(before)==8100
    execute(run);assert before==fingerprints()
    durable(run/'completion_audit.json',dict(status='passed',unchanged_files=8100,extra_generations=0,utc=utc()))
