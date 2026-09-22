"""密度v1局部验证：只比较选帧线索，不生成答案，不改写正式方法或历史结果。"""
from fractions import Fraction
from pathlib import Path
import fcntl
import html
import math
import os
import re
import shutil
import time
import traceback
import numpy as np
from videoqa_runtime.common import ROOT, read_json, sha256, offline_environment
from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_runtime.baseline_run import available_gpus
from videoqa_methods.density_selection import build_regions, select_joint, validate_candidates, source_time, frame_key
from videoqa_methods.density_pipeline import DensityScorer, decode_regions
from videoqa_methods.followups import durable, utc
from videoqa_methods.region_context import read_exact_frames

SOURCE = ROOT/'outputs/methods/retrieval_density_v1_videomme50_20260920_r1'
SORT_CASES = ['103-1','383-2','395-2','012-3','893-1','872-2']
WINDOW_CASES = ['306-2','409-3','443-1','103-1']
CASES = list(dict.fromkeys(SORT_CASES+WINDOW_CASES))
FOCUS = {'103-1':(4,8),'383-2':(940,950),'395-2':(269,278),'012-3':(14,22),
         '306-2':(86,91),'409-3':(75,83),'443-1':(135,146)}


def supported_selection(candidates, features, plan, video, cfg, quotas=None):
    """接口：固定原种子视觉支持L乘补充差异u；可固定旧配额做归因诊断。"""
    validate_candidates(candidates,video)
    features=np.asarray(features,dtype=np.float32)
    if features.ndim!=2 or features.shape[0]!=len(candidates) or not np.isfinite(features).all():
        raise ProtocolError('Invalid feature matrix')
    if not np.allclose(np.linalg.norm(features,axis=1),1,atol=1e-5,rtol=0):
        raise ProtocolError('Features not normalized')
    selected={};members={};support={};redundancy={};nearest={}
    # 步骤1：参照集合来自原始种子，固定不扩张；新旧来源不参与评分。
    for region in plan['regions']:
        rid=region['region_id'];left=Fraction(region['left_fraction']);right=Fraction(region['right_fraction'])
        ids=sorted([i for i,r in enumerate(candidates) if left<=source_time(r,video)<=right],key=lambda i:frame_key(candidates[i]))
        if region['anchor_index'] not in ids or not set(region['seed_indices'])<=set(ids):
            raise ProtocolError('Region lost its original seeds')
        if any(set(ids)&set(v) for v in members.values()):
            raise ProtocolError('Overlapping final regions')
        members[rid]=ids;selected[rid]=[region['anchor_index']]
        support[rid]=np.maximum.reduce([np.clip(features[ids]@features[j],0,1) for j in region['seed_indices']])
        redundancy[rid]=np.clip(features[ids]@features[region['anchor_index']],0,1)
        nearest[rid]=np.full(len(ids),region['anchor_index'],dtype=np.int64)
    target=min(cfg['frame_budget'],sum(map(len,members.values())))
    if len(selected)>target:raise ProtocolError('Anchor capacity exceeded')
    if quotas is not None:
        if set(quotas)!={str(k) for k in selected} or sum(quotas.values())!=target:
            raise ProtocolError('Fixed quota scope/budget mismatch')
        if any(not 1<=quotas[str(rid)]<=len(members[rid]) for rid in members):
            raise ProtocolError('Fixed quota outside candidate capacity')
    trace=[]
    # 步骤2：区域内最大L*u提议；联合模式用同一效用竞争，固定模式只额外限制旧配额。
    while sum(map(len,selected.values()))<target:
        proposals=[]
        for region in plan['regions']:
            rid=region['region_id']
            if quotas is not None and len(selected[rid])>=quotas[str(rid)]:continue
            remaining=[j for j,i in enumerate(members[rid]) if i not in selected[rid]]
            if not remaining:continue
            utility=lambda j:float(support[rid][j])*(1-float(redundancy[rid][j]))
            j=min(remaining,key=lambda j:(-utility(j),*frame_key(candidates[members[rid][j]])))
            proposals.append(dict(region_id=rid,candidate_index=members[rid][j],support=float(support[rid][j]),
                redundancy=float(redundancy[rid][j]),diversity=1-float(redundancy[rid][j]),utility=utility(j),
                weight=region['weight'],priority=region['weight']*utility(j),nearest_selected_index=int(nearest[rid][j])))
        if not proposals:raise ProtocolError('No proposal before budget completed')
        winner=min(proposals,key=lambda p:(-p['priority'],*frame_key(candidates[p['candidate_index']])))
        rid=winner['region_id'];i=winner['candidate_index'];selected[rid].append(i)
        trace.append(dict(step=len(trace),proposals=proposals,winner=winner))
        # 步骤3：只更新已选集合的重复度；L保持固定，不让选中离群画面扩张支持集合。
        values=np.clip(features[members[rid]]@features[i],0,1);changed=values>redundancy[rid]
        nearest[rid][changed]=i;redundancy[rid]=np.maximum(redundancy[rid],values)
    indices=sorted([i for group in selected.values() for i in group],key=lambda i:frame_key(candidates[i]))
    return indices,dict(protected_indices=[r['anchor_index'] for r in plan['regions']],rounds=trace,
        eligible_indices=sorted(i for ids in members.values() for i in ids),
        region_selected={str(k):v for k,v in selected.items()},region_budgets={str(k):len(v) for k,v in selected.items()},
        effective_frame_budget=target,fixed_quotas=quotas,
        supports={str(rid):dict(zip(map(str,members[rid]),map(float,support[rid]))) for rid in members})


def spec():
    """冻结9题、两个分支、源结果和执行文件；没有任何问答或ITM调用入口。"""
    old=read_json(SOURCE/'protocol.json')
    if np.__version__!=read_json(ROOT/'configs/runtime_environment.json')['packages']['numpy']:
        raise ProtocolError('Use the original conda numerical environment')
    for name,digest in old['code_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ProtocolError(f'v1 execution file changed: {name}')
    files=[ROOT/'src/videoqa_audit/density_local_validation.py',ROOT/'scripts/run_density_local_validation.py']
    sources=[SOURCE/'protocol.json',SOURCE/'dataset_manifest.json']
    sources += [ROOT/'outputs/analysis/evidence_videomme50_20260915_r1/full_review/witnesses_r6'/f'{q}.json' for q in ('893-1','872-2')]
    for q in CASES:
        record=read_json(SOURCE/'results'/f'{q}.json')
        sources.extend([SOURCE/'results'/f'{q}.json',SOURCE/record['feature_file']])
    return dict(version='density-local-validation-v1',sort_cases=SORT_CASES,window_cases=WINDOW_CASES,
        sort_modes=['control','soft_fixed','soft_joint'],window_radii=[4,5,6],config=old['config'],
        question_answer_calls=0,scope_calls=0,query_calls=0,new_itm_calls=0,
        selection_utility='max_similarity_to_original_seeds * (1-max_similarity_to_selected)',
        model_compute='only missing window-extension visual features; reuse by exact source PTS',
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sources},
        v1_execution_sha256=old['code_sha256'],historical_result_sha256=old['history_sha256'],
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in files},
        judgment='visible evidence and factual roles, not QA accuracy; known witnesses are non-exhaustive; no auto formal50')


def known_witnesses(qid, original):
    """评价接口：既有人工见证仅用于报告，不传入区域构建或选择器。"""
    indices={
        '103-1':{'fork_identity':[5,64,65]},
        '383-2':{'explicit_boat':[944,1106,1107]},
        '395-2':{'retreat_flyer':[273,276],'later_seaside':[414,424]},
        '012-3':{'gray_tape':[15,18,65,71]},
        '306-2':{'first_meeting_context':[88],'combing_closeup':[89,90]},
        '409-3':{'carrying_box':[77],'opening_rose_box':[80],'gun_after_opening':[81]},
        '443-1':{'first_game_partial_actions':[102,123,137],'first_game_result':[145]},
    }
    result={key:sorted({original['candidates'][i]['source_pts'] for i in ids}) for key,ids in indices.get(qid,{}).items()}
    if qid in ('893-1','872-2'):
        catalog=read_json(ROOT/'outputs/analysis/evidence_videomme50_20260915_r1/full_review/witnesses_r6'/f'{qid}.json')
        result={f['id']:sorted({w['source_pts'] for w in f['witnesses']}) for f in catalog['facts'] if f['role'] in ('required','partial')}
    return result


def render(path, rows, video, processor, title, selected_pts=None, eligible_pts=None):
    """生成384输入/焦点图板，标记源PTS、入选与区域资格；只做CPU图像处理。"""
    from PIL import Image,ImageDraw
    if not rows:return
    canvas=Image.new('RGB',(1536,32+math.ceil(len(rows)/4)*410),'white');draw=ImageDraw.Draw(canvas)
    draw.text((6,8),title,fill='black')
    # 步骤1：按精确源PTS取图，分批处理保证原分辨率RGB内存有界。
    for start in range(0,len(rows),16):
        batch=rows[start:start+16];images=read_exact_frames(video,batch)
        pixels=processor.preprocess(images,return_tensors='np')['pixel_values']
        for offset,(row,pixel) in enumerate(zip(batch,pixels)):
            j=start+offset;x=j%4*384;y=32+j//4*410
            rgb=pixel.transpose(1,2,0)*np.asarray(processor.image_std)+np.asarray(processor.image_mean)
            canvas.paste(Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype('uint8')),(x,y))
            chosen=selected_pts is not None and row['source_pts'] in selected_pts
            eligible=eligible_pts is None or row['source_pts'] in eligible_pts
            draw.rectangle((x,y,x+383,y+383),outline='green' if chosen else ('orange' if eligible else 'gray'),width=3)
            draw.text((x+4,y+386),f'#{row["candidate_index"]} {row["timestamp_seconds"]:.5f}s sel={int(chosen)} eligible={int(eligible)}',fill='black')
        for im in images:im.close()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def save_condition(run,qid,label,selection,features,indices,trace,original,processor,extra=None):
    """保存条件产物、事实可达性与图板，明确离线缓存成本不等于正式原视频E2E。"""
    key=f'{qid}__{label}';rows=selection['candidates'];video=selection['video']
    feature_path=run/'features'/f'{key}.npy'
    with feature_path.open('wb') as output:np.save(output,features,allow_pickle=False)
    chosen={rows[i]['source_pts'] for i in indices};original_chosen={r['source_pts'] for r in original['selected_frames']}
    eligible={rows[i]['source_pts'] for i in trace['eligible_indices']}
    witnesses=known_witnesses(qid,original)
    record=dict(question_id=qid,condition=label,selection=selection,selected_indices=indices,
        selected_frames=[rows[i] for i in indices],trace=trace,feature_file=str(feature_path.relative_to(run)),
        feature_sha256=sha256(feature_path),qa_calls=0,new_itm_calls=0,application_cache=True,
        change=dict(added_pts=sorted(chosen-original_chosen),removed_pts=sorted(original_chosen-chosen)),
        witness_proxy={fact:dict(listed_pts=pts,eligible_pts=sorted(set(pts)&eligible),selected_pts=sorted(set(pts)&chosen),
                                  limitation='No listed witness does not prove no equivalent evidence') for fact,pts in witnesses.items()},
        timing_kind='offline cached diagnostic, not independent algorithm E2E',extra=extra or {})
    render(run/'boards'/f'{key}.png',record['selected_frames'],video,processor,f'{qid} {label}: 16 selected; NO QA',chosen,eligible)
    if qid in FOCUS:
        lo,hi=FOCUS[qid];focus=sorted([r for r in rows if lo<=r['timestamp_seconds']<=hi],key=frame_key)
        render(run/'focus'/f'{key}.png',focus,video,processor,f'{qid} {label}: focus; green=selected orange=eligible gray=outside',chosen,eligible)
    durable(run/'records'/f'{key}.json',record)
    return record


def execute(run_id, requested_gpu='0', resume=False):
    """后台单GPU验证入口：A复用特征，B只编码扩窗新增帧；完成后不自动启动问答。"""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',run_id):raise ProtocolError('Invalid run ID')
    os.environ['CUDA_VISIBLE_DEVICES']=requested_gpu
    run=ROOT/'outputs/analysis'/run_id;run.mkdir(parents=True,exist_ok=resume)
    with (run/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ('records','features','boards','focus','fine_frames','jobs','code_snapshot'):(run/name).mkdir(exist_ok=True)
        protocol=spec()
        if resume:
            if read_json(run/'protocol.json')!=protocol:raise ProtocolError('Analysis fingerprint changed')
        else:
            durable(run/'protocol.json',protocol)
            for name in protocol['code_sha256']:
                target=run/'code_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        fingerprint=sha256(run/'protocol.json');start=time.perf_counter();scorer=None;processor=None
        rows={r['question_id']:r for r in read_json(SOURCE/'dataset_manifest.json')['rows']}
        records=[]
        def status(phase,qid=None):
            """观察只读取持久状态，不因连接断开重复模型计算。"""
            durable(run/'status.json',dict(status=phase,question_id=qid,completed_conditions=len(records),qa_calls=0,
                    blip_counts=None if scorer is None else scorer.counts,elapsed_seconds=time.perf_counter()-start,utc=utc()))
        def existing(key):
            """完成条件只读复用；未完成状态不自动重跑视觉批次。"""
            job=run/'jobs'/f'{key}.json'
            if not job.exists():return None
            state=read_json(job)
            if state['protocol_sha256']!=fingerprint or state['status']!='completed':raise ProtocolError(f'Incomplete job needs inspection: {key}')
            for name,digest in state['files'].items():
                if sha256(run/name)!=digest:raise ProtocolError(f'Analysis artifact changed: {name}')
            return read_json(run/'records'/f'{key}.json')
        def begin(key):durable(run/'jobs'/f'{key}.json',dict(status='started',protocol_sha256=fingerprint,utc=utc()))
        def finish(key):
            """绑定本条件全部输出的哈希，恢复不能读到半个池或半张图板。"""
            record=read_json(run/'records'/f'{key}.json')
            paths=[run/'records'/f'{key}.json',run/record['feature_file'],run/'boards'/f'{key}.png']
            if (run/'focus'/f'{key}.png').exists():paths.append(run/'focus'/f'{key}.png')
            for asset in record['selection'].get('new_assets',[]):paths.append(Path(asset['path']))
            durable(run/'jobs'/f'{key}.json',dict(status='completed',protocol_sha256=fingerprint,
                files={str(p.relative_to(run)):sha256(p) for p in paths},utc=utc()))
        # 已完成恢复只校验，不覆盖原计算账本，也不加载模型或重新解码。
        if resume and (run/'completion.json').exists() and read_json(run/'completion.json')['status']=='completed_compute':
            keys={f'{q}__control' for q in CASES}|{f'{q}__{mode}' for q in SORT_CASES for mode in ('soft_fixed','soft_joint')}|{f'{q}__window_{h}' for q in WINDOW_CASES for h in (5,6)}
            if {p.stem for p in (run/'records').glob('*.json')}!=keys:raise ProtocolError('Completed condition set differs')
            for key in keys:
                if existing(key) is None:raise ProtocolError('Missing completed condition')
            for name,digest in protocol['historical_result_sha256'].items():
                if sha256(ROOT/name)!=digest:raise ProtocolError('Historical result changed')
            print('Verified all 29 conditions; no models loaded, no decoding or feature computation repeated.',flush=True)
            return run
        durable(run/'pid.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        try:
            from transformers import SiglipImageProcessor
            offline_environment();processor=SiglipImageProcessor.from_pretrained(read_json(ROOT/'configs/local_model_inventory.json')['models']['vision']['path'],local_files_only=True)
            # 步骤1：同环境控制重放，并冻结评价见证；选择函数没有见证/答案参数。
            for qid in CASES:
                status('control_and_sort',qid)
                source=read_json(SOURCE/'results'/f'{qid}.json');original=source['selection'];features=np.load(SOURCE/source['feature_file'],allow_pickle=False)
                if sha256(Path(original['video']['path']))!=source['source_video_sha256']:raise ProtocolError('Source video changed')
                chosen,trace=select_joint(original['candidates'],features,original['plan'],original['video'],protocol['config'])
                if chosen!=original['selected_indices'] or trace!=original['selection_trace']:raise ProtocolError(f'Control replay differs: {qid}')
                base={k:original[k] for k in ('video','candidates','initial_count','initial_scores','plan')}
                key=f'{qid}__control';record=existing(key)
                if record is None:
                    begin(key);record=save_condition(run,qid,'control',base,features,chosen,trace,original,processor,dict(control_replay_exact=True));finish(key)
                records.append(record)
                if qid in SORT_CASES:
                    for label,quotas in [('soft_fixed',original['selection_trace']['region_budgets']),('soft_joint',None)]:
                        key=f'{qid}__{label}';record=existing(key)
                        if record is None:
                            begin(key);chosen,trace=supported_selection(base['candidates'],features,base['plan'],base['video'],protocol['config'],quotas)
                            record=save_condition(run,qid,label,base,features,chosen,trace,original,processor,dict(fixed_original_quotas=quotas is not None));finish(key)
                        records.append(record)
                # 步骤2：扩窗分支复用所有旧特征，实际解码完整新区域，再补齐缺失视觉特征。
                if qid in WINDOW_CASES:
                    feature_cache={r['source_pts']:features[i] for i,r in enumerate(original['candidates'])}
                    for radius in (5,6):
                        status('window',qid);label=f'window_{radius}';key=f'{qid}__{label}';record=existing(key)
                        if record is not None:
                            loaded=np.load(run/record['feature_file'],allow_pickle=False)
                            feature_cache.update({r['source_pts']:loaded[i] for i,r in enumerate(record['selection']['candidates'])});records.append(record);continue
                        begin(key);cfg=dict(protocol['config'],window_radius_seconds=str(radius));n=original['initial_count'];coarse=original['candidates'][:n]
                        plan=build_regions(coarse,original['initial_scores'],original['video'],cfg)
                        added,assets,reports,timing=decode_regions(original['video'],coarse,plan,cfg,run/'fine_frames'/key)
                        missing=[a for a in assets if added[a['candidate_index']-n]['source_pts'] not in feature_cache]
                        if missing and scorer is None:
                            selected,states=available_gpus([requested_gpu]);os.environ['CUDA_VISIBLE_DEVICES']=selected[0]
                            import torch
                            torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(2027)
                            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
                            inventory=read_json(ROOT/'configs/local_model_inventory.json')['models']['blip']
                            for name,identity in inventory['files'].items():
                                if sha256(Path(inventory['path'])/name)!=identity['sha256']:raise ProtocolError('BLIP asset changed')
                            t=time.perf_counter();scorer=DensityScorer(inventory['path'])
                            from PIL import Image
                            dummy=Image.new('RGB',(384,384),'gray');scorer.visual_batch([dummy]);dummy.close()
                            durable(run/f'model_load_{os.getpid()}_{time.time_ns()}.json',dict(seconds=time.perf_counter()-t,gpu=selected[0],before=states,synthetic_counts=dict(scorer.counts)))
                        before=dict(scorer.counts) if scorer else dict(itm_batches=0,itm_frames=0,visual_batches=0,visual_frames=0)
                        from PIL import Image
                        timing.update(new_visual_preprocess_seconds=0.,new_visual_forward_seconds=0.)
                        for offset in range(0,len(missing),16):
                            batch=missing[offset:offset+16];images=[]
                            for asset in batch:
                                with Image.open(asset['path']) as image:images.append(image.convert('RGB'))
                            new_features,pre,forward=scorer.visual_batch(images)
                            for image in images:image.close()
                            timing['new_visual_preprocess_seconds']+=pre;timing['new_visual_forward_seconds']+=forward
                            for asset,vector in zip(batch,new_features):feature_cache[added[asset['candidate_index']-n]['source_pts']]=vector
                        candidates=coarse+added;vectors=np.stack([feature_cache[r['source_pts']] for r in candidates])
                        chosen,trace=select_joint(candidates,vectors,plan,original['video'],cfg)
                        selection=dict(video=original['video'],candidates=candidates,initial_count=n,initial_scores=original['initial_scores'],
                            plan=plan,refinement=reports,new_assets=assets,new_candidate_count=len(added))
                        after=scorer.counts if scorer else before
                        record=save_condition(run,qid,label,selection,vectors,chosen,trace,original,processor,
                            dict(window_radius=radius,missing_feature_frames=len(missing),reused_fine_features=len(added)-len(missing),
                                 compute={k:after[k]-before[k] for k in before},timings=timing));finish(key);records.append(record)
            # 步骤3：确认没有改写v1代码/历史输出，生成机器摘要与人工复核入口。
            if spec()!=protocol:raise ProtocolError('Source fingerprint changed during validation')
            for name,digest in protocol['historical_result_sha256'].items():
                if sha256(ROOT/name)!=digest:raise ProtocolError('Historical answer changed')
            summary=[]
            for record in records:
                s=record['selection'];summary.append(dict(question_id=record['question_id'],condition=record['condition'],
                    regions=len(s['plan']['regions']),refined_regions=sum(r['refine'] for r in s['plan']['regions']),
                    pool_size=len(s['candidates']),eligible_size=len(record['trace']['eligible_indices']),
                    selected_new=sum(i>=s['initial_count'] for i in record['selected_indices']),
                    region_budgets=record['trace']['region_budgets'],changed_slots=len(record['change']['added_pts']),
                    witnesses=record['witness_proxy'],extra=record['extra']))
            # 分条件账本可跨恢复求和，避免把本进程缓存命中误报成全轮零计算。
            names=('itm_batches','itm_frames','visual_batches','visual_frames')
            real_counts={k:sum(r['extra'].get('compute',{}).get(k,0) for r in records) for k in names}
            loads=[read_json(p) for p in run.glob('model_load_*.json')]
            synthetic_counts={k:sum(r['synthetic_counts'][k] for r in loads) for k in names}
            if real_counts['itm_batches'] or synthetic_counts['itm_batches']:raise ProtocolError('Unexpected new ITM computation')
            durable(run/'summary.json',dict(status='awaiting_frame_review',conditions=summary,qa_calls=0,scope_calls=0,query_calls=0,
                new_itm_calls=0,real_blip_counts=real_counts,synthetic_blip_counts=synthetic_counts,wall_seconds=time.perf_counter()-start,
                limitation='No semantic quality inferred from catalog membership; no QA accuracy measured'))
            blocks=['<!doctype html><meta charset="utf-8"><style>body{font:16px/1.7 sans-serif;max-width:1200px;margin:auto}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>局部选帧验证：不运行问答</h1><p>事后AI/用户复核，非盲审。绿色为入选，橙色为区域内未选，灰色为区域外。见证目录不穷尽证据。</p>']
            for qid in CASES:
                blocks.append(f'<h2 id="{qid}">{qid}</h2><pre>{html.escape(rows[qid]["question"])}</pre>')
                for record in [r for r in records if r['question_id']==qid]:
                    key=f'{qid}__{record["condition"]}';blocks.append(f'<details><summary>{record["condition"]}：替换{len(record["change"]["added_pts"])}，区域配额{record["trace"]["region_budgets"]}</summary><a href="records/{key}.json">完整轨迹</a><img loading="lazy" src="boards/{key}.png">')
                    if qid in FOCUS:blocks.append(f'<p>焦点候选（可含区域外原粗帧作为参照）：</p><img loading="lazy" src="focus/{key}.png">')
                    blocks.append('</details>')
            (run/'index.html').write_text(''.join(blocks));status('awaiting_frame_review')
            durable(run/'completion.json',dict(status='completed_compute',conditions=len(records),qa_calls=0,new_itm_calls=0,
                real_blip_counts=real_counts,synthetic_blip_counts=synthetic_counts,wall_seconds=time.perf_counter()-start,utc=utc()))
        except BaseException as error:
            durable(run/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),utc=utc(),
                blip_counts=None if scorer is None else scorer.counts));status('paused');raise
    return run
