"""exp07双模型M16/K16：等权覆盖单因素、同帧答案导入和安全后台执行。"""
import contextlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import time
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,rows,pilot_ids,utc
from videoqa_full.prepare import environment
from videoqa_budget32.frames import FrameProvider,validate_frames
from videoqa_methods.density_selection import frame_key
from videoqa_methods.prefix_coverage_budget import select_prefix_coverage
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_lmms_bridge.common import framework_identity,load_task,task_doc,POST_PROMPT
from .inputs import BASE,OLD_QWEN,LLAVA_BASE,LLAVA_UNIFORM,DEV,frozen
from . import run as runtime

SOURCE=ROOT/'outputs/full_videoqa/exp12_generalization__videomme2700__20261005_r1'
HISTORY=ROOT/'outputs/full_videoqa/rd12_exp07__qwen32__videomme2700__20260930_r1'
RELEASE=ROOT/'configs/releases/rd_1_2_exp07_qwen32_dev200_20260928.json'
METHOD='RD-1.2-exp07'
QPY,LPY=runtime.QPY,runtime.LPY
_VALIDATE,_WORKER,_PROCESS=runtime.validate,runtime.worker,runtime.process


def selection_from_prefix(pool,trace):
    """接口：从冻结等权前缀与配额恢复有序索引，不读取预测或标准答案。"""
    ids=[i for r in trace['regions'] for i in r['prefix'][:trace['region_budgets'][str(r['region_id'])]]]
    if len(ids)!=len(set(ids)) or len(ids)!=16:raise ValueError('M16 quota or uniqueness differs')
    return sorted(ids,key=lambda i:frame_key(pool['candidates'][i]))


def prepare_one(job):
    """接口：重放旧M32与新M16两目标，保存exp07输入/轨迹及唯一同帧划分。"""
    root,row=job;q=row['question_id'];started=time.perf_counter()
    paths=dict(pool32=BASE/'pools'/(q+'.json'),features=BASE/'features'/(q+'.npy'),
        historic=HISTORY/'selections/RD-1.2-exp07'/(q+'.json'),historic_trace=HISTORY/'traces'/(q+'.json'),
        pool16=SOURCE/'inputs/pools'/(q+'.json'),trace16=SOURCE/'inputs/traces'/(q+'.json'),
        input16=SOURCE/'inputs/m16'/(q+'.json'))
    sources={str(p):sha256(p) for p in paths.values()};dest=root/'inputs/checkpoints'/(q+'.json')
    if dest.exists():
        saved=read_json(dest)
        if saved['sources']!=sources:raise ValueError('Selection source changed')
        for rel,h in saved['files'].items():
            if sha256(root/rel)!=h:raise ValueError('Prepared asset changed')
        return saved
    features=np.load(paths['features'],allow_pickle=False);pool32=read_json(paths['pool32'])
    old_trace=read_json(paths['historic_trace']);historical=read_json(paths['historic'])
    # 步骤1：独立重放全量exp07 M32，原参照配额只作目标下界检查。
    control,ct=select_prefix_coverage(pool32['candidates'],features,pool32['plan'],pool32['video'],
        dict(frame_budget=32),old_trace['reference_quotas'],pool32['initial_scores'])
    if control!=historical['selected_indices']:raise ValueError('Historic M32 selection differs: '+q)
    for a,b in zip(ct['regions'],old_trace['regions']):
        if a['prefix']!=b['prefix'] or not np.allclose(a['coverage_curve'],b['coverage_curve'],atol=1e-6,rtol=0):
            raise ValueError('Historic M32 prefix or curve differs')
    pool=read_json(paths['pool16']);saved=read_json(paths['trace16']);parent=saved['prefix_trace']
    mapping=pool['original_candidate_indices'];f=features[mapping]
    loaded=time.perf_counter()
    # 步骤2：同一M16池重算等权目标，再复算exp12作为控制；仅最终配额允许不同。
    chosen,trace=select_prefix_coverage(pool['candidates'],f,pool['plan'],pool['video'],
        dict(frame_budget=16),parent['reference_quotas'],pool['initial_scores'])
    if chosen!=selection_from_prefix(pool,parent) or trace['region_budgets']!=parent['region_budgets']:
        raise ValueError('Saved M16 uniform selection differs')
    for a,b in zip(trace['regions'],parent['regions']):
        if a['prefix']!=b['prefix'] or not np.allclose(a['coverage_curve'],b['coverage_curve'],atol=1e-6,rtol=0):
            raise ValueError('M16 local sequence or curve differs')
    exp12,_=select_relevance_coverage(pool,f,trace);old=read_json(paths['input16'])
    if exp12!=old['selected_indices']:raise ValueError('Exp12 M16 control differs')
    selected=time.perf_counter();selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in chosen])
    validate_frames(selection,16);same=selection==old['selection']
    images,hashes=FrameProvider(selection).decode()
    for image in images:image.close()
    if same and hashes!=old['rgb_sha256']:raise ValueError('Same source RGB differs')
    files={}
    for rel,value in {
        'inputs/selections/'+q+'.json':dict(question_id=q,video_id=row['video_id'],method=METHOD,
            selection=selection,rgb_sha256=hashes,sources=sources,selected_indices=chosen,
            original_candidate_indices=[mapping[i] for i in chosen],same_as_exp12=same,configuration='M16/K16'),
        'inputs/traces/'+q+'.json':trace,
    }.items():
        frozen(root/rel,value);files[rel]=sha256(root/rel)
    record=dict(question_id=q,same_as_exp12=same,m32_control_exact=True,m16_control_exact=True,
        sources=sources,files=files,new_model_calls=0,timings=dict(asset_read_and_m32_control_seconds=loaded-started,
            m16_selection_seconds=selected-loaded,final_decode_seconds=time.perf_counter()-selected),
        total_seconds=time.perf_counter()-started)
    frozen(dest,record);return record


def prepare(root):
    """接口：绑定模型/环境/输入来源并冻结797/1903划分，任何差异暂停。"""
    from videoqa_budget32.prefix_full import release_check
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'prepare.lock'):
        release_check();source=read_json(SOURCE/'preflight.json')
        if read_json(SOURCE/'independent_completion_audit.json')['status']!='passed':raise ValueError('Source not accepted')
        for name in ['qwen16','llava16']:runtime.verify(SOURCE/name)
        paths=[RELEASE,SOURCE/'input_summary.json',SOURCE/'preflight.json',SOURCE/'independent_completion_audit.json']
        spec=dict(method=METHOD,configuration='M16/K16',question_ids=[r['question_id'] for r in rows()],
            same_expected=797,new_expected_per_model=1903,max_real_generations=3806,
            sources={str(p):sha256(p) for p in paths},new_blip_calls=0,new_classification_calls=0,new_query_calls=0)
        frozen(root/'preflight.json',dict(source,exp07_experiment=spec))
        frozen(root/'experiment.json',spec)
        if (root/'input_summary.json').exists():
            result=read_json(root/'input_summary.json')
            for p,h in result['checkpoints'].items():
                if sha256(p)!=h:raise ValueError('Completed input changed')
            return result
        # 步骤1：900视频及两模型字节再次核验，模型只读，CPU线程有界。
        for i,(vid,st) in enumerate(source['video_stats'].items(),1):
            if sha256(st['path'])!=source['video_hashes'][vid]:raise ValueError('Video bytes changed')
            if i%50==0:durable(root/'prepare_status.json',dict(status='video_hashing',completed=i,total=900,utc=utc()))
        for p,v in source['model_files'].items():
            if sha256(p)!=v['sha256']:raise ValueError('Model bytes changed')
        start=time.perf_counter();records=[]
        with ThreadPoolExecutor(max_workers=8) as executor:
            for r in executor.map(prepare_one,[(root,row) for row in rows()]):
                records.append(r);durable(root/'prepare_status.json',dict(status='selecting',completed=len(records),total=2700,utc=utc()))
        # 步骤2：划分只由完整输入身份决定，不读取模型对错来决定导入。
        same=[r['question_id'] for r in records if r['same_as_exp12']];new=[r['question_id'] for r in records if not r['same_as_exp12']]
        if (len(same),len(new))!=(797,1903):raise ValueError('Frozen partition differs; no call-budget expansion')
        result=dict(status='passed',questions=2700,m32_control_exact=2700,m16_control_exact=2700,
            imported_ids=same,new_ids=new,new_model_calls=0,
            checkpoints={str(root/'inputs/checkpoints'/(r['question_id']+'.json')):sha256(root/'inputs/checkpoints'/(r['question_id']+'.json')) for r in records})
        frozen(root/'input_summary.json',result)
        frozen(root/'input_execution.json',dict(wall_seconds=time.perf_counter()-start,
            source_classification='Cached inputs; no independent raw-video selection E2E',utc=utc()))
        return result


def cpu_import_check(root,kind):
    """接口：797题独立CPU重建像素和提示Token，不加载问答权重或生成答案。"""
    offline_environment();root=Path(root);directory=root/'import_checks'/kind
    source_stage='qwen16' if kind=='qwen' else 'llava16';ids=read_json(root/'input_summary.json')['imported_ids']
    spec=runtime.verify(SOURCE/source_stage);data={r['question_id']:r for r in rows()}
    import torch
    torch.set_num_threads(2)
    if kind=='qwen':
        from transformers import AutoProcessor
        from videoqa_full.state import QWEN_PATH
        from videoqa_budget32.observe import tensor_hash
        processor=AutoProcessor.from_pretrained(QWEN_PATH,local_files_only=True,use_fast=False,
            min_pixels=100352,max_pixels=602112,size=dict(shortest_edge=100352,longest_edge=602112))
    else:
        from transformers import AutoTokenizer
        from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from videoqa_lmms_bridge.llava_adapter import tensor_hash
        processor=SigLipImageProcessor();tokenizer=AutoTokenizer.from_pretrained(
            '/home/models/lmms-lab/LLaVA-Video-7B-Qwen2',local_files_only=True)
    start=time.perf_counter();files={}
    # 步骤1：在每题真正导入前，从新选择独立解码并按官方CPU处理器重建输入。
    for number,q in enumerate(ids,1):
        path=directory/(q+'.json');source_path=SOURCE/source_stage/'results/RD-1.2-exp12'/(q+'.json')
        input_path=root/'inputs/selections'/(q+'.json')
        sources={str(source_path):sha256(source_path),str(input_path):sha256(input_path),
                 str(SOURCE/source_stage/'protocol.json'):sha256(SOURCE/source_stage/'protocol.json')}
        if path.exists():
            saved=read_json(path)
            if saved['sources']!=sources or saved['status']!='passed':raise ValueError('Imported input check changed')
            files[q]=sha256(path);continue
        old=runtime.validate(SOURCE/source_stage,data[q],spec);item=read_json(input_path)
        if not item['same_as_exp12'] or item['selection']!=old['selection']:raise ValueError('Import source not identical')
        images,rgb=FrameProvider(item['selection']).decode();observed=old['answer']['observed']
        if rgb!=item['rgb_sha256'] or rgb!=old['rgb_sha256']:raise ValueError('Import RGB changed')
        context=load_task().videomme_doc_to_text(task_doc(data[q]),{'post_prompt':POST_PROMPT})
        if kind=='qwen':
            prompt=processor.apply_chat_template([dict(role='system',content='You are a helpful assistant.'),
                dict(role='user',content=[dict(type='video'),dict(type='text',text=context)])],tokenize=False,add_generation_prompt=True)
            texts=[prompt]
            inputs=processor(text=texts,videos=[np.stack([np.asarray(im.convert('RGB')) for im in images])],
                fps=2.0,images_kwargs={'size':dict(shortest_edge=100352,longest_edge=602112)},truncation=False,return_tensors='pt')
            prompt=texts[0]
            pixels=tensor_hash(inputs['pixel_values_videos']);tokens=inputs['input_ids'][0].tolist()
            expected=observed['processor_pixel_sha256']
            if inputs['video_grid_thw'].tolist()!=observed['video_grid_thw']:raise ValueError('Import grid changed')
        else:
            conv=conv_templates['qwen_1_5'].copy();conv.append_message(conv.roles[0],'<image>\n'+context);conv.append_message(conv.roles[1],None)
            prompt=conv.get_prompt();tokens=tokenizer_image_token(prompt,tokenizer,-200)
            pixels=tensor_hash(processor.preprocess(images,return_tensors='pt')['pixel_values'].to(dtype=torch.bfloat16))
            expected=observed['pixel_sha256']
        for im in images:im.close()
        if pixels!=expected or tokens!=observed['input_token_ids'] or prompt!=observed['prompt']:
            raise ValueError('Import processor pixel or prompt tokens differ: '+q)
        frozen(path,dict(status='passed',question_id=q,sources=sources,rgb_sha256=rgb,
            processor_pixel_sha256=pixels,input_token_ids=tokens,prompt=prompt,new_model_calls=0))
        files[q]=sha256(path)
        if number%25==0:durable(root/'import_status.json',dict(kind=kind,completed=number,total=797,utc=utc()))
    frozen(root/('import_'+kind+'_acceptance.json'),dict(status='passed',questions=797,files=files,new_model_calls=0))
    if not (root/('import_'+kind+'_execution.json')).exists():
        frozen(root/('import_'+kind+'_execution.json'),dict(wall_seconds=time.perf_counter()-start,utc=utc()))


def prepare_stage(root,stage,gpus):
    """接口：两模型各冻结797导入＋1903新调用，阶段及模型ID严格隔离。"""
    root=Path(root);run=root/stage;run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():
            spec=runtime.verify(run);import_answers(root,stage,spec);return spec
        kind='qwen' if stage=='qwen16' else 'llava';old_spec=runtime.verify(SOURCE/stage)
        inputs=read_json(root/'input_summary.json');check=read_json(root/('import_'+kind+'_acceptance.json'))
        if check['status']!='passed' or len(check['files'])!=797:raise ValueError('Import acceptance incomplete')
        exports={}
        for row in rows():
            q=row['question_id'];p=root/'inputs/selections'/(q+'.json');rel=f'selections/{METHOD}/{q}.json'
            dest=run/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                if sha256(dest)!=sha256(p):raise ValueError('Stage source changed')
            else:shutil.copy2(p,dest)
            exports[rel]=sha256(dest)
        code=dict(old_spec['code_sha256'])
        for n in ['src/videoqa_generalization/exp07_budget16.py','scripts/run_exp07_budget16_full.py']:
            code[n]=sha256(ROOT/n)
        spec=dict(old_spec,methods=[METHOD],configuration='M16/K16 exp07 uniform-reference coverage',
            max_calls=1903,max_calls_per_method=1903,imported_count=797,allowed_gpus=gpus,
            selection_exports=exports,code_sha256=code,preflight_sha256=sha256(root/'preflight.json'),
            input_summary_sha256=sha256(root/'input_summary.json'),import_acceptance_sha256=sha256(root/('import_'+kind+'_acceptance.json')),
            source_run=str(SOURCE/stage),source_protocol_sha256=sha256(SOURCE/stage/'protocol.json'),
            imported_ids=inputs['imported_ids'],new_ids=inputs['new_ids'],pilot_question_ids=pilot_ids(rows()),utc=utc())
        for n,h in code.items():
            if sha256(ROOT/n)!=h:raise ValueError('Frozen dependency changed')
            dest=run/'code_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
        frozen(run/'protocol.json',spec)
        import_answers(root,stage,spec)
        return spec


def import_answers(root,stage,spec):
    """接口：恢复缺失的已验收导入结果；从原返回重建，不消耗新问答或覆盖冲突。"""
    root=Path(root);run=root/stage;kind=spec['model_kind']
    check_path=root/('import_'+kind+'_acceptance.json');check=read_json(check_path)
    if sha256(check_path)!=spec['import_acceptance_sha256']:raise ValueError('Import acceptance changed')
    runtime.verify(SOURCE/stage)
    # 步骤1：CPU像素/Token验收与旧正式账本共同绑定，不能只以文件存在代替正确性。
    for q in spec['imported_ids']:
        old_path=SOURCE/stage/'results/RD-1.2-exp12'/(q+'.json');old=read_json(old_path)
        verified=root/'import_checks'/kind/(q+'.json')
        if sha256(verified)!=check['files'][q]:raise ValueError('Import verification changed')
        for p,h in read_json(verified)['sources'].items():
            if sha256(p)!=h:raise ValueError('Imported input source changed')
        origins={str(SOURCE/stage/folder/'RD-1.2-exp12'/(q+'.json')):sha256(SOURCE/stage/folder/'RD-1.2-exp12'/(q+'.json'))
                 for folder in ['results','calls','raw_generations']}
        value=dict(old,method=METHOD,protocol_sha256=sha256(run/'protocol.json'),imported=True,
            import_sources=origins,import_check=str(verified),import_check_sha256=sha256(verified),
            source_result=str(old_path),historical_protocol_sha256=old['protocol_sha256'])
        frozen(run/'results'/METHOD/(q+'.json'),value)


def validate(run,row,spec):
    """接口：新生成复用原完整校验；797导入核对像素验收、原账本与任务计分。"""
    run=Path(run);q=row['question_id'];r=read_json(run/'results'/METHOD/(q+'.json'))
    if not r.get('imported'):
        if q not in spec['new_ids']:raise ValueError('New generation outside frozen changed subset')
        return _VALIDATE(run,row,spec)
    if q not in spec['imported_ids']:raise ValueError('Unexpected imported question')
    item=read_json(run/'selections'/METHOD/(q+'.json'))
    if not item['same_as_exp12'] or item['selection']!=r['selection'] or item['rgb_sha256']!=r['rgb_sha256']:
        raise ValueError('Imported source differs')
    if r['method']!=METHOD or r['model']!=spec['model'] or r['protocol_sha256']!=sha256(run/'protocol.json'):
        raise ValueError('Imported model or protocol differs')
    for k in ['question_id','video_id','question','options','stratum']:
        if r[k]!=row[k]:raise ValueError('Imported question differs')
    for p,h in r['import_sources'].items():
        if sha256(p)!=h:raise ValueError('Imported source record changed')
    if sha256(r['import_check'])!=r['import_check_sha256']:raise ValueError('Pixel acceptance changed')
    metric=load_task().videomme_process_results(task_doc(row),[r['answer']['raw_output']])['videomme_perception_score']
    if metric!=r['answer']['framework_metric'] or r['correct']!=(metric['pred_answer']==metric['answer']):
        raise ValueError('Imported score differs')
    return r


@contextlib.contextmanager
def evaluation_scope():
    """接口：仅本进程注入导入校验和spawn入口，退出恢复绑定，不改旧文件或模型计算。"""
    old_validate,old_worker,old_process=runtime.validate,runtime.worker,runtime.process
    runtime.validate=validate;runtime.worker=worker;runtime.process=process
    try:yield
    finally:runtime.validate=old_validate;runtime.worker=old_worker;runtime.process=old_process


def process(run,row,backend,adapter,gpu,spec):
    """接口：导入条目即使结果缺失也禁止新生成；改变输入题沿用原一次调用逻辑。"""
    q=row['question_id'];path=Path(run)/'results'/METHOD/(q+'.json')
    if path.exists():validate(run,row,spec);return
    if q not in spec['new_ids']:raise ValueError('Missing imported result; generation forbidden')
    return _PROCESS(run,row,backend,adapter,gpu,spec)


def worker(*args):
    """接口：spawn子进程显式恢复同一验收绑定，问答执行仍用冻结原后端。"""
    with evaluation_scope():_WORKER(*args)


def execute(run):
    """接口：后台协调复用原调度/恢复，导入题不进入GPU队列。"""
    with evaluation_scope():runtime.execute(run);runtime.audit(run)


def summarize(root):
    """接口：完整5400题后比较两模型的单因素得失，导入与新问答分别统计。"""
    from .report import reference
    from videoqa_full.report import compare
    from videoqa_runtime.baseline_report import wilson,distribution
    root=Path(root);data=rows();dev=read_json(DEV)['rows'];dq={r['question_id'] for r in dev};dv={r['video_id'] for r in dev}
    partition=read_json(root/'input_summary.json');same=set(partition['imported_ids']);changed=set(partition['new_ids'])
    subsets={'all2700':data,'development200':[r for r in data if r['question_id'] in dq],
        'other2500':[r for r in data if r['question_id'] not in dq],
        'outside_development_videos2100':[r for r in data if r['video_id'] not in dv],
        'same_input797':[r for r in data if r['question_id'] in same],
        'changed_input1903':[r for r in data if r['question_id'] in changed]}
    groups={};reports={};all_calls=[];file_sources={}
    # 步骤1：独立逐题校验，主比较使用全量，变化子集仅补充机制归因。
    for stage in ['qwen16','llava16']:
        run=root/stage;spec=runtime.verify(run);audit=read_json(run/'completion_audit.json')
        if audit['status']!='passed' or audit['new_calls']!=1903:raise ValueError('Stage completion differs')
        values={r['question_id']:validate(run,r,spec) for r in data};groups[stage]=values
        refs={'RD-1.2-exp12':{r['question_id']:read_json(SOURCE/stage/'results/RD-1.2-exp12'/(r['question_id']+'.json')) for r in data}}
        if stage=='qwen16':refs.update({m:reference(OLD_QWEN,m,data) for m in ['BASE-Uniform','BASE-BLIP-TopK','RD-1.2']})
        else:
            refs.update({m:reference(LLAVA_BASE,m,data) for m in ['BASE-BLIP-TopK','RD-1.2']})
            refs['BASE-Uniform']=reference(LLAVA_UNIFORM,'BASE-Uniform',data,uniform=True)
        common=[r for r in data if all(len(g[r['question_id']]['selection']['selected_frames'])==16 for g in [values,*refs.values()])]
        layers=dict(subsets,common_full_budget=common);metrics={}
        for name,rs in layers.items():
            metrics[name]={}
            for layer in ['all','short','medium','long']:
                part=[r for r in rs if layer=='all' or r['stratum']==layer]
                if not part:continue
                c=sum(values[r['question_id']]['correct'] for r in part)
                metrics[name][layer]=dict(n=len(part),correct=c,accuracy=c/len(part),wilson_95=wilson(c,len(part)))
        pairs={name:{subset:compare(rs,values,group) for subset,rs in layers.items()} for name,group in refs.items()}
        for q in same:
            if values[q]['answer']!=refs['RD-1.2-exp12'][q]['answer']:raise ValueError('Imported output differs')
        fresh=[values[q] for q in partition['new_ids']];perf={}
        for layer in ['all','short','medium','long']:
            part=[v for v in fresh if layer=='all' or v['stratum']==layer];timing={};memory={}
            if part:
                for key in sorted(set.intersection(*(set(v['timings']) for v in part))):timing[key]=distribution([v['timings'][key] for v in part])
                for key in ['qa_seconds','generation_seconds','peak_allocated_gib','peak_reserved_gib']:
                    nums=[v['answer']['observed'][key] for v in part if key in v['answer']['observed']]
                    if nums:(memory if 'gib' in key else timing)[key]=distribution(nums)
            perf[layer]=dict(n=len(part),timings=timing,memory=memory)
        tasks={}
        for task in sorted({r['task_type'] for r in data}):
            rs=[r for r in data if r['task_type']==task];c=sum(values[r['question_id']]['correct'] for r in rs)
            tasks[task]=dict(n=len(rs),correct=c,accuracy=c/len(rs))
        calls=list((run/'calls'/METHOD).glob('*.json'))
        if {p.stem for p in calls}!=changed:raise ValueError('Actual calls differ from frozen partition')
        all_calls.extend(read_json(p)['call_id'] for p in calls)
        workers=[read_json(p) for p in (run/'workers').glob('*.json')]
        reports[stage]=dict(metrics=metrics,comparisons=pairs,official_tasks=tasks,performance=perf,
            imported_answers=797,new_calls=1903,parse_failures=[q for q,v in values.items() if v['answer']['parsed_answer'] is None],
            worker_load_seconds=distribution([v['load_seconds'] for v in workers]),
            worker_warmup_seconds=distribution([v['warmup_seconds'] for v in workers]),
            worker_synthetic_generations=sum(v['synthetic_generations'] for v in workers),execution=read_json(run/'execution.json'))
        for q in values:
            path=run/'results'/METHOD/(q+'.json');file_sources[str(path)]=sha256(path)
    # 步骤2：跨模型实际源帧/RGB一致；新调用ID共3806且唯一。
    for q in groups['qwen16']:
        a,b=groups['qwen16'][q],groups['llava16'][q]
        if a['selection']!=b['selection'] or a['rgb_sha256']!=b['rgb_sha256']:raise ValueError('Cross-model source differs')
    if len(all_calls)!=3806 or len(set(all_calls))!=3806:raise ValueError('Global generation count differs')
    result=dict(status='completed',method=METHOD,configuration='M16/K16',new_answer_calls=3806,
        imported_answers=1594,source_rgb_exact=2700,reports=reports,source_files=file_sources,utc=utc())
    durable(root/'summary.json',result)
    # 步骤3：派生报告完整归档，不覆写既有人工review内容。
    lines=['# exp07双模型16帧全量对照结果','',
        '本轮仅恢复等权覆盖目标；两模型各2700题，797同帧导入和1903新答。3806次新问答，同帧核验不包含模型前向。','',
        '| 模型 | exp07正确/精度 | exp12 | 均匀 |','|---|---:|---:|---:|']
    for stage,r in reports.items():
        m=r['metrics']['all2700']['all'];base=read_json(SOURCE/'summary.json')['reports'][stage]['metrics']['all2700']['all']
        uniform_count=sum(refs_value['correct'] for refs_value in (reference(OLD_QWEN,'BASE-Uniform',data).values() if stage=='qwen16' else reference(LLAVA_UNIFORM,'BASE-Uniform',data,uniform=True).values()))
        lines.append(f'| {stage} | {m["correct"]}/2700（{m["accuracy"]*100:.2f}%） | {base["correct"]} | {uniform_count} |')
    for stage,r in reports.items():
        lines+=['',f'## {stage}配对比较','']
        for baseline,pair in r['comparisons'].items():
            v=pair['all2700'];lo,hi=v['stratified_video_cluster_bootstrap_95']
            lines.append(f'- 对{baseline}：{v["difference_pp"]:+.3f}个百分点，区间[{lo:.3f},{hi:.3f}]；新增正确{len(v["cells"]["left_only"])}、退化{len(v["cells"]["right_only"])}。')
    lines+=['','完整时长、官方题型、子集及全部配对题号见运行目录summary.json；新1903条性能与导入成本另列。数据曾被观察，不称盲测；本轮据实结束，不自动开始新调参。']
    text='\n'.join(lines)+'\n';(root/'report.md').write_text(text,encoding='utf-8')
    archive=ROOT/'docs/experiments/video_mme_full/2026-10-05/exp07_budget16/实验结果.md'
    if archive.exists() and archive.read_text(encoding='utf-8')!=text:raise ValueError('Reviewed report differs')
    archive.parent.mkdir(parents=True,exist_ok=True);archive.write_text(text,encoding='utf-8')
    return result


def synthetic(root,kind,allowed):
    """接口：新入口1/11/16帧合成验收；失败尝试保留，不自动重复未确认调用。"""
    from videoqa_runtime.baseline_run import available_gpus
    root=Path(root);run=root/'synthetic'/kind
    if (run/'acceptance.json').exists():
        accepted=read_json(run/'acceptance.json')
        if accepted['status']!='passed':raise ValueError('Synthetic status differs')
        return
    if list((run/'calls').glob('*.json')):raise ValueError('Prior synthetic attempt requires audit')
    while True:
        try:free,_=available_gpus(allowed)
        except RuntimeError:free=[]
        if free:break
        durable(run/'status.json',dict(status='waiting_resources',utc=utc()));time.sleep(15)
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=free[0];offline_environment()
    import torch
    from PIL import Image
    from videoqa_lmms_bridge.bridge import synthetic_row
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if kind=='qwen':
        from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
        from videoqa_budget32.qwen_adapter import FrozenInputQwen
        from videoqa_budget32.observe import observe
        backend=QwenFA2VideoBackend();adapter=FrozenInputQwen(backend)
    else:
        from videoqa_runtime.llava_backend import LlavaBackend
        from videoqa_lmms_bridge.llava_adapter import FrozenInputLlavaVid
        from .llava_observe import observe
        backend=LlavaBackend();adapter=FrozenInputLlavaVid(backend)
    checks=[]
    # 步骤1：合成也在调用前记账，与3806次真实问答严格分开。
    for n,size in [(1,(320,240)),(11,(720,1280)),(16,(896,672))]:
        path=run/'calls'/(str(n)+'.json');durable(path,dict(status='started',frames=n,utc=utc()))
        images=[Image.new('RGB',size,(40,i*7,150)) for i in range(n)];state={}
        identity=dict(synthetic_frames=n,kind=kind)
        if kind=='qwen':result=observe(backend,adapter,images,synthetic_row(n),state,run/'raw'/(str(n)+'.json'),identity)
        else:
            s=dict(video=dict(path='synthetic.mp4',duration_seconds=float(n)),selected_frames=[dict(timestamp_seconds=float(i)) for i in range(n)])
            result=observe(backend,adapter,images,s,synthetic_row(n),state,run/'raw'/(str(n)+'.json'),identity)
        for im in images:im.close()
        if state!={'started':True,'returned':True}:raise ValueError('Synthetic generation uncertain')
        durable(path,dict(status='completed',frames=n,size=list(size),state=state,result=result,utc=utc()))
        checks.append(dict(n=n,size=list(size),call_sha256=sha256(path)))
    # 步骤2：只检查工程约束；不根据合成回答选择配置。
    frozen(run/'acceptance.json',dict(status='passed',checks=checks,real_generations=0,
        synthetic_generations=3,load_report=backend.load_report))


def audit_all(root):
    """接口：完成后实际执行安全恢复，13012份答案文件不变，新增生成为0。"""
    root=Path(root);folders=['results','calls','raw_generations']
    before={str(p):sha256(p) for stage in ['qwen16','llava16'] for folder in folders for p in (root/stage/folder).glob('*/*.json')}
    if len(before)!=13012:raise ValueError('Answer record population differs')
    for stage in ['qwen16','llava16']:execute(root/stage)
    after={str(p):sha256(p) for stage in ['qwen16','llava16'] for folder in folders for p in (root/stage/folder).glob('*/*.json')}
    if before!=after:raise ValueError('Completed recovery changed answers')
    durable(root/'completion_audit.json',dict(status='passed',questions_per_model=2700,new_calls=3806,
        imported=1594,unchanged_files=13012,extra_generations=0,utc=utc()))
