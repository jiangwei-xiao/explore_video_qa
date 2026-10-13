"""评分器全量准备：原候选身份核验、控制重放、新窗口补查及混合特征诊断。"""
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, save_array, rows, digest, utc
from videoqa_audit.framework_contraction import BASE, locations
from videoqa_methods.budget_contract import budget_config, select_exp07
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_methods.density_selection import build_regions
from videoqa_runtime.baseline_selection import select_topk

METHODS = ('BASE-BLIP-TopK', 'RD-1.2-exp07', 'RD-1.2-exp12')
PROFILES = {'qwen32': 32, 'llava8': 8}
PYTHON = '/App/conda/envs/explore_video_qa_retrieval/bin/python'
MODEL = ROOT / '.cache/retrieval_models/blip2-itm-vit-g'
_SCORER = None


def source_keys(frames):
    """接口：跨候选池只按源帧号和真实PTS比较，不依赖局部候选编号。"""
    return [(f['source_frame_index'], f['source_pts']) for f in frames]


def completed(root, name, q, sources):
    """接口：完成检查点必须绑定输入和全部输出；恢复不重新评分或选择。"""
    p = Path(root) / 'checkpoints' / name / (q+'.json')
    if not p.exists(): return False
    c = read_json(p)
    if c['sources'] != sources: raise ValueError('Preparation sources changed: '+q)
    for rel, h in c['files'].items():
        if sha256(Path(root)/rel) != h: raise ValueError('Preparation output changed: '+rel)
    return True


def checkpoint(root, name, q, sources, paths, extra=None):
    """接口：特征与JSON均落盘后标记完成，防止只有半份输入时进入问答。"""
    root = Path(root)
    durable(root/'checkpoints'/name/(q+'.json'), dict(sources=sources,
        files={str(Path(p).relative_to(root)):sha256(p) for p in paths}, **(extra or {}), utc=utc()))


def control_one(job):
    """接口：同一旧池重放exp07/12，覆盖两个工作点；不调用模型。"""
    from videoqa_audit.region_scope_runtime import parent_result
    root, row = Path(job[0]), job[1]; q = row['question_id']; reports=[]; sources={}
    existing=root/'controls'/(q+'.json')
    if existing.exists():
        saved=read_json(existing)
        for path,h in saved['sources'].items():
            if sha256(path)!=h:raise ValueError('Completed control source changed')
        return str(existing.relative_to(root)),sha256(existing)
    for profile, k in PROFILES.items():
        pp, fp, _, _ = locations(q,k)
        from videoqa_audit.framework_contraction import check_binding,RUN8
        check_binding(fp,BASE/'selection_checkpoints'/(q+'.json'),BASE)
        if k==32:check_binding(pp,BASE/'selection_checkpoints'/(q+'.json'),BASE)
        else:check_binding(pp,RUN8/'inputs/checkpoints'/(q+'.json'),RUN8)
        for p in (pp,fp): sources[str(p)]=sha256(p)
        pool=read_json(pp); f=np.load(fp,allow_pickle=False)
        if k==8: f=f[pool['original_candidate_indices']]
        a, trace=select_exp07(pool,f,k); b,_=select_relevance_coverage(pool,f,trace)
        for m,ids in zip(METHODS[1:],(a,b)):
            old=parent_result(profile,q,m); sources[str(old)]=sha256(old)
            expected=read_json(old)['selection']['selected_frames']
            if source_keys([pool['candidates'][i] for i in ids]) != source_keys(expected):
                raise ValueError('S1 control mismatch: '+profile+'/'+m+'/'+q)
        reports.append(dict(profile=profile, frame_budget=k, methods=2, status='passed'))
    p=root/'controls'/(q+'.json');durable(p,dict(question_id=q,reports=reports,sources=sources,new_model_calls=0))
    return str(p.relative_to(root)),sha256(p)


def controls(root):
    """接口：下载等待期间可独立执行全部S1控制，完成来源一致时正式预检直接引用。"""
    root=Path(root);data=rows();values={}
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn')) as executor:
        for i,(rel,h) in enumerate(executor.map(control_one,[(str(root),r) for r in data]),1):
            values[rel]=h
            if i%25==0:durable(root/'preflight_status.json',dict(stage='S1_control',completed=i,total=2700,utc=utc()))
    durable(root/'controls_summary.json',dict(status='passed',questions=2700,selection_replays=10800,files=values,new_model_calls=0))
    return values


def preflight(root):
    """接口：全量控制、视频/权重字节、模型revision和新旧指纹在评分前冻结。"""
    import shutil
    from videoqa_methods.retrieval_frontends import WEIGHT_SHA256,REVISION,MODEL_ID
    from videoqa_audit.framework_contraction import RUN32,RUN8
    from videoqa_full.prepare import environment
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    if (root/'preflight.json').exists():
        spec=read_json(root/'preflight.json')
        for p,h in spec['sources'].items():
            if sha256(p)!=h:raise ValueError('Preflight source changed')
        for rel,h in spec['controls'].items():
            if sha256(root/rel)!=h:raise ValueError('Control changed')
        return spec
    # 步骤1：新权重绑定官方LFS摘要，不接受中转过程中不完整或被替换的文件。
    if sha256(MODEL/'model.safetensors')!=WEIGHT_SHA256:raise ValueError('Official weight hash mismatch')
    model_files={str(p):sha256(p) for p in sorted(MODEL.iterdir()) if p.is_file()}
    sources=dict(model_files)
    for r in (BASE,RUN32,RUN8):sources[str(r/'protocol.json')]=sha256(r/'protocol.json')
    old=read_json(BASE/'protocol.json');data=rows()
    if len(data)!=2700 or len({r['video_id'] for r in data})!=900:raise ValueError('Wrong dataset')
    for i,(vid,st) in enumerate(old['video_stats'].items(),1):
        h=sha256(st['path'])
        if h!=old['video_hashes'][vid]:raise ValueError('Source video changed: '+vid)
        sources[st['path']]=h
        if i%50==0:durable(root/'preflight_status.json',dict(stage='video_hash',completed=i,total=900,utc=utc()))
    # 步骤2：原前端全量重放是运行正确性门槛，不按新评分效果决定是否继续。
    control_exports=controls(root)
    if shutil.disk_usage(root).free<100*1024**3:raise ValueError('Insufficient asset space')
    fixed=['src/videoqa_methods/retrieval_frontends.py','scripts/run_scorer_comparison_full.py',
        'src/videoqa_methods/budget_contract.py','src/videoqa_methods/density_selection.py',
        'src/videoqa_methods/density_pipeline.py','src/videoqa_methods/coarse_eligibility.py',
        'src/videoqa_methods/weighted_seed_support.py','src/videoqa_methods/prefix_coverage_budget.py',
        'src/videoqa_methods/relevance_coverage_budget.py','configs/retrieval_density_v1.json',
        'src/videoqa_budget32/frames.py','src/videoqa_full/state.py','src/videoqa_budget32/observe.py',
        'src/videoqa_budget32/qwen_adapter.py','src/videoqa_generalization/llava_observe.py',
        'src/videoqa_lmms_bridge/common.py','src/videoqa_lmms_bridge/llava_adapter.py',
        'src/videoqa_runtime/llava_backend.py','src/videoqa_full/qwen_fa2_backend.py',
        'src/videoqa_full/qwen_backend.py','src/videoqa_generalization/llava_budget8.py']
    code={str(p.relative_to(ROOT)):sha256(p) for p in list((ROOT/'src/videoqa_scorer_comparison').glob('*.py'))+
          [ROOT/p for p in fixed]}
    for rel in code:
        dest=root/'code_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    spec=dict(scorer_id='S2-BLIP-2-ITM',model_id=MODEL_ID,revision=REVISION,model_files=model_files,
        sources=sources,controls=control_exports,code_sha256=code,environment=environment(PYTHON),
        methods=list(METHODS),profiles=PROFILES,question_ids=[r['question_id'] for r in data],
        max_new_final_qa=16200,precision='float32',batch_size=16,tf32=False,
        scope='S2 ITM plus S2 pure visual CLS; not score-only',utc=utc())
    durable(root/'preflight.json',spec);return spec


def initialize_scorer(root, gpu,stage='scorer'):
    """接口：一卡常驻S2；每卡合成与全量无截断核验独立记录，不产生真实答案。"""
    global _SCORER
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from videoqa_methods.retrieval_frontends import Blip2ItmFrontend
    t=time.perf_counter();_SCORER=Blip2ItmFrontend(MODEL)
    loaded=time.perf_counter()-t;t=time.perf_counter()
    acceptance=_SCORER.synthetic_acceptance()
    counts=[int(_SCORER.tokenize(r['question'])['input_ids'].shape[1]) for r in rows()]
    durable(Path(root)/'workers'/(stage+'_'+gpu+'_'+str(os.getpid())+'.json'),dict(stage=stage,gpu=gpu,pid=os.getpid(),
        sid=os.getsid(0),load_seconds=loaded,warmup_seconds=time.perf_counter()-t,
        acceptance=acceptance,text_token_min=min(counts),text_token_max=max(counts),
        resident_bytes=torch.cuda.memory_allocated(),utc=utc()))


def coarse_one(root,row):
    """接口：沿原视频逐批扫描评分，与历史粗候选身份完全核对，内存最多16RGB。"""
    from videoqa_methods.density_pipeline import scan_original
    root=Path(root);q=row['question_id'];oldp=BASE/'pools'/(q+'.json')
    sources={str(oldp):sha256(oldp),'preflight':sha256(root/'preflight.json')}
    if completed(root,'coarse',q,sources):return
    old=read_json(oldp);before=dict(_SCORER.counts);t=time.perf_counter()
    _SCORER.torch.cuda.reset_peak_memory_stats()
    pool,f,timing=scan_original(old['video']['path'],row['question'],_SCORER)
    if pool['video']!=old['video'] or pool['candidates']!=old['candidates'][:old['initial_count']]:
        raise ValueError('S2 scan differs from frozen initial identity: '+q)
    pool.update(initial_count=len(pool['candidates']),initial_scores=pool.pop('scores'),scorer_id='S2-BLIP-2-ITM',
                feature_backbone='Salesforce/blip2-itm-vit-g',feature_type='pure_vision_CLS_L2')
    pp=root/'coarse/pools'/(q+'.json');fp=root/'coarse/features'/(q+'.npy')
    durable(pp,pool);save_array(fp,f)
    timing.update(wall_seconds=time.perf_counter()-t,
        model_calls={k:_SCORER.counts[k]-before[k] for k in before},
        peak_allocated_gib=_SCORER.torch.cuda.max_memory_allocated()/2**30)
    checkpoint(root,'coarse',q,sources,[pp,fp],dict(timing=timing))


def select_pool(pool,features,k):
    """接口：共享exp07前缀构造exp12，唯一差别仍为粗参考覆盖加权。"""
    a,trace=select_exp07(pool,features,k);b,weighted=select_relevance_coverage(pool,features,trace)
    return ({METHODS[0]:select_topk(pool['candidates'][:pool['initial_count']],pool['initial_scores'],k),
        METHODS[1]:[pool['candidates'][i] for i in a],METHODS[2]:[pool['candidates'][i] for i in b]},
        dict(exp07=trace,exp12=weighted))


def refine_one(root,row):
    """接口：两个预算各按新种子构建合法细池；临时PNG编码后安全释放，不长期占盘。"""
    from PIL import Image
    from videoqa_methods.density_pipeline import decode_regions
    root=Path(root);q=row['question_id'];cp=root/'coarse/pools'/(q+'.json');cf=root/'coarse/features'/(q+'.npy')
    coarse=read_json(cp);coarse_f=np.load(cf,allow_pickle=False)
    for profile,k in PROFILES.items():
        sources={str(cp):sha256(cp),str(cf):sha256(cf),'preflight':sha256(root/'preflight.json')}
        if completed(root,profile,q,sources):continue
        start=time.perf_counter();cfg=budget_config(k)
        before=dict(_SCORER.counts);_SCORER.torch.cuda.reset_peak_memory_stats()
        plan=build_regions(coarse['candidates'],coarse['initial_scores'],coarse['video'],cfg)
        assets_dir=root/'temporary_rgb'/profile/q
        added,assets,refinement,timing=decode_regions(coarse['video'],coarse['candidates'],plan,cfg,assets_dir)
        blocks=[coarse_f];visual_pre=visual_forward=0.
        for offset in range(0,len(assets),16):
            images=[]
            for asset in assets[offset:offset+16]:
                with Image.open(asset['path']) as image:images.append(image.convert('RGB'))
            try:v,pre,forward=_SCORER.visual_batch(images)
            finally:
                for image in images:image.close()
            blocks.append(v);visual_pre+=pre;visual_forward+=forward
        f=np.concatenate(blocks,axis=0)
        pool=dict(coarse,candidates=coarse['candidates']+added,plan=plan,refinement=refinement,
            new_candidate_count=len(added),new_frames_have_itm_scores=False)
        selections,trace=select_pool(pool,f,k)
        pp=root/'pools'/profile/(q+'.json');fp=root/'features'/profile/(q+'.npy');tp=root/'traces'/profile/(q+'.json')
        durable(pp,pool);save_array(fp,f);durable(tp,trace);paths=[pp,fp,tp]
        for method,frames in selections.items():
            dest=root/'selections'/profile/method/(q+'.json')
            durable(dest,dict(question_id=q,method=method,scorer_id='S2-BLIP-2-ITM',
                frame_budget=k,selection=dict(video=pool['video'],selected_frames=frames)));paths.append(dest)
        timing.update(wall_seconds=time.perf_counter()-start,fine_visual_preprocess_seconds=visual_pre,
                      fine_visual_forward_seconds=visual_forward,
                      model_calls={key:_SCORER.counts[key]-before[key] for key in before},
                      peak_allocated_gib=_SCORER.torch.cuda.max_memory_allocated()/2**30)
        checkpoint(root,profile,q,sources,paths,dict(timing=timing,temporary_rgb_sha256=assets))
        # 步骤3：只有特征/池/完成检查点均落盘后，删除本轮明确创建的暂存图像。
        for asset in assets:
            p=Path(asset['path'])
            if p.parent!=assets_dir or sha256(p)!=asset['sha256']:raise ValueError('Unexpected transient image')
            p.unlink()


def mixed_diagnostics(root):
    """接口：S2分数/窗口+S1特征的400项离线选择；缺失细帧只补S1纯视觉，不做ITM。"""
    from videoqa_methods.density_pipeline import DensityScorer
    from PIL import Image
    from videoqa_budget32.frames import FrameProvider
    root=Path(root);dev=read_json(ROOT/'data/manifests/video_mme_development_200_20260920.json')['rows']
    scorer=None
    for row in dev:
        q=row['question_id'];old=read_json(BASE/'pools'/(q+'.json'));oldf=np.load(BASE/'features'/(q+'.npy'),allow_pickle=False)
        bank={key:oldf[i] for i,key in enumerate(source_keys(old['candidates']))}
        for profile,k in PROFILES.items():
            pp=root/'pools'/profile/(q+'.json');sp=root/'coarse/pools'/(q+'.json');dest=root/'mixed_diagnostic'/profile/(q+'.json')
            if dest.exists():
                saved=read_json(dest)
                if saved['pool_sha256']!=sha256(pp):raise ValueError('Mixed source changed')
                continue
            pool=read_json(pp);features=[];missing=[]
            for i,key in enumerate(source_keys(pool['candidates'])):
                features.append(bank.get(key))
                if key not in bank:missing.append(i)
            timing=[]
            if missing and scorer is None:scorer=DensityScorer('/home/models/Salesforce/blip-itm-base-coco')
            for offset in range(0,len(missing),16):
                indices=missing[offset:offset+16]
                images,rgb=FrameProvider(dict(video=pool['video'],selected_frames=sorted(
                    [pool['candidates'][i] for i in indices],key=lambda f:f['source_pts']))).decode()
                order=sorted(indices,key=lambda i:pool['candidates'][i]['source_pts'])
                try:v,pre,forward=scorer.visual_batch(images)
                finally:
                    for im in images:im.close()
                for i,feature in zip(order,v):features[i]=feature;bank[source_keys([pool['candidates'][i]])[0]]=feature
                timing.append(dict(indices=order,rgb_sha256=rgb,preprocess_seconds=pre,forward_seconds=forward))
            selections,trace=select_pool(pool,np.stack(features),k)
            durable(dest,dict(question_id=q,profile=profile,pool_sha256=sha256(pp),
                source_pool_sha256=sha256(BASE/'pools'/(q+'.json')),source_features_sha256=sha256(BASE/'features'/(q+'.npy')),
                selected_source_keys={m:source_keys(f) for m,f in selections.items()},trace=trace,
                newly_encoded_S1_visual_frames=len(missing),costs=timing,new_itm=0,new_QA=0))
    durable(root/'mixed_diagnostic/summary.json',dict(status='completed',questions_per_profile=200,
        new_QA=0,new_S1_ITM=0,new_S1_visual_frames=0 if scorer is None else scorer.counts['visual_frames'],utc=utc()))
