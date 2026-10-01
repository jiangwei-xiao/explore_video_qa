#!/usr/bin/env python3
"""exp08开发200题独立入口：缓存评分、新窗口补查、冻结输入、后台一次问答。"""
import argparse
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,utc,save_array,rows
from videoqa_audit.query_input_probe import BASE,PYTHON,MODEL,FULL
from videoqa_budget32.selection import config,configure_visual_precision
from videoqa_budget32.prefix_full import release_check
from videoqa_budget32.frames import FrameProvider
from videoqa_methods.option_seed_entry import option_seed_plan,select_option_pool
from videoqa_methods.density_selection import refinement_targets
SOURCE=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1'
REPLAY=ROOT/'outputs/analysis/option_coarse_replay200_20261001_r1'
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
METHOD='RD-1.2-exp08'
QA_PYTHON='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'


def verify(run):
    """接口：固定原评分、代码、数据与模型环境；恢复不得接受变更资产。"""
    release_check()
    runpy.run_path(str(ROOT/'scripts/run_option_rank_development.py'))['verify'](SOURCE)
    spec=read_json(run/'protocol.json')
    for path,h in spec['sources'].items():assert sha256(path)==h,path
    for q,h in spec['inputs'].items():assert sha256(run/'inputs'/(q+'.json'))==h
    return spec


def prepare(run,gpus):
    """接口：冻结200题、新窗口及调用上限，旧问答不导入新版本。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():return verify(run)
        release_check()
        old=runpy.run_path(str(ROOT/'scripts/run_option_rank_development.py'))['verify'](SOURCE)
        assert gpus and len(gpus)==len(set(gpus)) and set(gpus)<=set(map(str,range(8)))
        ids=[r['question_id'] for r in read_json(DATA)['rows']];assert ids==old['question_ids'] and len(ids)==200
        data={r['question_id']:r for r in rows()};sources={str(SOURCE/'protocol.json'):sha256(SOURCE/'protocol.json'),str(DATA):sha256(DATA)}
        inputs={};targets=0;batches=0;unchanged=[];cfg=config(32,32)
        names=['scripts/run_option_seed_experiment.py','src/videoqa_methods/option_seed_entry.py',
            'src/videoqa_audit/option_rank_selection.py','src/videoqa_audit/option_coarse_replay.py',
            'src/videoqa_methods/density_pipeline.py','src/videoqa_methods/model_ops.py',
            'src/videoqa_methods/density_selection.py','src/videoqa_methods/prefix_coverage_budget.py',
            'src/videoqa_methods/weighted_seed_support.py','src/videoqa_methods/coarse_eligibility.py',
            'src/videoqa_budget32/prefix_full.py','configs/retrieval_density_v1.json']
        for name in names:
            sources[str(ROOT/name)]=sha256(ROOT/name);dest=run/'code_snapshot'/name
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        # 步骤1：重新核对已评分源池，不把旧细帧混入初始候选。
        for q in ids:
            pp=BASE/'pools'/(q+'.json');fp=BASE/'features'/(q+'.npy');cp=BASE/'selection_checkpoints'/(q+'.json')
            rp=SOURCE/'results'/(q+'.json');offline=REPLAY/'questions'/(q+'.json')
            pool=read_json(pp);check=read_json(cp);n=pool['initial_count']
            for p in (pp,fp):assert sha256(p)==check['files'][str(p.relative_to(BASE))]
            plan,ranking=option_seed_plan(pool['candidates'][:n],pool['initial_scores'],read_json(rp)['scores'],pool['video'],cfg)
            assert plan==read_json(offline)['conditions']['option_seeds']['plan']
            equal=plan==pool['plan']
            if equal:unchanged.append(q)
            upper=sum(len(refinement_targets(r,pool['video'],cfg)) for r in plan['regions'] if r['refine']);targets+=upper;batches+=(upper+15)//16
            for p in (pp,fp,cp,rp,offline,FULL/'imported/results/RD-1.2-exp07'/(q+'.json')):sources[str(p)]=sha256(p)
            item=dict(question_id=q,video_id=data[q]['video_id'],pool_path=str(pp),feature_path=str(fp),
                plan=plan,entry_ranking=ranking,unchanged_plan=equal,target_upper=upper)
            durable(run/'inputs'/(q+'.json'),item);inputs[q]=sha256(run/'inputs'/(q+'.json'))
        # 步骤2：保守预留PNG及轨迹空间，模型只读，不通过减帧规避空间或显存约束。
        estimate=targets*8*1024**2+5*1024**3;free=shutil.disk_usage(run).free;assert free>=1.5*estimate
        spec=dict(method=METHOD,question_ids=ids,allowed_gpus=gpus,configuration=cfg,sources=sources,inputs=inputs,
            max_qa_calls=200,new_itm_calls=0,new_query_generation_calls=0,new_scope_calls=0,
            fine_target_upper=targets,visual_batch_upper=batches,unchanged_plan_questions=unchanged,estimated_asset_bytes=estimate,free_bytes=free,utc=utc())
        durable(run/'protocol.json',spec);return spec


def visual_once(run,q,start,scorer,images,assets,gpu):
    """接口：纯视觉批次先记账，未知返回禁止再次前向；新帧不执行ITM。"""
    path=run/'visual_calls'/q/(str(start)+'.json');fp=path.with_suffix('.npy')
    identity=dict(question_id=q,start=start,gpu=gpu,assets=[a['sha256'] for a in assets],protocol_sha256=sha256(run/'protocol.json'))
    if path.exists():
        saved=read_json(path);assert saved['identity']==identity and saved['status']=='completed','Uncertain visual call'
        assert sha256(fp)==saved['features_sha256'];return np.load(fp,allow_pickle=False)
    # 步骤1：开始状态持久化后才允许一次视觉前向。
    durable(path,dict(status='started_may_forward',identity=identity,utc=utc()))
    values,pre,forward=scorer.visual_batch(images);h=save_array(fp,values)
    # 步骤2：返回特征和计时一起落盘，恢复只读已完成批次。
    durable(path,dict(status='completed',identity=identity,features_sha256=h,preprocess_seconds=pre,forward_seconds=forward,utc=utc()))
    return values


def select_one(run,q,scorer,gpu,spec):
    """接口：新窗口2FPS解码→新帧视觉特征→原exp07选择，最后冻结精确输入。"""
    from PIL import Image
    from videoqa_methods.density_pipeline import decode_regions
    dest=run/'selections'/(q+'.json')
    if dest.exists():
        saved=read_json(dest)
        for p,h in saved['sources'].items():assert sha256(p)==h
        return
    item=read_json(run/'inputs'/(q+'.json'));old=read_json(item['pool_path']);n=old['initial_count']
    coarse=old['candidates'][:n];original=np.load(item['feature_path'],allow_pickle=False);plan=item['plan'];started=time.perf_counter()
    dp=run/'decode_checkpoints'/(q+'.json')
    # 步骤1：完整解码检查点可复用；不完整解码只有CPU文件产物，不复用未知视觉调用。
    if dp.exists():
        decoded=read_json(dp);assert decoded['protocol_sha256']==sha256(run/'protocol.json')
    else:
        added,assets,refinement,timing=decode_regions(old['video'],coarse,plan,spec['configuration'],run/'fine_assets'/q)
        decoded=dict(added=added,assets=assets,refinement=refinement,timing=timing,protocol_sha256=sha256(run/'protocol.json'))
        durable(dp,decoded)
    for a in decoded['assets']:assert sha256(a['path'])==a['sha256']
    blocks=[original[:n].copy()]
    for start in range(0,len(decoded['assets']),16):
        if (run/'STOP.json').exists():raise RuntimeError('Peer failure')
        assets=decoded['assets'][start:start+16];images=[]
        try:
            for a in assets:
                with Image.open(a['path']) as image:images.append(image.convert('RGB'))
            blocks.append(visual_once(run,q,start,scorer,images,assets,gpu))
        finally:
            for image in images:image.close()
    features=np.concatenate(blocks);pool=dict(video=old['video'],candidates=coarse+decoded['added'],initial_count=n,
        initial_scores=old['initial_scores'],plan=plan,refinement=decoded['refinement'],new_assets=decoded['assets'])
    # 步骤2：相同种子必须复现旧完整池、数值和离散选择，不能借新批次掩盖实现差异。
    selected,trace=select_option_pool(pool,features,spec['configuration']);replay=None
    if item['unchanged_plan']:
        assert pool['candidates']==old['candidates'];error=float(np.max(np.abs(features-original)));assert error<=1e-6
        previous=read_json(FULL/'selections/RD-1.2-exp07'/(q+'.json'));assert selected==previous['selected_indices']
        replay=dict(status='passed',max_feature_error=error,discrete_selection_equal=True)
    selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in selected])
    images,hashes=FrameProvider(selection).decode()
    for image in images:image.close()
    pp=run/'pools'/(q+'.json');fp=run/'features'/(q+'.npy');tp=run/'traces'/(q+'.json')
    durable(pp,pool);save_array(fp,features);durable(tp,trace)
    sources={str(p):sha256(p) for p in (run/'protocol.json',run/'inputs'/(q+'.json'),pp,fp,tp,dp)}
    durable(dest,dict(question_id=q,video_id=item['video_id'],method=METHOD,selection=selection,rgb_sha256=hashes,
        selected_indices=selected,sources=sources,selection_seconds=time.perf_counter()-started,unchanged_plan_replay=replay,
        new_fine_frames=len(decoded['added']),new_itm_calls=0))


def worker(run,gpu,questions):
    """接口：单卡常驻BLIP只处理新帧视觉；合成验收、真实计算分别记账。"""
    os.environ['CUDA_VISIBLE_DEVICES']=gpu;offline_environment();q=None
    try:
        import torch
        from videoqa_methods.density_pipeline import DensityScorer
        torch.set_num_threads(2);torch.set_num_interop_threads(1);configure_visual_precision(torch)
        started=time.perf_counter();scorer=DensityScorer(str(MODEL));load=time.perf_counter()-started
        p=run/'synthetic'/(gpu+'.json')
        if p.exists():assert read_json(p)['status']=='passed','Uncertain synthetic call'
        else:
            durable(p,dict(status='started',utc=utc()));report=scorer.synthetic_acceptance()
            durable(p,dict(report,load_seconds=load,utc=utc()))
        spec=read_json(run/'protocol.json')
        for q in questions:
            if (run/'STOP.json').exists():raise RuntimeError('Peer failure')
            select_one(run,q,scorer,gpu,spec)
            durable(run/'workers'/(gpu+'.json'),dict(question_id=q,pid=os.getpid(),utc=utc()))
    except BaseException as exc:
        issue=dict(question_id=q,gpu=gpu,error=repr(exc),traceback=traceback.format_exc(),utc=utc())
        durable(run/'failures'/f'{gpu}_{time.time_ns()}.json',issue);durable(run/'STOP.json',issue);raise


def export_inputs(run,spec):
    """接口：选帧全部完成才冻结200份问答输入，不导入任何旧预测。"""
    exports={};directory=run/'qa_inputs';expected_calls=set();fine_frames=0
    for q in spec['question_ids']:
        source=run/'selections'/(q+'.json');item=read_json(source)
        for p,h in item['sources'].items():assert sha256(p)==h
        fine_frames+=item['new_fine_frames']
        expected_calls.update(str(run/'visual_calls'/q/(str(i)+'.json')) for i in range(0,item['new_fine_frames'],16))
        dest=directory/(q+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha256(dest)==sha256(source)
        else:shutil.copy2(source,dest)
        exports[q]=sha256(dest)
    calls=list((run/'visual_calls').glob('*/*.json'))
    assert {str(p) for p in calls}==expected_calls and len(calls)<=spec['visual_batch_upper']
    assert all(read_json(p)['status']=='completed' for p in calls)
    assert sum(len(read_json(p)['identity']['assets']) for p in calls)==fine_frames<=spec['fine_target_upper']
    summary=dict(status='completed',fine_frames=fine_frames,visual_batch_calls=len(calls),new_itm_calls=0,new_scope_calls=0,
                 new_query_calls=0,qa_input_count=200,unchanged_plan_replays=spec['unchanged_plan_questions'])
    durable(run/'selection_summary.json',summary)
    manifest=dict(status='completed',question_count=200,frame_budget=32,question_ids=spec['question_ids'],dataset_manifest=str(DATA),
        manifest_sha256=sha256(DATA),parent_run=str(BASE),parent_protocol_sha256=sha256(BASE/'protocol.json'),method=METHOD,
        configuration='M32/K32 option-rank seeds; original-question anchors/support; exp07 prefix/DP; original conditional 2FPS refinement',
        diagnostic=False,exports=exports)
    p=directory/'selection_manifest.json'
    if p.exists():assert read_json(p)==manifest
    else:durable(p,manifest)
    return directory


def execute(run):
    """接口：脱离会话完成选择和一次200题问答，异常暂停，不自动重试。"""
    from videoqa_runtime.baseline_run import available_gpus
    with exclusive(run/'coordinator.lock'):
        spec=verify(run);assert not (run/'STOP.json').exists(),'Inspect paused run before recovery'
        started=time.perf_counter();remaining=[q for q in spec['question_ids'] if not (run/'selections'/(q+'.json')).exists()]
        if remaining:
            ap=run/'assignment.json'
            while True:
                try:free,_=available_gpus(spec['allowed_gpus'])
                except RuntimeError:free=[]
                if ap.exists():
                    assignment=read_json(ap);needed=[g for g,qs in assignment.items() if set(qs)&set(remaining)]
                    if set(needed)<=set(free):break
                elif free:
                    assignment={g:[] for g in free};loads={g:0 for g in free}
                    for q in sorted(remaining,key=lambda q:-read_json(run/'inputs'/(q+'.json'))['target_upper']):
                        g=min(free,key=lambda g:(loads[g],g));assignment[g].append(q);loads[g]+=read_json(run/'inputs'/(q+'.json'))['target_upper']+1
                    durable(ap,assignment);needed=free;break
                durable(run/'status.json',dict(status='waiting_resources',pid=os.getpid(),utc=utc()));time.sleep(15)
            children=[];logs=[];(run/'logs').mkdir(exist_ok=True)
            for gpu in needed:
                qs=[q for q in assignment[gpu] if q in remaining];log=(run/'logs'/(gpu+'.log')).open('a');logs.append(log)
                children.append(subprocess.Popen([PYTHON,__file__,'--run-id',run.name,'--stage','worker','--gpu',gpu,'--questions',*qs],
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL))
            while any(p.poll() is None for p in children):
                durable(run/'status.json',dict(status='selecting',completed=len(list((run/'selections').glob('*.json'))),total=200,
                    pid=os.getpid(),sid=os.getsid(0),elapsed_seconds=time.perf_counter()-started,utc=utc()));time.sleep(5)
            for log in logs:log.close()
            assert all(p.returncode==0 for p in children),'Selection paused; no automatic retry'
        # 步骤1：完整身份复核后冻结输入；问答采用原专用环境和已验收框架。
        verify(run);inputs=export_inputs(run,spec)
        completed=run/'selection_complete.json'
        if not completed.exists():durable(completed,dict(status='completed',seconds=time.perf_counter()-started,utc=utc()))
        entry=str(ROOT/'scripts/run_frozen_development.py')
        subprocess.run([QA_PYTHON,entry,'--run-id',run.name,'--stage','prepare','--input-dir',str(inputs),'--gpus',*spec['allowed_gpus'],'--resume'],check=True)
        durable(run/'status.json',dict(status='qa',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤2：旧执行器自动通过前5题运行验收并核对完成恢复，预算仍为200。
        subprocess.run([QA_PYTHON,entry,'--run-id',run.name,'--stage','qa','--resume'],check=True)
        durable(run/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),seconds=time.perf_counter()-started,utc=utc()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=['prepare','run','worker'],required=True)
    p.add_argument('--gpus',nargs='+',default=list(map(str,range(8))));p.add_argument('--gpu');p.add_argument('--questions',nargs='+');p.add_argument('--resume',action='store_true')
    a=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id);run=ROOT/'outputs/methods'/a.run_id
    if a.stage=='prepare':prepare(run,a.gpus)
    elif a.stage=='worker':worker(run,a.gpu,a.questions)
    else:
        if (run/'visual_calls').exists() and not a.resume:raise RuntimeError('Explicit resume required')
        execute(run)
