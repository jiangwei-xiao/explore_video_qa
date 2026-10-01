#!/usr/bin/env python3
"""exp10独立开发200题入口：严格恒定RGB资格检查，后台选择与一次问答。"""
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
from videoqa_methods.option_seed_entry import select_option_pool
from videoqa_methods.constant_frame_eligibility import pixel_statistics,valid_seed_plan,select_eligible,select_checked
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
METHOD='RD-1.2-exp10'
QA_PYTHON='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'
VISUAL_SOURCE=ROOT/'scripts/run_joint_seed_experiment.py'


def verify(run):
    """接口：恢复核对代码、旧池、模型发布与200题身份，不静默接纳改变来源。"""
    s=read_json(run/'protocol.json');assert s['method']==METHOD and s['max_qa_calls']==200
    release_check()
    for p,h in s['sources'].items():assert sha256(p)==h,p
    return s


def prepare(run,gpus):
    """接口：冻结原问题评分与exp07对照，最多200次QA；新真实ITM/分类/查询均0。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():return verify(run)
        release_check();assert gpus and len(gpus)==len(set(gpus)) and set(gpus)<=set(map(str,range(8)))
        ids=[r['question_id'] for r in read_json(DATA)['rows']];assert len(ids)==len(set(ids))==200
        sources={str(DATA):sha256(DATA)};inputs={}
        names=['scripts/run_constant_frame_experiment.py','src/videoqa_methods/constant_frame_eligibility.py',
            'scripts/run_joint_seed_experiment.py','src/videoqa_methods/option_seed_entry.py','src/videoqa_audit/option_coarse_replay.py',
            'src/videoqa_methods/density_pipeline.py','src/videoqa_methods/model_ops.py','src/videoqa_methods/density_selection.py',
            'src/videoqa_methods/prefix_coverage_budget.py','src/videoqa_methods/weighted_seed_support.py',
            'src/videoqa_methods/coarse_eligibility.py','src/videoqa_budget32/frames.py','src/videoqa_budget32/prefix_full.py']
        # 步骤1：只新增文件，不修改exp07或后端；快照绑定本轮独立选择实现。
        for name in names:
            p=ROOT/name;sources[str(p)]=sha256(p);dest=run/'code_snapshot'/name
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        for q in ids:
            pp=BASE/'pools'/(q+'.json');fp=BASE/'features'/(q+'.npy');cp=BASE/'selection_checkpoints'/(q+'.json')
            check=read_json(cp)
            for p in (pp,fp):assert sha256(p)==check['files'][str(p.relative_to(BASE))]
            answer=FULL/'imported/results/RD-1.2-exp07'/(q+'.json')
            for p in (pp,fp,cp,answer):sources[str(p)]=sha256(p)
            pool=read_json(pp);inputs[q]=dict(pool_path=str(pp),feature_path=str(fp),duration=pool['video']['duration_seconds'])
        # 步骤2：磁盘不足暂停；不改变分辨率或帧数规避资源不足。
        assert shutil.disk_usage(run).free>150*1024**3
        s=dict(method=METHOD,parent_method='RD-1.2-exp07',question_ids=ids,allowed_gpus=gpus,configuration=config(32,32),
            sources=sources,inputs=inputs,max_qa_calls=200,new_itm_calls=0,new_query_calls=0,new_scope_calls=0,
            predicate='uint8 RGB global min == global max; exact scalar constant only',
            coverage_reference='all original coarse candidates unchanged',fine_reuse='only when plan exactly unchanged',utc=utc())
        durable(run/'protocol.json',s);return s


class PixelInspector:
    """接口：逐源帧精确RGB检查点；仅CPU解码可安全恢复，不参与答案计分。"""
    def __init__(self,run,q,pool):
        self.path=run/'pixels'/(q+'.json');self.pool=pool;self.protocol=sha256(run/'protocol.json');self.values={}
        if self.path.exists():
            saved=read_json(self.path);assert saved['protocol_sha256']==self.protocol;self.values=saved['values']

    def __call__(self,indices):
        """步骤1核对源身份，步骤2分批读取未知RGB并原子保存，步骤3返回像素判定。"""
        c=self.pool['candidates'];missing=[]
        for i in indices:
            key=str(i);identity=[c[i]['source_frame_index'],c[i]['source_pts']]
            if key in self.values:assert self.values[key]['source_identity']==identity
            else:missing.append(i)
        for start in range(0,len(missing),32):
            batch=sorted(missing[start:start+32],key=lambda i:c[i]['source_pts'])
            images,hashes=FrameProvider(dict(video=self.pool['video'],selected_frames=[c[i] for i in batch])).decode()
            for i,image,h in zip(batch,images,hashes):
                self.values[str(i)]=dict(pixel_statistics(image),rgb_sha256=h,source_identity=[c[i]['source_frame_index'],c[i]['source_pts']]);image.close()
            durable(self.path,dict(protocol_sha256=self.protocol,values=self.values))
        return {i:self.values[str(i)] for i in indices}


def select_one(run,q,gpu,spec,get_scorer):
    """接口：检查原题种子→原条件补查→资格受限exp07前缀/DP→最终RGB冻结。"""
    dest=run/'selections'/(q+'.json')
    if dest.exists():
        saved=read_json(dest)
        for p,h in saved['sources'].items():assert sha256(p)==h
        return
    started=time.perf_counter();item=spec['inputs'][q];old=read_json(item['pool_path']);n=old['initial_count']
    original=np.load(item['feature_path'],allow_pickle=False);coarse=old['candidates'][:n]
    pool=dict(video=old['video'],candidates=coarse,initial_count=n,initial_scores=old['initial_scores'])
    inspector=PixelInspector(run,q,pool)
    plan,entry=valid_seed_plan(coarse,old['initial_scores'],old['video'],spec['configuration'],inspector)
    same=plan==old['plan'];pool['plan']=plan;new_visual=0
    # 步骤1：同计划复用冻结细帧；计划改变才按原2FPS规则解码/视觉计算，不复用越界旧细帧。
    if same:
        pool.update(candidates=old['candidates'],refinement=old['refinement'],new_assets=old['new_assets'])
        features=original.copy()
    else:
        from videoqa_methods.density_pipeline import decode_regions
        from PIL import Image
        dp=run/'decode_checkpoints'/(q+'.json')
        if dp.exists():
            decoded=read_json(dp);assert decoded['protocol_sha256']==sha256(run/'protocol.json')
        else:
            added,assets,refinement,timing=decode_regions(old['video'],coarse,plan,spec['configuration'],run/'fine_assets'/q)
            decoded=dict(added=added,assets=assets,refinement=refinement,timing=timing,protocol_sha256=sha256(run/'protocol.json'))
            durable(dp,decoded)
        blocks=[original[:n].copy()]
        visual_once=runpy.run_path(str(VISUAL_SOURCE))['visual_once']
        for offset in range(0,len(decoded['assets']),16):
            if (run/'STOP.json').exists():raise RuntimeError('Peer failure')
            assets=decoded['assets'][offset:offset+16];images=[]
            try:
                for a in assets:
                    assert sha256(a['path'])==a['sha256']
                    with Image.open(a['path']) as image:images.append(image.convert('RGB'))
                call=run/'visual_calls'/q/(str(offset)+'.json')
                scorer=None if call.exists() else get_scorer()
                blocks.append(visual_once(run,q,offset,scorer,images,assets,gpu))
            finally:
                for image in images:image.close()
        features=np.concatenate(blocks);new_visual=len(decoded['added'])
        pool.update(candidates=coarse+decoded['added'],refinement=decoded['refinement'],new_assets=decoded['assets'])
    inspector.pool=pool
    # 步骤2：无禁用时逐题精确重放冻结exp07选择器；同计划还核对历史正式输入。
    control,control_trace=select_option_pool(pool,features,spec['configuration'])
    unmasked,unmasked_trace=select_eligible(pool,features,spec['configuration'],set())
    assert control==unmasked and control_trace['region_budgets']==unmasked_trace['region_budgets']
    historical=read_json(FULL/'imported/results/RD-1.2-exp07'/(q+'.json'))
    if same:assert [pool['candidates'][i]['source_pts'] for i in control]==[f['source_pts'] for f in historical['selection']['selected_frames']]
    banned={int(i) for i,v in inspector.values.items() if v['constant_rgb']}
    selected,trace=select_checked(pool,features,spec['configuration'],inspector,banned)
    assert all(not inspector.values[str(i)]['constant_rgb'] for i in selected)
    selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in selected])
    hashes=[inspector.values[str(i)]['rgb_sha256'] for i in selected]
    # 步骤3：保存可复算池/特征/排除轨迹；像素检查与缓存计算成本不冒充独立原片E2E。
    pp=run/'pools'/(q+'.json');fp=run/'features'/(q+'.npy');tp=run/'traces'/(q+'.json');ep=run/'entries'/(q+'.json')
    durable(pp,pool);save_array(fp,features);durable(tp,trace);durable(ep,entry)
    sources={str(p):sha256(p) for p in (run/'protocol.json',pp,fp,tp,ep,inspector.path)}
    durable(dest,dict(question_id=q,method=METHOD,selection=selection,rgb_sha256=hashes,selected_indices=selected,sources=sources,
        selection_seconds=time.perf_counter()-started,unchanged_plan=same,new_fine_visual_frames=new_visual,
        control_replay_passed=True,all_final_pixels_verified=True,
        same_rgb_as_exp07=hashes==historical['rgb_sha256'],banned_count=len(trace['banned_indices'])))


def worker(run,gpu,questions):
    """接口：一卡一进程，只有新细帧需要时才加载BLIP；失败暂停，未知视觉调用不重试。"""
    os.environ['CUDA_VISIBLE_DEVICES']=gpu;offline_environment();holder=[];q=None
    def get_scorer():
        if not holder:
            import torch
            from videoqa_methods.density_pipeline import DensityScorer
            torch.set_num_threads(2);torch.set_num_interop_threads(1);configure_visual_precision(torch)
            scorer=DensityScorer(str(MODEL));p=run/'synthetic'/(gpu+'.json')
            if p.exists():assert read_json(p)['status']=='passed'
            else:
                durable(p,dict(status='started',utc=utc()));report=scorer.synthetic_acceptance();durable(p,dict(report,utc=utc()))
            holder.append(scorer)
        return holder[0]
    try:
        spec=read_json(run/'protocol.json');assert spec['method']==METHOD
        for q in questions:
            if (run/'STOP.json').exists():raise RuntimeError('Peer failure')
            select_one(run,q,gpu,spec,get_scorer)
            durable(run/'workers'/(gpu+'.json'),dict(question_id=q,pid=os.getpid(),utc=utc()))
    except BaseException as exc:
        issue=dict(question_id=q,gpu=gpu,error=repr(exc),traceback=traceback.format_exc(),utc=utc())
        durable(run/'failures'/f'{gpu}_{time.time_ns()}.json',issue);durable(run/'STOP.json',issue);raise


def export_inputs(run,spec):
    """接口：200题像素/源帧全部验收后导出，旧答案不导入本轮新版本。"""
    exports={};directory=run/'qa_inputs';records=[]
    for q in spec['question_ids']:
        p=run/'selections'/(q+'.json');r=read_json(p);assert r['method']==METHOD and r['all_final_pixels_verified']
        for path,h in r['sources'].items():assert sha256(path)==h
        dest=directory/(q+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha256(dest)==sha256(p)
        else:shutil.copy2(p,dest)
        exports[q]=sha256(dest);records.append(r)
    calls=list((run/'visual_calls').glob('*/*.json'));assert all(read_json(p)['status']=='completed' for p in calls)
    assert sum(len(read_json(p)['identity']['assets']) for p in calls)==sum(r['new_fine_visual_frames'] for r in records)
    durable(run/'selection_summary.json',dict(question_count=200,changed_plan_questions=[r['question_id'] for r in records if not r['unchanged_plan']],
        changed_rgb_questions=[r['question_id'] for r in records if not r['same_rgb_as_exp07']],
        new_fine_visual_frames=sum(r['new_fine_visual_frames'] for r in records),visual_batches=len(calls),new_itm_calls=0,
        final_constant_frames=0,actual_frame_counts={r['question_id']:len(r['selected_indices']) for r in records}))
    manifest=dict(status='completed',question_count=200,frame_budget=32,question_ids=spec['question_ids'],dataset_manifest=str(DATA),
        manifest_sha256=sha256(DATA),parent_run=str(BASE),parent_protocol_sha256=sha256(BASE/'protocol.json'),method=METHOD,
        configuration='M32/K32 exp07 plus exact scalar-constant RGB exclusion from seeds and selected candidates; coarse coverage unchanged',diagnostic=False,exports=exports)
    p=directory/'selection_manifest.json'
    if p.exists():assert read_json(p)==manifest
    else:durable(p,manifest)
    return directory


def execute(run):
    """接口：独立后台完成输入检查、先行验收与200题Qwen，断线不依赖交互放行。"""
    from videoqa_runtime.baseline_run import available_gpus
    with exclusive(run/'coordinator.lock'):
        spec=verify(run);assert not (run/'STOP.json').exists(),'Inspect failure before recovery'
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
                    assignment={g:[] for g in free};loads={g:0. for g in free}
                    for q in sorted(remaining,key=lambda q:-spec['inputs'][q]['duration']):
                        g=min(free,key=lambda g:(loads[g],g));assignment[g].append(q);loads[g]+=spec['inputs'][q]['duration']
                    durable(ap,assignment);needed=free;break
                durable(run/'status.json',dict(status='waiting_resources',pid=os.getpid(),utc=utc()));time.sleep(15)
            children=[];logs=[];(run/'logs').mkdir(exist_ok=True)
            for g in needed:
                log=(run/'logs'/(g+'.log')).open('a');logs.append(log)
                children.append(subprocess.Popen([PYTHON,__file__,'--run-id',run.name,'--stage','worker','--gpu',g,'--questions',*[q for q in assignment[g] if q in remaining]],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL))
            while any(p.poll() is None for p in children):
                durable(run/'status.json',dict(status='selecting',completed=len(list((run/'selections').glob('*.json'))),pid=os.getpid(),sid=os.getsid(0),utc=utc()));time.sleep(5)
            for log in logs:log.close()
            assert all(p.returncode==0 for p in children),'Selection paused; no automatic retry'
        verify(run);inputs=export_inputs(run,spec)
        if not (run/'selection_complete.json').exists():durable(run/'selection_complete.json',dict(seconds=time.perf_counter()-started,utc=utc()))
        entry=str(ROOT/'scripts/run_frozen_development.py')
        subprocess.run([QA_PYTHON,entry,'--run-id',run.name,'--stage','prepare','--input-dir',str(inputs),'--gpus',*spec['allowed_gpus'],'--resume'],check=True)
        durable(run/'status.json',dict(status='qa',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        subprocess.run([QA_PYTHON,entry,'--run-id',run.name,'--stage','qa','--resume'],check=True)
        durable(run/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),seconds=time.perf_counter()-started,utc=utc()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=['prepare','run','worker'],required=True)
    p.add_argument('--gpus',nargs='+',default=list(map(str,range(8))));p.add_argument('--gpu');p.add_argument('--questions',nargs='+');p.add_argument('--resume',action='store_true')
    a=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id);run=ROOT/'outputs/methods'/a.run_id
    if a.stage=='prepare':prepare(run,a.gpus)
    elif a.stage=='worker':worker(run,a.gpu,a.questions)
    else:
        if (run/'visual_calls').exists() and not a.resume:raise ValueError('Explicit resume required')
        execute(run)
