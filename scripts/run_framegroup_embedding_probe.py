#!/usr/bin/env python3
"""两题帧组能力探针：52次非生成式嵌入，禁止变成自动选帧或准确率成绩。"""
import argparse
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
import numpy as np
from PIL import Image
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,utc,save_array,digest
from videoqa_full.prepare import environment
from videoqa_audit.framegroup_design import create_design
from videoqa_audit.framegroup_embedding import StrictEmbeddingBackend,prepare_view
ENV=Path('/App/conda/envs/explore_video_qa_vlembedding')


def verify(run):
    """接口：核验独立环境、模型、原始帧及代码，不接受运行中改参数或改指纹。"""
    spec=read_json(run/'embedding_protocol.json')
    if Path(sys.prefix)!=ENV or environment(str(ENV/'bin/python'))!=spec['environment']:raise ValueError('Embedding environment changed')
    for p,h in {**spec['sources'],**spec['model_files']}.items():assert sha256(p)==h,p
    return spec


def prepare(run,gpu):
    """接口：事前冻结实际模型视图并做全部CPU输入验收，不加载模型权重或生成答案。"""
    if (run/'embedding_protocol.json').exists():return verify(run)
    if Path(sys.prefix)!=ENV or read_json(run/'setup_status.json')['status']!='completed':raise ValueError('Isolated environment is not ready')
    env=environment(str(ENV/'bin/python'))
    for name,version in {'torch':'2.8.0','torchvision':'0.23.0','transformers':'4.57.1','qwen-vl-utils':'0.0.14'}.items():assert env[name].split('+')[0]==version
    design=create_design(run);model=Path(design['embedding_model']);assets=read_json(design['asset_manifest']);views={}
    # 步骤1：同一源帧只缩放一次，两种媒体模式共用相同模型视图。
    for fid,row in design['frame_sources'].items():views[fid]=prepare_view(row['file'],run/'model_frames'/(fid+'.png'))
    y,x=np.indices((720,1280),dtype=np.int32)
    for i in range(10):
        p=run/'synthetic_sources'/(str(i)+'.png');p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():Image.fromarray(np.stack([x%256,(y+17*i)%256,(x+y+19*i)%256],axis=-1).astype(np.uint8)).save(p)
        views['synthetic_'+str(i)]=prepare_view(p,run/'model_frames'/('synthetic_'+str(i)+'.png'))
    synthetic=[dict(id='synthetic_text',kind='text',text='A synthetic color gradient.'),dict(id='synthetic_image',kind='image',frame_ids=['synthetic_0']),
        dict(id='synthetic_video',kind='video',frame_ids=['synthetic_'+str(i) for i in range(10)])]
    # 步骤2：独立预处理对象不含模型，验证所有像素/顺序/Token，无NULL或截断回退。
    from transformers import Qwen3VLProcessor
    checker=object.__new__(StrictEmbeddingBackend);checker.processor=Qwen3VLProcessor.from_pretrained(str(model),local_files_only=True,padding_side='right');checker.normalized_images={}
    attempt=run/'preflight'/str(time.time_ns());checks={}
    try:
        for job in synthetic+design['jobs']:
            _,meta=checker.preprocess(job,views);p=attempt/(job['id']+'.json');durable(p,meta);checks[job['id']]=str(p)
    except BaseException as exc:
        durable(attempt/'failure.json',dict(error=repr(exc),job=job['id'],model_forwards=0,utc=utc()));raise
    del checker
    sources=dict(assets['sources']);sources[str(run/'design_protocol_r2.json')]=sha256(run/'design_protocol_r2.json');sources[str(run/'evaluation_notes_r3.json')]=sha256(run/'evaluation_notes_r3.json')
    for v in views.values():sources[v['path']]=v['sha256']
    for p in checks.values():sources[p]=sha256(p)
    for name in ['scripts/run_framegroup_embedding_probe.py','src/videoqa_audit/framegroup_embedding.py','src/videoqa_audit/framegroup_design.py','src/videoqa_full/state.py','src/videoqa_runtime/common.py']:
        p=ROOT/name;sources[str(p)]=sha256(p);dest=run/'embedding_code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    # 步骤3：只读绑定全部实际权重/分词/预处理文件及官方封装；不写共享模型缓存。
    files=[p for p in model.iterdir() if p.is_file() and p.suffix in ('.json','.txt','.jinja','.safetensors')]+[model/'scripts/qwen3_vl_embedding.py']
    model_files={str(p):sha256(p) for p in files}
    spec=dict(design=design,views=views,synthetic_jobs=synthetic,input_checks=checks,sources=sources,model_files=model_files,environment=env,gpu=gpu,
        max_real_forwards=52,max_synthetic_forwards=3,final_qa_calls=0,utc=utc())
    durable(run/'embedding_protocol.json',spec);return spec


def encode_once(run,spec,backend,job,synthetic):
    """接口：调用前落盘，返回后保存向量检查点；已完成/已返回项不再次前向。"""
    p=run/'calls'/('synthetic' if synthetic else 'real')/(job['id']+'.json');raw=run/'embedding_raw_returns'/(job['id']+'.json');vp=run/'vectors'/(job['id']+'.npy')
    started=time.perf_counter();inputs,meta=backend.preprocess(job,spec['views']);pre=time.perf_counter()-started
    assert meta==read_json(spec['input_checks'][job['id']]),('Preflight input differs',job['id'])
    identity=dict(job=job,input_digest=digest(meta),gpu=spec['gpu'],protocol_sha256=sha256(run/'embedding_protocol.json'))
    if p.exists():
        old=read_json(p);assert old['identity']==identity
        if old['status']=='completed':assert sha256(vp)==old['vector_sha256'];return
    if raw.exists():
        value=read_json(raw);assert value['identity']==identity and sha256(vp)==value['vector_sha256'];durable(p,dict(value,status='completed',recovered_postprocessing=True));return
    if p.exists():
        raise RuntimeError('Uncertain embedding forward; no automatic retry')
    durable(p,dict(status='started_may_forward',identity=identity,utc=utc()))
    vector,metrics=backend.forward(inputs);save_array(vp,vector)
    value=dict(status='returned',identity=identity,vector_sha256=sha256(vp),input=meta,input_prepare_seconds=pre,metrics=metrics,utc=utc())
    durable(raw,value);durable(p,dict(value,status='completed'))


def summarize(run,spec):
    """接口：只给每个查询的帧组/逐帧相似度，不预测正确选项，不计算新方法准确率。"""
    design=spec['design'];assets=read_json(design['asset_manifest']);table=[]
    for q,case in assets['cases'].items():
        for view in design['queries'][q]:
            query=np.load(run/'vectors'/(q+'__'+view+'.npy'),allow_pickle=False)
            for group,pts in case['clips'].items():
                images=np.stack([np.load(run/'vectors'/('image__'+q+'__'+p+'.npy'),allow_pickle=False) for p in pts]);mean=images.mean(axis=0);mean/=np.linalg.norm(mean)
                ordered=np.load(run/'vectors'/(q+'__'+group+'__original.npy'),allow_pickle=False);reverse=np.load(run/'vectors'/(q+'__'+group+'__reversed.npy'),allow_pickle=False)
                table.append(dict(question_id=q,query_variant=view,group=group,image_max_cosine=float(np.max(images@query)),image_mean_embedding_cosine=float(mean@query),
                    video_cosine=float(ordered@query),reversed_video_cosine=float(reverse@query),order_embedding_cosine=float(ordered@reverse),
                    unique_source_frames=10,unique_rgb_views=len({case['frames'][p]['rgb_sha256'] for p in pts})))
    real=list((run/'calls/real').glob('*.json'));synthetic=list((run/'calls/synthetic').glob('*.json'))
    assert {p.stem for p in real}=={j['id'] for j in design['jobs']} and len(real)==52 and len(synthetic)==3
    assert all(read_json(p)['status']=='completed' for p in real+synthetic)
    artifacts={str(p):sha256(p) for folder in ['calls','embedding_raw_returns','vectors','input_checks','embedding_code_snapshot'] for p in sorted((run/folder).rglob('*')) if p.is_file()}
    durable(run/'embedding_summary.json',dict(status='completed',comparisons=table,real_embedding_forwards=52,synthetic_forwards=3,final_qa_calls=0,
        limitations='Two manually constructed cases; similarities are not answer confidence, temporal understanding, or algorithm accuracy; no training-overlap guarantee',artifacts=artifacts,utc=utc()))


def execute(run,gpu,stage):
    """接口：单GPU后台执行有限嵌入任务，资源占用或不确定前向时暂停，不降配置。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'embedding_run.lock'):
        spec=prepare(run,gpu)
        if stage=='prepare':return
        spec=verify(run)
        if (run/'embedding_summary.json').exists():
            for p,h in read_json(run/'embedding_summary.json')['artifacts'].items():assert sha256(p)==h
            durable(run/'embedding_recovery.json',dict(status='passed',extra_forwards=0,utc=utc()));print('Completed recovery, forwards 0',flush=True);return
        for folder,jobs in [('real',spec['design']['jobs']),('synthetic',spec['synthetic_jobs'])]:
            existing=list((run/'calls'/folder).glob('*.json'));assert {p.stem for p in existing}<={j['id'] for j in jobs}
            for p in existing:
                if read_json(p)['status']!='completed' and not (run/'embedding_raw_returns'/(p.stem+'.json')).exists():raise RuntimeError('Uncertain forward before model load')
        states=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
        if int(states[0])>=256 or int(states[1])!=0:raise RuntimeError('Configured GPU is not free')
        model=spec['design']['embedding_model'];source=str(Path(model)/'scripts/qwen3_vl_embedding.py')
        backend=StrictEmbeddingBackend(model,spec['model_files'][source]);durable(run/'embedding_load.json',backend.load_report)
        # 步骤1：合成接口验收；步骤2：所有真实输入一次运行；步骤3：核验与汇总。
        jobs=[(j,True) for j in spec['synthetic_jobs']]+[(j,False) for j in spec['design']['jobs']]
        for index,(job,synthetic) in enumerate(jobs):
            encode_once(run,spec,backend,job,synthetic);durable(run/'embedding_status.json',dict(status='running',completed=index+1,total=55,current=job['id'],pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        verify(run);summarize(run,spec);durable(run/'embedding_status.json',dict(status='completed',completed=55,total=55,pid=os.getpid(),utc=utc()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--gpu',default='0');p.add_argument('--stage',choices=['prepare','run'],default='run');a=p.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id) and a.gpu in list(map(str,range(8)))
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;offline_environment()
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(2027)
    run=ROOT/'outputs/analysis'/a.run_id
    try:execute(run,a.gpu,a.stage)
    except BaseException as exc:durable(run/('embedding_failure_'+str(time.time_ns())+'.json'),dict(error=repr(exc),utc=utc()));raise
