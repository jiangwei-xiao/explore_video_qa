#!/usr/bin/env python3
"""验证缓存CLS的官方ITC投影恢复，只有视觉前向，没有ITM或问答调用。"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,utc,save_array
from videoqa_full.prepare import environment
from videoqa_budget32.frames import FrameProvider
from videoqa_audit.blip_projection import load_projection_parameters,recover_projected_features
from videoqa_audit.query_input_probe import MODEL,PYTHON
PARENT=ROOT/'outputs/analysis/blip_joint_bank6_20261002_r1'


def prepare(run,gpu):
    """接口：保持相同六题310帧库，冻结投影代数和23批视觉调用上限。"""
    pp=run/'protocol.json'
    if pp.exists():
        s=read_json(pp)
        for p,h in {**s['sources'],**s['model_files']}.items():assert sha256(p)==h,p
        assert s['environment']==environment(PYTHON) and s['gpu']==gpu;return s
    old=read_json(PARENT/'protocol.json');sources=dict(old['sources']);sources[str(PARENT/'protocol.json')]=sha256(PARENT/'protocol.json')
    for name in ['scripts/run_blip_projection_probe.py','src/videoqa_audit/blip_projection.py']:
        p=ROOT/name;sources[str(p)]=sha256(p);dest=run/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    s=dict(question_ids=old['question_ids'],inputs=old['inputs'],sources=sources,model_files=old['model_files'],environment=old['environment'],gpu=gpu,
        visual_frames=310,visual_batches=23,synthetic_batches=1,synthetic_frames=16,itm_calls=0,qa_calls=0,tolerance=1e-6,
        scope='Algebraic recovery of official image-text contrastive vision projection from cached normalized post-LN CLS; fixed bank only',utc=utc())
    durable(pp,s);return s


def forward_once(run,label,start,scorer,images,ids):
    """接口：同一视觉前向保留原CLS与官方带偏置投影；未知返回禁止自动重复。"""
    import numpy as np
    import torch
    p=run/'calls'/label/(str(start)+'.json');rawpath=p.with_suffix('.raw.npy');projectpath=p.with_suffix('.projected.npy')
    identity=dict(ids=ids,rgb_sha256=[hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest() for im in images],protocol_sha256=sha256(run/'protocol.json'))
    if p.exists():
        x=read_json(p);assert x['status']=='completed' and x['identity']==identity,'Uncertain forward'
        assert sha256(rawpath)==x['raw_sha256'] and sha256(projectpath)==x['projected_sha256'];return np.load(rawpath,allow_pickle=False),np.load(projectpath,allow_pickle=False)
    pixels=scorer.processor(images=images,return_tensors='pt')['pixel_values'].to('cuda',torch.float32)
    durable(p,dict(status='started_may_forward',identity=identity,utc=utc()));torch.cuda.synchronize();started=time.perf_counter()
    # 步骤1：官方ITC使用last_hidden_state首位置，不使用重复LayerNorm的pooler_output。
    with torch.inference_mode():
        cls=scorer.model.vision_model(pixel_values=pixels,return_dict=True).last_hidden_state[:,0,:]
        projected=torch.nn.functional.normalize(scorer.model.vision_proj(cls),p=2,dim=-1).cpu().numpy();raw=cls.cpu().numpy()
    torch.cuda.synchronize();seconds=time.perf_counter()-started;save_array(rawpath,raw);save_array(projectpath,projected)
    durable(p,dict(status='completed',identity=identity,raw_sha256=sha256(rawpath),projected_sha256=sha256(projectpath),seconds=seconds,utc=utc()))
    return raw,projected


def execute(run,gpu):
    """接口：先合成验证代数，再核对310个实际缓存向量，失败不调整容差。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'run.lock'):
        s=prepare(run,gpu)
        if (run/'summary.json').exists():
            for p,h in read_json(run/'summary.json')['artifacts'].items():assert sha256(p)==h
            print('Complete recovery, extra visual forwards 0',flush=True);return
        memory=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if memory>256:raise RuntimeError('GPU not free')
        os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');offline_environment()
        import numpy as np
        import torch
        from PIL import Image
        from videoqa_methods.model_ops import FeatureScorer
        from videoqa_budget32.selection import configure_visual_precision
        torch.set_num_threads(2);torch.set_num_interop_threads(1);precision=configure_visual_precision(torch)
        params=load_projection_parameters(MODEL/'pytorch_model.bin');scorer=FeatureScorer(str(MODEL));started=time.perf_counter()
        images=[Image.new('RGB',(1280,720),color) for color in ['gray','red','blue','white']*4]
        raw,direct=forward_once(run,'synthetic',0,scorer,images,list(range(16)))
        for im in images:im.close()
        normalized=raw/np.linalg.norm(raw,axis=1,keepdims=True);recovered,scales=recover_projected_features(normalized,params)
        synthetic_error=float(np.max(np.abs(recovered-direct)));assert synthetic_error<=1e-6,('Synthetic recovery failed',synthetic_error)
        summaries={}
        # 步骤2：同时比较当前CLS恢复及旧缓存恢复，分清代数误差和历史批次浮点差异。
        for q in s['question_ids']:
            item=s['inputs'][q];pool=read_json(item['pool']);cached=np.load(item['features'],allow_pickle=False);bank=item['bank_indices'];errors=[];output=[]
            for start in range(0,len(bank),16):
                ids=bank[start:start+16];images,_=FrameProvider(dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in ids])).decode()
                raw,direct=forward_once(run,q,start,scorer,images,ids)
                for im in images:im.close()
                unit=raw/np.linalg.norm(raw,axis=1,keepdims=True);fresh,scale=recover_projected_features(unit,params);restored,oldscale=recover_projected_features(cached[ids],params)
                row=dict(start=start,pure_error=float(np.max(np.abs(unit-cached[ids]))),fresh_projection_error=float(np.max(np.abs(fresh-direct))),
                    cached_projection_error=float(np.max(np.abs(restored-direct))),raw_norm_error=float(np.max(np.abs(scale-np.linalg.norm(raw,axis=1)))),minimum_scale=float(oldscale.min()),maximum_scale=float(oldscale.max()))
                if max(row['pure_error'],row['fresh_projection_error'],row['cached_projection_error'])>1e-6:raise ValueError(('Projection recovery mismatch',q,row))
                errors.append(row);output.extend(restored);durable(run/'status.json',dict(status='validating',question_id=q,completed_frames=start+len(ids),pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            save_array(run/'features'/(q+'.npy'),np.asarray(output,dtype=np.float32));summaries[q]=errors;durable(run/'results'/(q+'.json'),dict(question_id=q,bank_indices=bank,errors=errors))
        # 步骤3：仅宣告投影恢复数值验收，不宣告语义、选帧或正确率改善。
        calls=[read_json(p) for p in (run/'calls').rglob('*.json')];assert len(calls)==24 and all(x['status']=='completed' for x in calls)
        artifacts={str(p):sha256(p) for folder in ('calls','features','results','code_snapshot') for p in sorted((run/folder).rglob('*')) if p.is_file()}
        durable(run/'summary.json',dict(status='completed',question_errors=summaries,synthetic_projection_error=synthetic_error,visual_frames=310,visual_batches=23,
            synthetic_frames=16,synthetic_batches=1,itm_calls=0,qa_calls=0,precision=precision,seconds=time.perf_counter()-started,artifacts=artifacts,utc=utc()))
        durable(run/'status.json',dict(status='completed',pid=os.getpid(),utc=utc()));print('Projection validation passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--gpu',default='0');a=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id) and a.gpu in list(map(str,range(8)))
    run=ROOT/'outputs/analysis'/a.run_id
    try:execute(run,a.gpu)
    except BaseException as exc:durable(run/'failure.json',dict(error=repr(exc),utc=utc()));raise
