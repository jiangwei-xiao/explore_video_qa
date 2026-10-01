#!/usr/bin/env python3
"""同一310帧诊断库，按历史粗批次做严格重放，额外上下文帧只为数值一致。"""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,utc,save_array
from videoqa_full.prepare import environment
from videoqa_budget32.frames import FrameProvider
from run_blip_joint_probe import score_once,MODEL,PYTHON,IDS
PARENT=ROOT/'outputs/analysis/blip_joint_bank6_20261002_r1'
CONTROL=ROOT/'outputs/analysis/blip_joint_native_control_20261002_r1'


def native_groups(pool,bank):
    """接口：粗帧按原始16帧块闭包取齐，细帧另分组，禁止混合破坏末批形状。"""
    n=pool['initial_count'];c=pool['candidates']
    if not bank or len(bank)!=len(set(bank)) or min(bank)<0 or max(bank)>=len(c):raise ValueError('Invalid bank')
    # 步骤1：粗块保留原始最后不足16帧的形状；步骤2：细帧只查纯视觉重放，没有旧ITM。
    groups=[dict(label='coarse_'+str(start),ids=list(range(start,min(n,start+16)))) for start in sorted({i//16*16 for i in bank if i<n})]
    fine=sorted((i for i in bank if i>=n),key=lambda i:(c[i]['source_pts'],i))
    groups.extend(dict(label='fine_'+str(start),ids=fine[start:start+16]) for start in range(0,len(fine),16))
    flat=[i for g in groups for i in g['ids']]
    if len(flat)!=len(set(flat)) or not set(bank)<=set(flat):raise ValueError('Batch identity mismatch')
    return groups


def prepare(run,gpu):
    """接口：原诊断库不变，冻结新的计算批次和完整调用额度，不放宽容差。"""
    pp=run/'protocol.json'
    if pp.exists():
        s=read_json(pp)
        for p,h in {**s['sources'],**s['model_files']}.items():assert sha256(p)==h,p
        assert s['environment']==environment(PYTHON) and s['gpu']==gpu;return s
    old=read_json(PARENT/'protocol.json');assert read_json(CONTROL/'result.json')['passed']
    for p,h in {**old['sources'],**old['model_files']}.items():assert sha256(p)==h,p
    inputs={q:dict(v,groups=native_groups(read_json(v['pool']),v['bank_indices'])) for q,v in old['inputs'].items()}
    sources=dict(old['sources']);sources[str(PARENT/'protocol.json')]=sha256(PARENT/'protocol.json');sources[str(CONTROL/'result.json')]=sha256(CONTROL/'result.json');sources[str(Path(__file__).resolve())]=sha256(__file__)
    previous=[p for p in (PARENT/'calls').rglob('*.json') if p.parent.name!='synthetic']
    assert sum(len(read_json(p)['identity']['ids']) for p in previous)==53
    for p in (PARENT/'calls').rglob('*'):
        if p.is_file():sources[str(p)]=sha256(p)
    for name in ['scripts/run_blip_joint_native_probe.py','scripts/run_blip_joint_probe.py','src/videoqa_audit/blip_joint_probe.py']:
        dest=run/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    s=dict(question_ids=IDS,inputs=inputs,sources=sources,model_files=old['model_files'],environment=old['environment'],gpu=gpu,
        evaluation_frames=old['frame_text_calls'],frame_text_calls=sum(len(g['ids']) for v in inputs.values() for g in v['groups']),
        batch_calls=sum(len(v['groups']) for v in inputs.values()),synthetic_batches=2,qa_calls=0,fp32_tolerance=1e-6,
        scope='Same frozen diagnostic bank; extra native coarse batch members only for numerical replay; not a production selector',utc=utc())
    durable(pp,s);return s


def execute(run,gpu):
    """接口：独立后台完成匹配批次诊断，不复用失败轮的新分数为旧结果，不生成答案。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'run.lock'):
        s=prepare(run,gpu)
        if (run/'summary.json').exists():
            for p,h in read_json(run/'summary.json')['artifacts'].items():assert sha256(p)==h
            print('Complete recovery, extra forwards 0',flush=True);return
        memory=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if memory>256:raise RuntimeError('GPU not free; pause')
        os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');offline_environment()
        import numpy as np
        import torch
        from PIL import Image
        from videoqa_methods.model_ops import FeatureScorer
        from videoqa_budget32.selection import configure_visual_precision
        from videoqa_audit.blip_joint_probe import geometry_summary
        torch.set_num_threads(2);torch.set_num_interop_threads(1);precision=configure_visual_precision(torch);started=time.perf_counter();scorer=FeatureScorer(str(MODEL))
        for k in range(2):
            images=[Image.new('RGB',(1280,720),'gray') for _ in range(16)]
            score_once(run,'synthetic',k,scorer,scorer.tokenize('Which synthetic color is visible?'),images,list(range(16)),gpu)
            for im in images:im.close()
        summaries={}
        # 步骤1：每批实际源身份和旧数值校验；所有已开始批次失败也留在账本。
        for q in IDS:
            item=s['inputs'][q];pool=read_json(item['pool']);frozen=np.load(item['features'],allow_pickle=False);encoded=scorer.tokenize(item['question']);values_by_id={};errors=[]
            for g in item['groups']:
                ids=g['ids'];images,_=FrameProvider(dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in ids])).decode()
                scores,a,b=score_once(run,q,g['label'],scorer,encoded,images,ids,gpu)
                for im in images:im.close()
                err=float(np.max(np.abs(a-frozen[ids])));serr=max([abs(v-pool['initial_scores'][i]) for i,v in zip(ids,scores) if i<pool['initial_count']]+[0.])
                if max(err,serr)>1e-6:raise ValueError(('Native replay mismatch',q,g['label'],err,serr))
                errors.append(dict(batch=g['label'],feature_error=err,score_error=serr))
                for i,v,x,y in zip(ids,scores,a,b):values_by_id[i]=(v,x,y)
                durable(run/'status.json',dict(status='scoring',question_id=q,completed_groups=len(errors),pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            # 步骤2：几何统计仅取原310帧库，额外批次成员不能改变评价分母。
            bank=item['bank_indices'];a=np.asarray([values_by_id[i][1] for i in bank]);b=np.asarray([values_by_id[i][2] for i in bank]);scores=[values_by_id[i][0] for i in bank]
            save_array(run/'features'/(q+'.pure.npy'),a);save_array(run/'features'/(q+'.joint.npy'),b);pos={i:j for j,i in enumerate(bank)};nearest={}
            for i in item['witness_indices']:
                peers=[j for j in item['selected_indices'] if j!=i];neighbors={}
                for name,x in [('pure',a),('joint',b)]:
                    val=x[[pos[j] for j in peers]]@x[pos[i]];order=sorted(range(len(peers)),key=lambda k:(-float(val[k]),pool['candidates'][peers[k]]['source_pts']))[:5]
                    neighbors[name]=[dict(candidate_index=peers[k],timestamp_seconds=pool['candidates'][peers[k]]['timestamp_seconds'],cosine=float(val[k])) for k in order]
                nearest[str(i)]=neighbors
            summaries[q]=dict(frames=len(bank),pure=geometry_summary(a),joint=geometry_summary(b),errors=errors,nearest_selected_to_witness=nearest)
            durable(run/'results'/(q+'.json'),dict(question_id=q,bank_indices=bank,scores=scores,**summaries[q]))
        # 步骤3：完整账本核对，保留原首轮停止和数值对照的成本，不冒充零成本恢复。
        calls=[read_json(p) for p in (run/'calls').rglob('*.json')];assert len(calls)==s['batch_calls']+2 and all(x['status']=='completed' for x in calls)
        real=[read_json(p) for p in (run/'calls').rglob('*.json') if p.parent.name!='synthetic'];assert sum(len(x['identity']['ids']) for x in real)==s['frame_text_calls']
        artifacts={str(p):sha256(p) for folder in ('calls','features','results','code_snapshot') for p in sorted((run/folder).rglob('*')) if p.is_file()}
        durable(run/'summary.json',dict(status='completed',questions=summaries,evaluation_frames=310,frame_text_calls=s['frame_text_calls'],batch_calls=s['batch_calls'],synthetic_batches=2,qa_calls=0,
            previous_attempt_frame_text_calls=53,native_control_frame_text_calls=12,precision=precision,seconds=time.perf_counter()-started,artifacts=artifacts,utc=utc()))
        durable(run/'status.json',dict(status='completed',pid=os.getpid(),utc=utc()));print('Completed',s['frame_text_calls'],s['batch_calls'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--gpu',default='0');a=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id) and a.gpu in list(map(str,range(8)))
    run=ROOT/'outputs/analysis'/a.run_id
    try:execute(run,a.gpu)
    except BaseException as exc:durable(run/'failure.json',dict(error=repr(exc),utc=utc()));raise
