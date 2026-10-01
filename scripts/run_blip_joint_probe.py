#!/usr/bin/env python3
"""六题固定帧库的BLIP表征诊断，零问答，不改变正式算法或旧特征。"""
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
from videoqa_full.state import durable,exclusive,utc,save_array,rows
from videoqa_full.prepare import environment
from videoqa_audit.query_input_probe import BASE,MODEL,PYTHON
from videoqa_budget32.frames import FrameProvider
IDS=['103-1','395-2','383-2','538-1','630-1','597-1']
PARENT=ROOT/'outputs/analysis/relevance_coverage_dev200_20261002_r1'
FACTS=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'


def prepare(run,gpu):
    """接口：冻结原Top32、exp12输入及已知见证的并集，供诊断而非生产选帧。"""
    pp=run/'protocol.json'
    if pp.exists():
        s=read_json(pp)
        for p,h in {**s['sources'],**s['model_files']}.items():assert sha256(p)==h,p
        assert s['environment']==environment(PYTHON) and s['gpu']==gpu
        return s
    data={x['question_id']:x for x in rows()};base=read_json(BASE/'protocol.json');facts=read_json(FACTS)['entries'];inputs={};sources={str(FACTS):sha256(FACTS)}
    for q in IDS:
        poolpath=BASE/'pools'/(q+'.json');featurepath=BASE/'features'/(q+'.npy');oldpath=PARENT/'questions'/(q+'.json')
        pool=read_json(poolpath);old=read_json(oldpath);c=pool['candidates'];n=pool['initial_count']
        top=sorted(range(n),key=lambda i:(-pool['initial_scores'][i],c[i]['source_pts'],i))[:32]
        witness={i for f in facts if f['question_id']==q for i in f['initial_indices']}
        if q=='395-2':witness.update(i for i,x in enumerate(c) if abs(x['timestamp_seconds']-272.75)<1e-8)
        bank=sorted(set(top+old['selected_indices'])|witness,key=lambda i:(c[i]['source_pts'],i))
        inputs[q]=dict(question=data[q]['question'],pool=str(poolpath),features=str(featurepath),bank_indices=bank,
            selected_indices=old['selected_indices'],witness_indices=sorted(witness))
        for p in (poolpath,featurepath,oldpath):sources[str(p)]=sha256(p)
        video=pool['video']['path'];h=sha256(video);assert h==base['video_hashes'][data[q]['video_id']];sources[video]=h
    # 步骤1：新旧表征来自同一前向；固定原始L2归一化，不根据结果中心化或白化。
    names=['scripts/run_blip_joint_probe.py','src/videoqa_audit/blip_joint_probe.py','src/videoqa_methods/model_ops.py','src/videoqa_runtime/baseline_selection.py']
    for name in names:
        p=ROOT/name;sources[str(p)]=sha256(p);d=run/'code_snapshot'/name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
    files={str(p):sha256(p) for p in sorted(MODEL.iterdir()) if p.is_file() and p.suffix in ('.json','.txt','.bin','.safetensors')}
    s=dict(question_ids=IDS,inputs=inputs,sources=sources,model_files=files,environment=environment(PYTHON),gpu=gpu,
        frame_text_calls=sum(len(x['bank_indices']) for x in inputs.values()),batch_calls=sum((len(x['bank_indices'])+15)//16 for x in inputs.values()),
        synthetic_batches=2,qa_calls=0,batch_size=16,fp32_tolerance=1e-6,
        scope='Retrospective fixed diagnostic bank includes known witnesses, not an automatic selector or QA experiment',utc=utc())
    durable(pp,s);return s


def score_once(run,label,start,scorer,encoded,images,ids,gpu):
    """接口：一次前向事务，开始前落盘，未知返回不自动重试。"""
    import numpy as np
    from videoqa_audit.blip_joint_probe import score_dual_embeddings
    p=run/'calls'/label/(str(start)+'.json');pure=p.with_suffix('.pure.npy');joint=p.with_suffix('.joint.npy')
    identity=dict(ids=ids,rgb_sha256=[hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest() for im in images],
        token_ids=encoded['input_ids'][0].tolist(),gpu=gpu,protocol_sha256=sha256(run/'protocol.json'))
    if p.exists():
        saved=read_json(p);assert saved['status']=='completed' and saved['identity']==identity,'Uncertain BLIP forward; no retry'
        assert sha256(pure)==saved['pure_sha256'] and sha256(joint)==saved['joint_sha256']
        return saved['scores'],np.load(pure,allow_pickle=False),np.load(joint,allow_pickle=False)
    # 步骤1：调用额度按已开始批次消耗；步骤2：先保存返回资产，再确认完成。
    durable(p,dict(status='started_may_forward',identity=identity,utc=utc()))
    scores,a,b,pre,forward=score_dual_embeddings(scorer,encoded,images);save_array(pure,a);save_array(joint,b)
    durable(p,dict(status='completed',identity=identity,scores=scores,pure_sha256=sha256(pure),joint_sha256=sha256(joint),preprocess_seconds=pre,forward_seconds=forward,utc=utc()))
    return scores,a,b


def execute(run,gpu):
    """接口：单空闲GPU完成六题固定帧库；历史数值不符暂停，禁止追加问答。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'run.lock'):
        s=prepare(run,gpu)
        if (run/'summary.json').exists():
            for p,h in read_json(run/'summary.json')['artifacts'].items():assert sha256(p)==h
            print('Complete recovery, extra forwards 0',flush=True);return
        memory=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if memory>256:raise RuntimeError('Requested GPU not free; pause')
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
        # 步骤3：精确取冻结源帧，新旧纯视觉特征/原ITM必须重放一致。
        for q in IDS:
            item=s['inputs'][q];pool=read_json(item['pool']);frozen=np.load(item['features'],allow_pickle=False);encoded=scorer.tokenize(item['question'])
            vectors=[];joint_vectors=[];scores=[];errors=[];bank=item['bank_indices']
            for begin in range(0,len(bank),16):
                ids=bank[begin:begin+16];images,_=FrameProvider(dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in ids])).decode()
                values,a,b=score_once(run,q,begin,scorer,encoded,images,ids,gpu)
                for im in images:im.close()
                err=float(np.max(np.abs(a-frozen[ids])));serr=max([abs(v-pool['initial_scores'][i]) for i,v in zip(ids,values) if i<pool['initial_count']]+[0.])
                if max(err,serr)>1e-6:raise ValueError(('Historical feature/score mismatch',q,begin,err,serr))
                errors.append(dict(start=begin,feature_error=err,score_error=serr));vectors.extend(a);joint_vectors.extend(b);scores.extend(values)
                durable(run/'status.json',dict(status='scoring',question_id=q,completed_in_question=begin+len(ids),pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            a=np.asarray(vectors);b=np.asarray(joint_vectors);path=run/'features'/(q+'.pure.npy');other=run/'features'/(q+'.joint.npy');save_array(path,a);save_array(other,b)
            pos={i:j for j,i in enumerate(bank)};nearest={}
            for i in item['witness_indices']:
                peers=[j for j in item['selected_indices'] if j!=i];neighbors={}
                for name,x in [('pure',a),('joint',b)]:
                    values=x[[pos[j] for j in peers]]@x[pos[i]];order=sorted(range(len(peers)),key=lambda k:(-float(values[k]),pool['candidates'][peers[k]]['source_pts']))[:5]
                    neighbors[name]=[dict(candidate_index=peers[k],timestamp_seconds=pool['candidates'][peers[k]]['timestamp_seconds'],cosine=float(values[k])) for k in order]
                nearest[str(i)]=neighbors
            summaries[q]=dict(frames=len(bank),pure=geometry_summary(a),joint=geometry_summary(b),errors=errors,nearest_selected_to_witness=nearest)
            durable(run/'results'/(q+'.json'),dict(question_id=q,bank_indices=bank,scores=scores,**summaries[q]))
        # 步骤4：所有新前向与合成分开计数，只给几何诊断，不伪造选帧或准确率。
        calls=[read_json(p) for p in (run/'calls').rglob('*.json')];assert len(calls)==s['batch_calls']+2 and all(x['status']=='completed' for x in calls)
        artifacts={str(p):sha256(p) for folder in ('calls','features','results','code_snapshot') for p in sorted((run/folder).rglob('*')) if p.is_file()}
        durable(run/'summary.json',dict(status='completed',questions=summaries,frame_text_calls=s['frame_text_calls'],batch_calls=s['batch_calls'],synthetic_batches=2,
            qa_calls=0,precision=precision,seconds=time.perf_counter()-started,artifacts=artifacts,utc=utc()))
        durable(run/'status.json',dict(status='completed',pid=os.getpid(),utc=utc()));print('Completed',s['frame_text_calls'],'frame-texts',s['batch_calls'],'batches',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--gpu',default='0');a=p.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id) and a.gpu in list(map(str,range(8)))
    run=ROOT/'outputs/analysis'/a.run_id
    try:execute(run,a.gpu)
    except BaseException as exc:durable(run/'failure.json',dict(error=repr(exc),utc=utc()));raise
