#!/usr/bin/env python3
"""同一人工帧组库的BLIP信号控制，所有选项对称评分，不产生新问答。"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,utc
from videoqa_full.prepare import environment
from videoqa_audit.framegroup_design import create_design
MODEL=Path('/home/models/Salesforce/blip-itm-base-coco')
PYTHON='/App/conda/envs/explore_video_qa/bin/python'


def score_once(run,scorer,text,images,ids,label,gpu):
    """接口：同次官方ITM前向保存概率和原始正负logit差，未确认返回不自动重试。"""
    import torch
    if not 1<=len(images)<=16 or len(ids)!=len(images):raise ValueError('Invalid frozen BLIP batch')
    encoded=scorer.tokenize(text);path=run/'blip_calls'/(label+'.json')
    identity=dict(frame_ids=ids,text=text,token_ids=encoded['input_ids'][0].tolist(),gpu=gpu,protocol_sha256=sha256(run/'blip_protocol.json'),
        rgb_sha256=[hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest() for im in images])
    if path.exists():
        old=read_json(path);assert old['identity']==identity and old['status']=='completed','Uncertain BLIP forward';return old
    # 步骤1：保持原FP32/384/无截断设置；本探针重新评分，不覆写历史粗分数。
    pixels=scorer.processor(images=images,return_tensors='pt')['pixel_values'].to('cuda',torch.float32)
    inputs={k:v.repeat(len(images),1).cuda() for k,v in encoded.items() if k in ('input_ids','attention_mask')}
    durable(path,dict(status='started_may_forward',identity=identity,utc=utc()));torch.cuda.synchronize();started=time.perf_counter()
    with torch.inference_mode():
        output=scorer.model(pixel_values=pixels,**inputs,use_itm_head=True).itm_score
        scores=output.softmax(-1)[:,1].cpu().tolist();margins=(output[:,1]-output[:,0]).cpu().tolist()
    torch.cuda.synchronize()
    if not np.isfinite(scores+margins).all() or any(not 0<=p<=1 for p in scores):raise ValueError('Invalid ITM output')
    result=dict(status='completed',identity=identity,scores=scores,match_log_odds=margins,forward_seconds=time.perf_counter()-started,utc=utc());durable(path,result);return result


def execute(run,gpu):
    """接口：固定32唯一帧、五查询视角，共160帧文本对/15批，另一次合成；零问答。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'blip_run.lock'):
        design=create_design(run);assets=read_json(design['asset_manifest']);sources=dict(assets['sources'])
        for p in (run/'design_protocol_r2.json',Path(__file__).resolve(),ROOT/'src/videoqa_runtime/baseline_selection.py',ROOT/'src/videoqa_budget32/selection.py'):sources[str(p)]=sha256(p)
        files={str(p):sha256(p) for p in MODEL.iterdir() if p.is_file() and p.suffix in ('.json','.txt','.bin')}
        assert files==read_json(ROOT/'outputs/analysis/blip_joint_native_bank6_20261002_r1/protocol.json')['model_files']
        spec=dict(sources=sources,model_files=files,environment=environment(PYTHON),gpu=gpu,real_frame_text_pairs=160,real_batches=15,synthetic_batches=1,qa_calls=0,
            note='Fresh controlled-bank scores, not replacement/replay of original full coarse scores; options symmetric')
        pp=run/'blip_protocol.json'
        if pp.exists():assert read_json(pp)==spec
        else:durable(pp,spec)
        if (run/'blip_summary.json').exists():
            for p,h in read_json(run/'blip_summary.json')['artifacts'].items():assert sha256(p)==h
            print('Completed BLIP recovery, extra forwards 0',flush=True);return
        import subprocess
        memory,util=map(int,subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
        if memory>=256 or util:raise RuntimeError('GPU not free')
        os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');offline_environment()
        import torch
        from videoqa_runtime.baseline_selection import BlipItmScorer
        from videoqa_budget32.selection import configure_visual_precision
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(2027);precision=configure_visual_precision(torch);started=time.perf_counter();scorer=BlipItmScorer(str(MODEL))
        assert scorer.model.dtype==torch.float32 and not scorer.model.training
        durable(run/'blip_load.json',dict(precision=precision,dtype=str(scorer.model.dtype),load_seconds=time.perf_counter()-started,pid=os.getpid(),sid=os.getsid(0),gpu=gpu))
        images=[Image.new('RGB',(1280,720),'gray') for _ in range(16)];score_once(run,scorer,'Which synthetic color is visible?',images,list(range(16)),'synthetic',gpu)
        for im in images:im.close()
        results={};groups={};pair_count=0;batch_count=0
        # 步骤2：全部原问题和选项视角同批像素，原始分数完整保留，不按正确选项选择分支。
        for q,case in assets['cases'].items():
            pts=sorted(case['frames'],key=int);results[q]={};groups[q]={}
            for view,text in design['queries'][q].items():
                values={};margins={}
                for start in range(0,len(pts),16):
                    ids=pts[start:start+16];images=[]
                    for p in ids:
                        row=case['frames'][p]
                        with Image.open(row['file']) as im:images.append(im.convert('RGB'))
                        assert hashlib.sha256(np.asarray(images[-1]).tobytes()).hexdigest()==row['rgb_sha256']
                    r=score_once(run,scorer,text,images,[q+'__'+p for p in ids],q+'__'+view+'__'+str(start),gpu)
                    values.update(zip(ids,r['scores']));margins.update(zip(ids,r['match_log_odds']));pair_count+=len(ids);batch_count+=1
                    for im in images:im.close()
                results[q][view]=dict(scores=values,match_log_odds=margins)
                groups[q][view]={name:dict(max=max(values[p] for p in ids),mean=float(np.mean([values[p] for p in ids]))) for name,ids in case['clips'].items()}
        # 步骤3：只汇总逐帧聚合，不预测答案，不把匹配概率解释为证据或答案概率。
        assert pair_count==160 and batch_count==15 and len(list((run/'blip_calls').glob('*.json')))==16
        durable(run/'blip_summary.json',dict(status='completed',scores=results,group_scores=groups,real_frame_text_pairs=160,real_batches=15,synthetic_batches=1,synthetic_frames=16,qa_calls=0,
            seconds=time.perf_counter()-started,artifacts={str(p):sha256(p) for p in (run/'blip_calls').glob('*.json')},utc=utc()))
        print('BLIP control complete:160 pairs,15 batches, QA0',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--gpu',default='1');a=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id) and a.gpu in list(map(str,range(8)))
    run=ROOT/'outputs/analysis'/a.run_id
    try:execute(run,a.gpu)
    except BaseException as exc:durable(run/('blip_failure_'+str(time.time_ns())+'.json'),dict(error=repr(exc),utc=utc()));raise
