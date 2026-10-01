#!/usr/bin/env python3
"""两个定位缺口的逐选项诊断；四个分支对称评分，只报告入口，不调用问答。"""
import argparse
import os
from pathlib import Path
import sys
import time
import hashlib
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing as mp
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit import query_input_probe as base
from videoqa_methods.density_selection import source_time
PARENT=ROOT/'outputs/analysis/query_input_pilot9_20261001_r1'
QUESTIONS=['602-2','607-2']


def seed_union(candidates,score_vectors,per_option=8):
    """接口：每个选项取同样深度，再按源身份去重；不借用标准答案、不补足重合名额。"""
    assert len(score_vectors)==4 and per_option==8
    branches=[base.ranking(candidates,s)[:per_option] for s in score_vectors]
    union=sorted(set(i for ids in branches for i in ids),key=lambda i:candidates[i]['source_pts'])
    assert len(union)<=32
    return branches,union


def seed_windows(candidates,seeds,video):
    """接口：仅复用2秒核心连接和±4秒几何，不定义锚点支持度或最终选帧算法。"""
    cores=[];windows=[]
    # 步骤1：按目标时间形成核心，真实PTS确定窗口，与冻结规则一致。
    for i in sorted(seeds,key=lambda i:candidates[i]['source_pts']):
        t=Fraction(str(candidates[i]['requested_seconds']))
        if not cores or t-Fraction(str(candidates[cores[-1][-1]]['requested_seconds']))>2:cores.append([])
        cores[-1].append(i)
    for ids in cores:
        left=max(Fraction(0),source_time(candidates[ids[0]],video)-4)
        right=min(Fraction(video['duration_fraction']),source_time(candidates[ids[-1]],video)+4)
        if windows and left<=windows[-1][1]:windows[-1][1]=max(windows[-1][1],right)
        else:windows.append([left,right])
    # 步骤2：只给出窗口与粗候选可达性，不能将其称为完整方法输入。
    return [dict(left_fraction=str(a),right_fraction=str(b)) for a,b in windows]


def prepare(run):
    """接口：先固定两题、四查询和Top8并集诊断，不按评分结果选择更有利的融合方式。"""
    from transformers import BlipProcessor
    run.mkdir(parents=True,exist_ok=True)
    if (run/'protocol.json').exists():return verify(run)
    parent=base.verify(PARENT);tokenizer=BlipProcessor.from_pretrained(base.MODEL,local_files_only=True).tokenizer
    sources={str(PARENT/'protocol.json'):sha256(PARENT/'protocol.json')};inputs={};frames=batches=0
    for q in QUESTIONS:
        p=PARENT/'inputs'/(q+'.json');item=read_json(p)
        assert 'answer_index' not in item and len(item['options'])==4
        queries=[item['question']+'\n'+option for option in item['options']]
        lengths=[len(tokenizer(x,truncation=False)['input_ids']) for x in queries];assert max(lengths)<=512
        durable(run/'inputs'/(q+'.json'),dict(item,queries=queries,query_tokens=lengths))
        inputs[q]=sha256(run/'inputs'/(q+'.json'))
        for src in (p,PARENT/'results'/(q+'.json')):sources[str(src)]=sha256(src)
        frames+=item['initial_count']*4;batches+=((item['initial_count']+15)//16)*4
    spec=dict(question_ids=QUESTIONS,gpus=['0','1'],input_sha256=inputs,sources=sources,parent_protocol_sha256=sha256(PARENT/'protocol.json'),
        runner_sha256=sha256(Path(__file__)),parent_scorer_sha256=sha256(ROOT/'src/videoqa_audit/query_input_probe.py'),
        frame_text_scores=frames,batch_calls=batches,synthetic_batch_calls=4,per_option_seeds=8,max_union_seeds=32,
        new_qa_calls=0,merge_rule='Top8 of every option; unique source union; no fill or best-option selection',utc=utc())
    import shutil
    shutil.copy2(__file__,run/'runner_snapshot.py');durable(run/'protocol.json',spec)
    return spec


def verify(run):
    """接口：绑定前轮原始评分与模型身份，任何输入变化均停止。"""
    spec=read_json(run/'protocol.json');base.verify(PARENT)
    assert spec['runner_sha256']==sha256(Path(__file__))
    assert spec['parent_scorer_sha256']==sha256(ROOT/'src/videoqa_audit/query_input_probe.py')
    for p,h in spec['sources'].items():assert sha256(p)==h
    for q,h in spec['input_sha256'].items():assert sha256(run/'inputs'/(q+'.json'))==h
    return spec


def worker(job):
    """接口：一题一张固定GPU，所有四选项执行；原题/拼接结果只复用，不重算。"""
    run,q,gpu=Path(job[0]),job[1],job[2]
    os.environ['CUDA_VISIBLE_DEVICES']=gpu;offline_environment()
    if (run/'results'/(q+'.json')).exists():return q
    try:
        import torch
        from PIL import Image
        from videoqa_methods.model_ops import FeatureScorer
        torch.set_num_threads(2);torch.set_num_interop_threads(1);base.configure_visual_precision(torch)
        scorer=FeatureScorer(str(base.MODEL));item=read_json(run/'inputs'/(q+'.json'));pool=read_json(item['pool_path'])
        old=read_json(PARENT/'results'/(q+'.json'));coarse=pool['candidates'][:item['initial_count']]
        features=np.load(item['feature_path'],mmap_mode='r',allow_pickle=False)[:len(coarse)]
        for j in range(2):
            path=run/'synthetic'/f'{gpu}_{j}.json'
            if path.exists():assert read_json(path)['status']=='completed';continue
            durable(path,dict(status='started',gpu=gpu,frames=16,utc=utc()))
            images=[Image.new('RGB',(1280,720),'gray') for _ in range(16)]
            scorer.score_batch_features(scorer.tokenize('Which synthetic color is visible?'),images)
            for im in images:im.close()
            durable(path,dict(status='completed',gpu=gpu,frames=16,utc=utc()))
        encoded=[scorer.tokenize(x) for x in item['queries']];scores=[[] for _ in range(4)];batch=[];seen=0;errors=[];metadata={};started=time.perf_counter()
        def flush():
            """步骤1核对已有RGB，步骤2对称评分四查询；新旧来源不影响名额。"""
            begin=seen-len(batch)
            if (run/'STOP.json').exists():raise RuntimeError('Peer failure')
            hashes=[hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest() for im in batch]
            assert hashes==old['rgb_sha256'][begin:seen]
            for k in range(4):
                values,vectors=base.score_once(run,q,f'option_{k}',begin,scorer,encoded[k],batch,gpu,hashes)
                error=float(np.max(np.abs(vectors-features[begin:seen])));assert error<=1e-6
                errors.append(error);scores[k].extend(values)
            for im in batch:im.close()
            batch.clear();durable(run/'workers'/(q+'.json'),dict(status='scoring',gpu=gpu,pid=os.getpid(),frames=seen,total=len(coarse),utc=utc()))
        for row,frame in base.iter_candidate_frames(pool['video']['path'],metadata):
            for key in ('candidate_index','source_pts','source_frame_index','requested_seconds'):assert row[key]==coarse[seen][key]
            seen+=1;batch.append(frame.to_image())
            if len(batch)==16:flush()
        if batch:flush()
        assert seen==len(coarse)
        branches,union=seed_union(coarse,scores)
        durable(run/'results'/(q+'.json'),dict(status='completed',question_id=q,protocol_sha256=sha256(run/'protocol.json'),scores=scores,
            rankings=[base.ranking(coarse,s) for s in scores],branch_top8=branches,union_seeds=union,
            union_windows=seed_windows(coarse,union,pool['video']),max_feature_error=max(errors),seconds=time.perf_counter()-started,new_qa_calls=0,utc=utc()))
        durable(run/'workers'/(q+'.json'),dict(status='completed',pid=os.getpid(),gpu=gpu,frames=seen,utc=utc()));return q
    except BaseException as exc:
        import traceback
        error=dict(question_id=q,gpu=gpu,error=repr(exc),traceback=traceback.format_exc(),utc=utc())
        durable(run/'failures'/f'{q}_{time.time_ns()}.json',error);durable(run/'STOP.json',error);raise


def summarize(run):
    """接口：分支排名与预算上限相同的联合入口分别报告，不选正确分支当成绩。"""
    spec=verify(run);items=[]
    calls=list((run/'calls').glob('*/*/*.json'));assert len(calls)==spec['batch_calls']
    assert all(read_json(p)['status']=='completed' for p in calls)
    assert sum(read_json(p)['identity']['frames'] for p in calls)==spec['frame_text_scores']
    for q in QUESTIONS:
        result=read_json(run/'results'/(q+'.json'));item=read_json(run/'inputs'/(q+'.json'));pool=read_json(item['pool_path'])
        coarse=pool['candidates'][:item['initial_count']];known=read_json(PARENT/'evaluation.json')['witnesses'][q]
        for name,ids,meaning in known:
            ranks=[min(order.index(i)+1 for i in ids) for order in result['rankings']]
            hit=bool(set(ids)&set(result['union_seeds']))
            window=any(Fraction(w['left_fraction'])<=source_time(coarse[i],pool['video'])<=Fraction(w['right_fraction']) for w in result['union_windows'] for i in ids)
            items.append(dict(question_id=q,witness=name,meaning=meaning,per_option_best_rank=ranks,
                per_option_top8_hit=[bool(set(ids)&set(b)) for b in result['branch_top8']],union_seed_hit=hit,union_window_hit=window,
                union_seed_count=len(result['union_seeds']),union_window_seconds=sum(float(Fraction(w['right_fraction'])-Fraction(w['left_fraction'])) for w in result['union_windows'])))
    summary=dict(status='completed',witnesses=items,frame_text_scores=spec['frame_text_scores'],batch_calls=len(calls),new_qa_calls=0,
        limitations='Two mechanism cases only. Top8 each is symmetric diagnostic, not a full selector or answer accuracy.',utc=utc())
    durable(run/'summary.json',summary);print(summary,flush=True);return summary


def execute(run):
    """接口：后台两进程执行，失败不重试；完成恢复不加载模型。"""
    from videoqa_runtime.baseline_run import available_gpus
    with exclusive(run/'coordinator.lock'):
        spec=verify(run)
        remaining=[q for q in QUESTIONS if not (run/'results'/(q+'.json')).exists()]
        if not remaining:return summarize(run)
        assert not (run/'STOP.json').exists(),'Inspect failure before recovery'
        required=[spec['gpus'][QUESTIONS.index(q)] for q in remaining]
        while True:
            try:free,_=available_gpus(required)
            except RuntimeError:free=[]
            if set(required)<=set(free):break
            durable(run/'status.json',dict(status='waiting_resources',pid=os.getpid(),utc=utc()));time.sleep(15)
        started=time.perf_counter()
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as ex:
            pending={ex.submit(worker,(str(run),q,spec['gpus'][QUESTIONS.index(q)])) for q in remaining}
            while pending:
                done,pending=wait(pending,timeout=3,return_when=FIRST_COMPLETED)
                for f in done:f.result()
                durable(run/'status.json',dict(status='running',completed=len(list((run/'results').glob('*.json'))),pid=os.getpid(),sid=os.getsid(0),seconds=time.perf_counter()-started,utc=utc()))
        summarize(run);durable(run/'status.json',dict(status='completed',completed=2,pid=os.getpid(),sid=os.getsid(0),seconds=time.perf_counter()-started,utc=utc()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=['prepare','run','report'],required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();import re
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id)
    run=ROOT/'outputs/analysis'/a.run_id
    if a.stage=='prepare':prepare(run)
    elif a.stage=='report':summarize(run)
    else:
        if (run/'calls').exists() and not a.resume:raise RuntimeError('Explicit resume required')
        execute(run)
