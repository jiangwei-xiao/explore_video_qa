#!/usr/bin/env python3
"""固定原问题窗口的局部查询诊断：仅补齐既有细帧评分，禁止修改选帧或调用问答。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import sys
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,utc,rows
from videoqa_full.prepare import environment
from videoqa_audit import query_input_probe as base
QUESTIONS=['103-1','395-2','409-3','383-2','260-2','170-2','544-1','004-3','893-1','872-2']
JOINED=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1'
OPTIONS=ROOT/'outputs/analysis/option_only_dev200_20261003_r1'


def verify(run):
    """接口：按已冻结源图、粗分数、模型和代码校验恢复，不重新评分粗候选。"""
    spec=read_json(run/'protocol.json')
    assert environment(base.PYTHON)==spec['environment']
    for p,h in spec['sources'].items():assert sha256(p)==h,p
    return spec


def prepare(run):
    """接口：冻结十题和完整原池；新调用仅为八个文本视角对既有细帧的评分。"""
    from transformers import BlipProcessor
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():return verify(run)
        assert read_json(OPTIONS/'summary.json')['status']=='completed'
        data={r['question_id']:r for r in rows()}
        tokenizer=BlipProcessor.from_pretrained(base.MODEL,local_files_only=True).tokenizer
        sources=dict(read_json(ROOT/'outputs/analysis/query_input_pilot9_20261001_r1/protocol.json')['model_files'])
        for name in ['scripts/run_local_option_probe.py','src/videoqa_audit/query_input_probe.py',
                     'src/videoqa_methods/model_ops.py','src/videoqa_runtime/baseline_selection.py','src/videoqa_budget32/selection.py']:
            sources[str(ROOT/name)]=sha256(ROOT/name)
        pairs=batches=fine_count=0
        for q in QUESTIONS:
            pp=base.BASE/'pools'/(q+'.json')
            fp=base.BASE/'features'/(q+'.npy')
            cp=base.BASE/'selection_checkpoints'/(q+'.json')
            check=read_json(cp)
            for p in [pp,fp]:assert sha256(p)==check['files'][str(p.relative_to(base.BASE))]
            pool=read_json(pp)
            assets=pool['new_assets']
            n=pool['initial_count']
            assert len(assets)==len(pool['candidates'])-n
            assert [a['candidate_index'] for a in assets]==list(range(n,len(pool['candidates'])))
            views={'joined':[data[q]['question']+'\n'+o for o in data[q]['options']], 'option':data[q]['options']}
            assert read_json(JOINED/'inputs'/(q+'.json'))['queries']==views['joined']
            assert read_json(OPTIONS/'inputs'/(q+'.json'))['queries']==views['option']
            assert max(len(tokenizer(t,truncation=False)['input_ids']) for texts in views.values() for t in texts)<=512
            paths=[pp,fp,cp,JOINED/'results'/(q+'.json'),OPTIONS/'results'/(q+'.json'),JOINED/'inputs'/(q+'.json'),OPTIONS/'inputs'/(q+'.json')]
            for a in assets:
                assert sha256(a['path'])==a['sha256']
                sources[a['path']]=a['sha256']
            for p in paths:sources[str(p)]=sha256(p)
            ip=run/'inputs'/(q+'.json')
            durable(ip,dict(question_id=q,pool=str(pp),features=str(fp),queries=views,coarse_count=n,
                coarse_scores={name:str(parent/'results'/(q+'.json')) for name,parent in [('joined',JOINED),('option',OPTIONS)]}))
            sources[str(ip)]=sha256(ip)
            fine_count+=len(assets)
            pairs+=8*len(assets)
            batches+=8*((len(assets)+15)//16)
        spec=dict(question_ids=QUESTIONS,sources=sources,environment=environment(base.PYTHON),
            assignment={'0':QUESTIONS[::2],'1':QUESTIONS[1::2]},fine_frames=fine_count,new_frame_text_pairs=pairs,
            real_batches=batches,synthetic_batches=2,qa_calls=0,new_sampling=0,
            scope='Original-question frozen M32/K32 pools and windows; compare joined and option-only queries locally; no selector change',utc=utc())
        durable(run/'protocol.json',spec)
        verify(run)
        print(dict(fine_frames=fine_count,new_pairs=pairs,real_batches=batches,qa_calls=0),flush=True)
        return spec


def worker(run,gpu,questions):
    """接口：单卡常驻BLIP，按原细帧批次对称评分，纯视觉不一致立即停止。"""
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    offline_environment()
    import torch
    from PIL import Image
    from videoqa_methods.model_ops import FeatureScorer
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    base.configure_visual_precision(torch)
    scorer=FeatureScorer(str(base.MODEL))
    try:
        syn=run/'synthetic'/(gpu+'.json')
        if syn.exists():assert read_json(syn)['status']=='completed'
        else:
            durable(syn,dict(status='started',frames=16,utc=utc()))
            images=[Image.new('RGB',(1280,720),'gray') for _ in range(16)]
            scorer.score_batch_features(scorer.tokenize('Which synthetic color is visible?'),images)
            for im in images:im.close()
            durable(syn,dict(status='completed',frames=16,utc=utc()))
        for q in questions:
            dest=run/'results'/(q+'.json')
            if dest.exists():
                assert read_json(dest)['protocol_sha256']==sha256(run/'protocol.json')
                continue
            item=read_json(run/'inputs'/(q+'.json'))
            pool=read_json(item['pool'])
            n=item['coarse_count']
            features=np.load(item['features'],allow_pickle=False)
            scores={name:[list(v) for v in read_json(p)['scores']] for name,p in item['coarse_scores'].items()}
            assert all(len(v)==n for views in scores.values() for v in views)
            encoded={name:[scorer.tokenize(t) for t in texts] for name,texts in item['queries'].items()}
            error=0.
            # 步骤1：不新增解码或补查；仅加载原来实际生成的细帧PNG，顺序与原特征对应。
            for start in range(0,len(pool['new_assets']),16):
                assert not (run/'STOP.json').exists()
                assets=pool['new_assets'][start:start+16]
                images=[]
                for a in assets:
                    assert sha256(a['path'])==a['sha256']
                    with Image.open(a['path']) as im:images.append(im.convert('RGB'))
                hashes=[hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in images]
                # 步骤2：两种文本各四路全部评分，原视觉特征及原选帧不被覆盖。
                for name in ['joined','option']:
                    for k in range(4):
                        values,vectors=base.score_once(run,q,name+'_'+str(k),n+start,scorer,encoded[name][k],images,gpu,hashes)
                        error=max(error,float(np.max(np.abs(vectors-features[n+start:n+start+len(images)]))))
                        assert error<=1e-6,(q,start,error)
                        scores[name][k].extend(values)
                for im in images:im.close()
            assert all(len(v)==len(pool['candidates']) for views in scores.values() for v in views)
            durable(dest,dict(status='completed',question_id=q,scores=scores,max_feature_error=error,
                protocol_sha256=sha256(run/'protocol.json'),qa_calls=0,utc=utc()))
    except BaseException as exc:
        durable(run/'STOP.json',dict(error=repr(exc),gpu=gpu,utc=utc()))
        raise


def execute(run):
    """接口：后台两卡执行有限细帧评分；完成恢复只验收，未知前向不重试。"""
    from videoqa_runtime.baseline_run import available_gpus
    with exclusive(run/'coordinator.lock'):
        spec=verify(run)
        if (run/'summary.json').exists():
            for p,h in read_json(run/'summary.json')['artifacts'].items():assert sha256(p)==h
            print('Completed recovery:0 forwards',flush=True)
            return
        assert not (run/'STOP.json').exists()
        free,_=available_gpus(list(spec['assignment']))
        assert set(spec['assignment'])<=set(free)
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as ex:
            pending={ex.submit(worker,run,g,qs) for g,qs in spec['assignment'].items()}
            while pending:
                done,pending=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                for f in done:f.result()
                durable(run/'status.json',dict(status='running',completed=len(list((run/'results').glob('*.json'))),pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤3：预算及全部题目闭合后保存完整评分，不生成最终选择或答案。
        verify(run)
        calls=list((run/'calls').glob('*/*/*.json'))
        assert len(calls)==spec['real_batches'] and all(read_json(p)['status']=='completed' for p in calls)
        assert sum(read_json(p)['identity']['frames'] for p in calls)==spec['new_frame_text_pairs']
        assert len(list((run/'results').glob('*.json')))==10
        artifacts={str(p):sha256(p) for folder in ['calls','results','synthetic'] for p in (run/folder).rglob('*') if p.is_file()}
        durable(run/'summary.json',dict(status='completed',questions=10,qa_calls=0,artifacts=artifacts,utc=utc()))
        durable(run/'status.json',dict(status='completed',completed=10,pid=os.getpid(),sid=os.getsid(0),utc=utc()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',required=True)
    p.add_argument('--stage',choices=['prepare','run'],required=True)
    args=p.parse_args()
    import re
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id)
    run=ROOT/'outputs/analysis'/args.run_id
    if args.stage=='prepare':prepare(run)
    else:execute(run)
