#!/usr/bin/env python3
"""局部描述覆盖的200题离线入口：补齐细帧评分、选择与恢复，不自动问答。"""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED,as_completed
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,utc,rows
from videoqa_full.prepare import environment
from videoqa_audit import query_input_probe as base
from videoqa_audit.option_structure_support import option_structure
from videoqa_budget32.selection import config
from videoqa_methods.local_query_coverage import select_local_query_coverage
COARSE=ROOT/'outputs/analysis/option_only_dev200_20261003_r1'
PILOT=ROOT/'outputs/analysis/local_option_evidence10_20261003_r1'
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
FACTS=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'


def verify(run):
    """接口：恢复核对固定数据、代码及环境；生成式问答不属于此入口。"""
    spec=read_json(run/'protocol.json')
    assert environment(base.PYTHON)==spec['environment']
    for p,h in spec['sources'].items():assert sha256(p)==h,p
    return spec


def prepare(run):
    """接口：冻结177题启用及23题语法回退，按细帧工作量分配GPU，不用答案路由。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():return verify(run)
        assert read_json(COARSE/'summary.json')['status']=='completed'
        assert read_json(PILOT/'summary.json')['status']=='completed'
        ids=[r['question_id'] for r in read_json(DATA)['rows']]
        data={r['question_id']:r for r in rows()}
        sources=dict(read_json(PILOT/'protocol.json')['sources'])
        for p in [DATA,FACTS,COARSE/'summary.json',PILOT/'summary.json',Path(__file__),
                  ROOT/'src/videoqa_methods/local_query_coverage.py',ROOT/'src/videoqa_audit/option_structure_support.py',
                  ROOT/'src/videoqa_methods/prefix_coverage_budget.py',ROOT/'src/videoqa_methods/option_seed_entry.py']:
            sources[str(p)]=sha256(p)
        inputs={};prepared={};jobs=[];pairs=batches=fine_count=0
        # 步骤1：粗分数全部复用，已有十题细分数复用；回退题不产生虚构细帧分数。
        for q in ids:
            pp=base.BASE/'pools'/(q+'.json');fp=base.BASE/'features'/(q+'.npy')
            cp=base.BASE/'selection_checkpoints'/(q+'.json');check=read_json(cp)
            for p in [pp,fp]:assert sha256(p)==check['files'][str(p.relative_to(base.BASE))]
            pool=read_json(pp);n=pool['initial_count'];assets=pool['new_assets']
            coarse=COARSE/'results'/(q+'.json')
            assert read_json(COARSE/'inputs'/(q+'.json'))['queries']==data[q]['options']
            route=option_structure(data[q]['question'],data[q]['options'])
            enabled=route=='other';cached=PILOT/'results'/(q+'.json')
            for p in [pp,fp,cp,coarse,COARSE/'inputs'/(q+'.json'),base.FULL/'selections/RD-1.2-exp07'/(q+'.json'),base.FULL/'traces'/(q+'.json')]:sources[str(p)]=sha256(p)
            for a in assets:
                assert sha256(a['path'])==a['sha256'];sources[a['path']]=a['sha256']
            item=dict(question_id=q,pool=str(pp),features=str(fp),queries=data[q]['options'],question=data[q]['question'],
                route=route,enabled=enabled,coarse_scores=str(coarse),fine_count=len(assets))
            ip=run/'inputs'/(q+'.json');durable(ip,item);sources[str(ip)]=sha256(ip)
            inputs[q]=dict(pool=str(pp),features=str(fp),input=str(ip))
            if not enabled:
                prepared[q]=dict(scores=None,scoring_scope='disabled; no scores consumed',cached_from=None)
            elif cached.exists():
                assert read_json(PILOT/'inputs'/(q+'.json'))['queries']['option']==item['queries']
                sources[str(cached)]=sha256(cached)
                prepared[q]=dict(scores=read_json(cached)['scores']['option'],scoring_scope='complete original pool',cached_from=str(cached))
            elif not assets:
                prepared[q]=dict(scores=read_json(coarse)['scores'],scoring_scope='coarse-only original pool',cached_from=str(coarse))
            else:
                jobs.append((q,len(assets)));fine_count+=len(assets);pairs+=4*len(assets);batches+=4*((len(assets)+15)//16)
        assignment={str(g):[] for g in range(8)};loads={g:0 for g in assignment}
        for q,count in sorted(jobs,key=lambda x:(-x[1],ids.index(x[0]))):
            g=min(loads,key=lambda g:(loads[g],int(g)));assignment[g].append(q);loads[g]+=count
        spec=dict(question_ids=ids,inputs=inputs,sources=sources,environment=environment(base.PYTHON),configuration=config(32,32),
            assignment=assignment,fine_frames=fine_count,new_frame_text_pairs=pairs,real_batches=batches,
            synthetic_batches=sum(bool(v) for v in assignment.values()),qa_calls=0,
            hypothesis='Protected question anchor; up to four local marginal ordinal description coverage steps; original visual completion and coverage DP; strict numeric/reference fallback',utc=utc())
        durable(run/'protocol.json',spec)
        for q,value in prepared.items():durable(run/'raw_scores'/(q+'.json'),dict(value,status='completed',question_id=q,protocol_sha256=sha256(run/'protocol.json')))
        verify(run)
        print(dict(prepared=len(prepared),score_jobs=len(jobs),fine_frames=fine_count,new_pairs=pairs,batches=batches,qa_calls=0),flush=True)
        return spec


def score_worker(run,gpu,questions):
    """接口：原细帧逐批四查询评分；不新增补查，不改变原问题分数或视觉特征。"""
    os.environ['CUDA_VISIBLE_DEVICES']=gpu;offline_environment()
    import torch
    from PIL import Image
    from videoqa_methods.model_ops import FeatureScorer
    torch.set_num_threads(2);torch.set_num_interop_threads(1);base.configure_visual_precision(torch)
    scorer=FeatureScorer(str(base.MODEL))
    try:
        syn=run/'synthetic'/(gpu+'.json')
        if syn.exists():assert read_json(syn)['status']=='completed'
        else:
            images=[Image.new('RGB',(1280,720),'gray') for _ in range(16)]
            durable(syn,dict(status='started',frames=16,utc=utc()))
            scorer.score_batch_features(scorer.tokenize('Which synthetic color is visible?'),images)
            for im in images:im.close()
            durable(syn,dict(status='completed',frames=16,utc=utc()))
        for q in questions:
            dest=run/'raw_scores'/(q+'.json')
            if dest.exists():
                assert read_json(dest)['status']=='completed' and read_json(dest)['protocol_sha256']==sha256(run/'protocol.json');continue
            item=read_json(run/'inputs'/(q+'.json'));assert item['enabled']
            pool=read_json(item['pool']);n=pool['initial_count'];f=np.load(item['features'],allow_pickle=False)
            values=[list(v) for v in read_json(item['coarse_scores'])['scores']]
            assert all(len(v)==n for v in values)
            encoded=[scorer.tokenize(t) for t in item['queries']];error=0.
            # 步骤2：保持原细帧批次和源图，先记账再前向，原视觉重放不符就停。
            for start in range(0,len(pool['new_assets']),16):
                assert not (run/'STOP.json').exists()
                assets=pool['new_assets'][start:start+16];images=[]
                for a in assets:
                    assert sha256(a['path'])==a['sha256']
                    with Image.open(a['path']) as im:images.append(im.convert('RGB'))
                hashes=[hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in images]
                for k in range(4):
                    scores,features=base.score_once(run,q,'option_'+str(k),n+start,scorer,encoded[k],images,gpu,hashes)
                    error=max(error,float(np.max(np.abs(features-f[n+start:n+start+len(images)]))))
                    assert error<=1e-6,(q,start,error);values[k].extend(scores)
                for im in images:im.close()
            assert all(len(v)==len(pool['candidates']) for v in values)
            durable(dest,dict(status='completed',question_id=q,scores=values,max_feature_error=error,
                scoring_scope='complete original pool',protocol_sha256=sha256(run/'protocol.json'),cached_from=None,utc=utc()))
    except BaseException as exc:
        durable(run/'STOP.json',dict(error=repr(exc),gpu=gpu,utc=utc()));raise


def select_one(job):
    """接口：原exp07空查询重放通过后才接受新选择；结构回退不读取空分数。"""
    run,q=Path(job[0]),job[1];spec=read_json(run/'protocol.json');dest=run/'questions'/(q+'.json')
    if dest.exists():
        assert read_json(dest)['protocol_sha256']==sha256(run/'protocol.json');return
    item=read_json(run/'inputs'/(q+'.json'));pool=read_json(item['pool']);features=np.load(item['features'],allow_pickle=False)
    reference=read_json(base.FULL/'traces'/(q+'.json'));old=read_json(base.FULL/'selections/RD-1.2-exp07'/(q+'.json'))['selected_indices']
    assert (option_structure(item['question'],item['queries'])=='other')==item['enabled']
    null,check=select_local_query_coverage(pool,features,spec['configuration'],reference['region_budgets'],np.zeros((4,len(features))))
    assert null==old and check['region_budgets']==reference['region_budgets']
    assert abs(check['objective']-reference['objective'])<=1e-6
    raw=read_json(run/'raw_scores'/(q+'.json'));assert raw['status']=='completed'
    selected,trace=select_local_query_coverage(pool,features,spec['configuration'],reference['region_budgets'],raw['scores'],enabled=item['enabled'])
    assert set(reference['protected_indices'])<=set(selected)
    if not item['enabled']:assert selected==old
    durable(dest,dict(question_id=q,protocol_sha256=sha256(run/'protocol.json'),selected_indices=selected,
        selected_frames=[pool['candidates'][i] for i in selected],old_indices=old,trace=trace,
        route=item['route'],enabled=item['enabled'],null_replay_equal=True,new_model_calls=0))


def finish(run,spec):
    """接口：评分预算及固定75事实闭合；不报告新准确率或触发问答。"""
    calls=list((run/'calls').glob('*/*/*.json'))
    assert len(calls)==spec['real_batches'] and all(read_json(p)['status']=='completed' for p in calls)
    assert sum(read_json(p)['identity']['frames'] for p in calls)==spec['new_frame_text_pairs']
    records={q:read_json(run/'questions'/(q+'.json')) for q in spec['question_ids']};facts=[]
    for fact in read_json(FACTS)['entries']:
        q=fact['question_id'];r=records[q];pool=read_json(spec['inputs'][q]['pool']);pts=set(fact['witness_pts'])
        facts.append(dict(question_id=q,fact_id=fact['fact_id'],before=bool(pts & {pool['candidates'][i]['source_pts'] for i in r['old_indices']}),
            after=bool(pts & {c['source_pts'] for c in r['selected_frames']})))
    assert len(facts)==75 and len({f['question_id'] for f in facts})==42
    result=dict(status='completed',question_count=200,qa_calls=0,facts=facts,before=sum(f['before'] for f in facts),after=sum(f['after'] for f in facts),
        gains=[f for f in facts if f['after'] and not f['before']],losses=[f for f in facts if f['before'] and not f['after']],
        changed=[q for q,r in records.items() if r['old_indices']!=r['selected_indices']],fallback=[q for q,r in records.items() if not r['enabled']],
        artifacts={str(p):sha256(p) for folder in ['questions','raw_scores','calls','synthetic'] for p in (run/folder).rglob('*') if p.is_file()},utc=utc())
    durable(run/'summary.json',result)
    print({k:v for k,v in result.items() if k not in ['facts','artifacts','changed','fallback']},flush=True)


def execute(run):
    """接口：GPU评分结束后释放模型，自动转CPU选择；不依赖交互会话，不自动问答。"""
    from videoqa_runtime.baseline_run import available_gpus
    with exclusive(run/'coordinator.lock'):
        spec=verify(run)
        if (run/'summary.json').exists():
            for p,h in read_json(run/'summary.json')['artifacts'].items():assert sha256(p)==h
            print('Completed recovery:0 model calls',flush=True);return
        assert not (run/'STOP.json').exists()
        active={g:qs for g,qs in spec['assignment'].items() if any(not (run/'raw_scores'/(q+'.json')).exists() for q in qs)}
        if active:
            while True:
                try:free,_=available_gpus(list(active))
                except RuntimeError:free=[]
                if set(active)<=set(free):break
                durable(run/'status.json',dict(status='waiting_resources',pid=os.getpid(),utc=utc()));time.sleep(15)
            with ProcessPoolExecutor(max_workers=len(active),mp_context=mp.get_context('spawn')) as ex:
                pending={ex.submit(score_worker,run,g,qs) for g,qs in active.items()}
                while pending:
                    done,pending=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                    for f in done:f.result()
                    durable(run/'status.json',dict(status='scoring',completed=len(list((run/'raw_scores').glob('*.json'))),pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        verify(run)
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as ex:
            futures=[ex.submit(select_one,(str(run),q)) for q in spec['question_ids']]
            for i,f in enumerate(as_completed(futures),1):
                f.result();durable(run/'status.json',dict(status='selecting',completed=i,total=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        finish(run,spec);durable(run/'status.json',dict(status='completed',completed=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=['prepare','run'],required=True)
    args=p.parse_args();import re
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id)
    run=ROOT/'outputs/analysis'/args.run_id
    if args.stage=='prepare':prepare(run)
    else:execute(run)
