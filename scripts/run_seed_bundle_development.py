#!/usr/bin/env python3
"""夜间固定双条件200题验证：复用离线帧组结果，独立后台执行两个调用账本。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_budget32.frames import FrameProvider
from videoqa_audit.query_input_probe import BASE,FULL
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
PYTHON='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'
CONDITIONS={'coverage_only':['RD-1.2-diag02','rd12_diag02__qwen32__dev200__20261002_r1'],
            'coverage_with_pairs':['RD-1.2-exp11','rd12_exp11__qwen32__dev200__20261002_r1']}


def export_one(job):
    """接口：实际解码每条件选帧，核验源身份并冻结RGB；不导入旧答案。"""
    run,source,q=map(Path,job);qid=str(q)
    spec=read_json(source/'protocol.json');path=source/'questions'/(qid+'.json');record=read_json(path)
    assert record['question_id']==qid and record['protocol_sha256']==sha256(source/'protocol.json')
    pp=Path(spec['inputs'][qid]['pool']);pool=read_json(pp)
    for condition,(method,_) in CONDITIONS.items():
        dest=run/'qa_inputs'/method/(qid+'.json')
        if dest.exists():
            saved=read_json(dest)
            for p,h in saved['sources'].items():assert sha256(p)==h
            continue
        value=record['conditions'][condition];selection=dict(video=pool['video'],selected_frames=value['selected_frames'])
        assert value['selected_frames']==[pool['candidates'][i] for i in value['selected_indices']]
        images,hashes=FrameProvider(selection).decode()
        for image in images:image.close()
        sources={str(p):sha256(p) for p in (run/'protocol.json',source/'protocol.json',path,pp)}
        durable(dest,dict(question_id=qid,method=method,selection=selection,rgb_sha256=hashes,
            selected_indices=value['selected_indices'],condition=condition,sources=sources,new_model_calls=0))
    return qid


def prepare(run,source,gpus):
    """接口：冻结400次总预算及两个200次账本，包含exp07和两基线比较来源。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'prepare.lock'):
        source=source.resolve();s=read_json(source/'protocol.json');summary=read_json(source/'summary.json')
        assert summary['status']=='completed' and s['budget']==32 and summary['new_model_calls']==0
        for p,h in s['sources'].items():assert sha256(p)==h,p
        ids=[r['question_id'] for r in read_json(DATA)['rows']];assert ids==s['question_ids'] and len(ids)==200
        assert gpus and len(set(gpus))==len(gpus) and set(gpus)<=set(map(str,range(8)))
        inputs={str(source/'protocol.json'):sha256(source/'protocol.json'),str(source/'summary.json'):sha256(source/'summary.json'),str(DATA):sha256(DATA)}
        # 步骤1：两个条件同上游同协议，各生成一次；分批运行如实记录，不声称同GPU/进程配对。
        for q in ids:
            for path in (source/'questions'/(q+'.json'),FULL/'imported/results/RD-1.2-exp07'/(q+'.json'),
                         BASE/'results/BASE-Uniform'/(q+'.json'),BASE/'results/BASE-BLIP-TopK'/(q+'.json')):
                inputs[str(path)]=sha256(path)
        for name in ('scripts/run_seed_bundle_development.py','scripts/analyze_seed_bundles.py','src/videoqa_methods/seed_bundle_selection.py'):
            path=ROOT/name;inputs[str(path)]=sha256(path);dest=run/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha256(dest)==inputs[str(path)]
            else:shutil.copy2(path,dest)
        protocol=dict(source_run=str(source),question_ids=ids,conditions=CONDITIONS,allowed_gpus=gpus,sources=inputs,
            max_total_qa_calls=400,max_calls_per_condition=200,real_blip_calls=0,new_refinement_calls=0,
            development_to_full_policy='Candidate must reach at least 140/200 and exceed coverage-only control before prioritizing full validation; exploratory resource gate, not significance or tuning target.')
        if (run/'protocol.json').exists():assert read_json(run/'protocol.json')==protocol
        else:durable(run/'protocol.json',protocol)
        # 步骤2：选帧RGB全部通过后才能交正式Qwen入口，恢复已完成导出只复核不重新答题。
        with ThreadPoolExecutor(max_workers=4) as executor:
            for n,_ in enumerate(executor.map(export_one,[(str(run),str(source),q) for q in ids]),1):
                durable(run/'status.json',dict(status='exporting',completed=n,total=200,utc=utc()))
        for condition,(method,_) in CONDITIONS.items():
            directory=run/'qa_inputs'/method;exports={q:sha256(directory/(q+'.json')) for q in ids}
            manifest=dict(status='completed',question_count=200,frame_budget=32,question_ids=ids,dataset_manifest=str(DATA),manifest_sha256=sha256(DATA),
                parent_run=str(BASE),parent_protocol_sha256=sha256(BASE/'protocol.json'),method=method,
                configuration='Frozen exp07 pool, protected anchors; '+condition+' unique-frame marginal greedy; original Qwen32 reference protocol',
                diagnostic=condition=='coverage_only',exports=exports)
            p=directory/'selection_manifest.json'
            if p.exists():assert read_json(p)==manifest
            else:durable(p,manifest)
        durable(run/'prepare_complete.json',dict(status='completed',question_count=200,condition_count=2,utc=utc()))
        return protocol


def execute(run):
    """接口：脱离会话顺序执行两个200题任务；任何异常停止，不自动追加生成。"""
    with exclusive(run/'run.lock'):
        spec=read_json(run/'protocol.json')
        for p,h in spec['sources'].items():assert sha256(p)==h,p
        entry=str(ROOT/'scripts/run_frozen_development.py');start=time.perf_counter()
        for condition,(method,name) in CONDITIONS.items():
            directory=run/'qa_inputs'/method
            subprocess.run([PYTHON,entry,'--run-id',name,'--stage','prepare','--input-dir',str(directory),'--gpus',*spec['allowed_gpus'],'--resume'],check=True)
            durable(run/'status.json',dict(status='qa',condition=condition,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            subprocess.run([PYTHON,entry,'--run-id',name,'--stage','qa','--resume'],check=True)
        durable(run/'status.json',dict(status='completed',seconds=time.perf_counter()-start,pid=os.getpid(),utc=utc()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=['prepare','run'],required=True)
    p.add_argument('--source-run',default='seed_bundles_dev200_20261002_r2');p.add_argument('--gpus',nargs='+',default=list(map(str,range(8))))
    a=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.source_run)
    run=ROOT/'outputs/methods'/a.run_id
    if a.stage=='prepare':prepare(run,ROOT/'outputs/analysis'/a.source_run,a.gpus)
    else:execute(run)
