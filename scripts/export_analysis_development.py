#!/usr/bin/env python3
"""将单条件冻结离线选择交给既有200题Qwen执行器；不修改公共推理层。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_budget32.frames import FrameProvider
from videoqa_audit.query_input_probe import BASE,FULL
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
PYTHON='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'


def export_one(job):
    """接口：复核离线选择对应源帧，再冻结实际RGB；完成项恢复只校验。"""
    run,q=Path(job[0]),job[1];spec=read_json(run/'protocol.json');source=Path(spec['source_run'])
    record_path=source/'questions'/(q+'.json');record=read_json(record_path);dest=run/'qa_inputs'/(q+'.json')
    if dest.exists():
        saved=read_json(dest)
        for p,h in saved['sources'].items():assert sha256(p)==h
        return
    protocol=read_json(source/'protocol.json');assert record['protocol_sha256']==sha256(source/'protocol.json')
    pp=Path(protocol['inputs'][q]['pool']);pool=read_json(pp)
    assert record['selected_frames']==[pool['candidates'][i] for i in record['selected_indices']]
    selection=dict(video=pool['video'],selected_frames=record['selected_frames']);images,hashes=FrameProvider(selection).decode()
    for im in images:im.close()
    sources={str(p):sha256(p) for p in (run/'protocol.json',source/'protocol.json',record_path,pp)}
    durable(dest,dict(question_id=q,method=spec['method'],selection=selection,selected_indices=record['selected_indices'],rgb_sha256=hashes,sources=sources))


def prepare(run,source,method,gpus):
    """接口：只接受冻结200题单条件选择，固定新问答上限和全部历史比较来源。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'prepare.lock'):
        s=read_json(source/'protocol.json');assert read_json(source/'summary.json')['status']=='completed'
        for p,h in s['sources'].items():assert sha256(p)==h,p
        assert method in {r['id'] for r in read_json(ROOT/'configs/method_versions.json')['entries']}
        ids=[r['question_id'] for r in read_json(DATA)['rows']];assert len(ids)==len(set(ids))==200 and ids==s['question_ids']
        assert gpus and len(gpus)==len(set(gpus)) and set(gpus)<=set(map(str,range(8)))
        assert shutil.disk_usage(run).free>2*1024**3
        sources={str(source/'protocol.json'):sha256(source/'protocol.json'),str(source/'summary.json'):sha256(source/'summary.json'),str(DATA):sha256(DATA),str(Path(__file__).resolve()):sha256(__file__)}
        # 步骤1：固定比较集合，不能按新输出挑选更有利的旧结果。
        for q in ids:
            for p in (source/'questions'/(q+'.json'),FULL/'imported/results/RD-1.2-exp07'/(q+'.json'),BASE/'results/BASE-Uniform'/(q+'.json'),BASE/'results/BASE-BLIP-TopK'/(q+'.json')):
                sources[str(p)]=sha256(p)
        spec=dict(method=method,source_run=str(source),question_ids=ids,sources=sources,allowed_gpus=gpus,max_qa_calls=200,
            full_priority_gate='At least 140/200 versus fixed exp07 136, plus input/mechanism review; not statistical significance or a parameter-search target')
        if (run/'protocol.json').exists():assert read_json(run/'protocol.json')==spec
        else:durable(run/'protocol.json',spec)
        snapshot=run/'code_snapshot'/Path(__file__).name;snapshot.parent.mkdir(exist_ok=True)
        if snapshot.exists():assert sha256(snapshot)==sha256(__file__)
        else:shutil.copy2(__file__,snapshot)
        # 步骤2：有界并行精确解码；完成全部输入后才允许加载Qwen。
        with ThreadPoolExecutor(max_workers=4) as executor:
            for n,_ in enumerate(executor.map(export_one,[(str(run),q) for q in ids]),1):
                durable(run/'status.json',dict(status='exporting',completed=n,total=200,pid=os.getpid(),utc=utc()))
        directory=run/'qa_inputs';manifest=dict(status='completed',question_count=200,frame_budget=32,question_ids=ids,
            dataset_manifest=str(DATA),manifest_sha256=sha256(DATA),parent_run=str(BASE),parent_protocol_sha256=sha256(BASE/'protocol.json'),
            method=method,configuration=s['hypothesis']+'; frozen Qwen32 reference protocol',
            diagnostic=False,exports={q:sha256(directory/(q+'.json')) for q in ids})
        p=directory/'selection_manifest.json'
        if p.exists():assert read_json(p)==manifest
        else:durable(p,manifest)
        durable(run/'prepare_complete.json',dict(status='completed',utc=utc()))


def execute(run,source,method,gpus):
    """接口：独立会话完成输入准备与200次Qwen；异常不自动重试或改配置。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'run.lock'):
        prepare(run,source,method,gpus);entry=str(ROOT/'scripts/run_frozen_development.py')
        subprocess.run([PYTHON,entry,'--run-id',run.name,'--stage','prepare','--input-dir',str(run/'qa_inputs'),'--gpus',*gpus,'--resume'],check=True)
        durable(run/'status.json',dict(status='qa',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        subprocess.run([PYTHON,entry,'--run-id',run.name,'--stage','qa','--resume'],check=True)
        durable(run/'status.json',dict(status='completed',pid=os.getpid(),utc=utc()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--source-run',required=True);p.add_argument('--method',required=True)
    p.add_argument('--gpus',nargs='+',default=list(map(str,range(8))));a=p.parse_args()
    assert all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',x) for x in (a.run_id,a.source_run,a.method))
    execute(ROOT/'outputs/methods'/a.run_id,ROOT/'outputs/analysis'/a.source_run,a.method,a.gpus)
