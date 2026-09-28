"""exp05离线准备：只统一预算支持，候选域和最终排序严格复用exp04。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,rows,exclusive,utc
from videoqa_budget32.frames import FrameProvider,validate_frames
from videoqa_methods.coarse_eligibility import select_global_coarse_two_pass
from videoqa_methods.consistent_support_budget import select_consistent_support

SOURCE=ROOT/'outputs/full_videoqa/uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01'
REFERENCE=ROOT/'outputs/analysis/rd12_global_coarse_20260928_r3'
MANIFEST=ROOT/'data/manifests/video_mme_development_200_20260920.json'
PROXY=ROOT/'outputs/methods/followups_clocal_query_videomme50_20260915_r1/proxy_manifest.json'
METHOD='RD-1.2-exp05'
PILOT=['260-2','204-3','395-2','591-3','584-3','326-2','007-3','103-1','170-2']

def select_one(job):
    """接口：不读取标准答案；同配额必须同输入，变化只能由新预算传到旧局部选择。"""
    output,q=job;pp=SOURCE/'pools'/(q+'.json');fp=SOURCE/'features'/(q+'.npy');cp=SOURCE/'selection_checkpoints'/(q+'.json')
    rp=REFERENCE/'selections'/(q+'.json');ref=read_json(rp);rt=REFERENCE/ref['trace_file'];baseline_trace=read_json(rt)
    check=read_json(cp)
    for p in (pp,fp):assert sha256(p)==check['files'][str(p.relative_to(SOURCE))]
    paths=[pp,fp,cp,rp,rt,ROOT/'src/videoqa_methods/coarse_eligibility.py',ROOT/'src/videoqa_methods/weighted_seed_support.py',ROOT/'src/videoqa_methods/consistent_support_budget.py',Path(__file__)]
    sources={str(p):sha256(p) for p in paths};dest=output/'selections'/(q+'.json')
    if dest.exists():
        rec=read_json(dest);assert rec['sources']==sources and sha256(output/rec['trace_file'])==rec['trace_sha256'];return rec
    pool=read_json(pp);f=np.load(fp,allow_pickle=False)
    # 步骤1：重放父版本，先排除候选、精度或旧规则错配。
    parent_args=(pool['candidates'],f,pool['plan'],pool['video'],dict(frame_budget=32),pool['selection_trace']['region_budgets'],pool['initial_scores'])
    assert select_global_coarse_two_pass(*parent_args)[0]==ref['selected_indices']
    args=parent_args[:5]+(baseline_trace['region_budgets'],pool['initial_scores'])
    selected,trace=select_consistent_support(*args)
    same_quota=trace['region_budgets']==baseline_trace['region_budgets']
    assert same_quota==(selected==ref['selected_indices'])
    assert trace['eligible_indices']==baseline_trace['eligible_indices']
    assert trace['protected_indices']==baseline_trace['protected_indices']
    # 步骤2：两遍支持值逐候选相同；不改变新旧帧来源或加入新的视觉计算。
    for rid,values in trace['supports'].items():assert values['values']==trace['budget_trace']['supports'][rid]
    selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in selected]);validate_frames(selection)
    images,hashes=FrameProvider(selection).decode()
    for im in images:im.close()
    tp=output/'traces'/(q+'.json');durable(tp,trace)
    rec=dict(question_id=q,method=METHOD,configuration='M32/K32; exp04 eligible pool; shared weighted support across both passes',
        selection=selection,rgb_sha256=hashes,sources=sources,selected_indices=selected,exp04_indices=ref['selected_indices'],
        removed=sorted(set(ref['selected_indices'])-set(selected)),added=sorted(set(selected)-set(ref['selected_indices'])),
        quota_changes=trace['quota_changes'],trace_file=str(tp.relative_to(output)),trace_sha256=sha256(tp),new_model_calls=0)
    durable(dest,rec);return rec

def main(output,stage):
    """接口：固定九题先行后复用同规则完整200题，不按结果改权重或名单。"""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with exclusive(output/'prepare.lock'):
        data=read_json(MANIFEST)['rows'];ids=[r['question_id'] for r in data];assert len(ids)==len(set(ids))==200
        requested=PILOT if stage=='pilot' else ids
        with ThreadPoolExecutor(max_workers=4) as ex:records=list(ex.map(select_one,[(output,q) for q in requested]))
        summary=dict(stage=stage,questions=len(records),new_model_calls=0,utc=utc(),
            selection_sha256={r['question_id']:sha256(output/'selections'/(r['question_id']+'.json')) for r in records},
            changed_questions=[r['question_id'] for r in records if r['removed']],quota_changes={r['question_id']:r['quota_changes'] for r in records})
        if stage=='all':
            # 步骤3：代理事实只用于事后评价；不按标注或答案改变选择。
            proxy=read_json(PROXY);facts=[];bank={r['question_id']:r for r in records}
            for fact in proxy['entries']:
                q=fact['question_id'];prev=read_json(REFERENCE/'selections'/(q+'.json'));now=bank[q];pts=set(fact['witness_pts'])
                a={f['source_pts'] for f in prev['selection']['selected_frames']};b={f['source_pts'] for f in now['selection']['selected_frames']}
                facts.append(dict(**fact,exp04_retained=bool(a&pts),exp05_retained=bool(b&pts)))
            assert len(facts)==75 and len({r['question_id'] for r in facts})==42
            durable(output/'proxy_facts.json',facts)
            summary['proxy_gains']=[f for f in facts if f['exp05_retained'] and not f['exp04_retained']]
            summary['proxy_losses']=[f for f in facts if f['exp04_retained'] and not f['exp05_retained']]
            full={r['question_id']:r for r in rows()};inputs=output/'inputs';inputs.mkdir(exist_ok=True);exports={}
            for q in ids:
                p=output/'selections'/(q+'.json');rec=bank[q];item=dict(rec,video_id=full[q]['video_id'],sources=dict(rec['sources'],**{str(p):sha256(p)}))
                dest=inputs/(q+'.json');durable(dest,item);exports[q]=sha256(dest)
            durable(inputs/'selection_manifest.json',dict(status='completed',method=METHOD,configuration=records[0]['configuration'],diagnostic=False,
                question_ids=ids,question_count=200,frame_budget=32,dataset_manifest=str(MANIFEST),manifest_sha256=sha256(MANIFEST),
                parent_run=str(SOURCE),parent_protocol_sha256=sha256(SOURCE/'protocol.json'),exports=exports,new_model_calls=0,utc=utc()))
        durable(output/(stage+'_summary.json'),summary)
        print('stage',stage,'questions',len(records),'changed',len(summary['changed_questions']),flush=True)
        if stage=='pilot':
            for r in records:print(r['question_id'],'quota',r['quota_changes'],'removed',r['removed'],'added',r['added'],flush=True)
        else:print('proxy gains',summary['proxy_gains'],'losses',summary['proxy_losses'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--stage',choices=['pilot','all'],required=True)
    args=parser.parse_args();main(args.output,args.stage)
