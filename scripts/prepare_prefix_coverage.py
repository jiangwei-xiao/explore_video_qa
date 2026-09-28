"""exp07前缀覆盖预算的离线验证与输入冻结，不改变实际局部选择规则。"""
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
from videoqa_methods.coarse_eligibility import select_global_coarse
from videoqa_methods.prefix_coverage_budget import select_prefix_coverage

SOURCE=ROOT/'outputs/full_videoqa/uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01'
REFERENCE=ROOT/'outputs/analysis/rd12_global_coarse_20260928_r3'
MANIFEST=ROOT/'data/manifests/video_mme_development_200_20260920.json'
PROXY=ROOT/'outputs/methods/followups_clocal_query_videomme50_20260915_r1/proxy_manifest.json'
METHOD='RD-1.2-exp07'
PILOT=['395-2','204-3','260-2','384-3','411-3','692-1','046-3','591-3','584-3','326-2','103-1','170-2']

def select_one(job):
    """接口：只优化预先固定局部前缀的预算，使用原始粗观测而非时间区间权重。"""
    output,q=job;pp=SOURCE/'pools'/(q+'.json');fp=SOURCE/'features'/(q+'.npy');cp=SOURCE/'selection_checkpoints'/(q+'.json')
    rp=REFERENCE/'selections'/(q+'.json');ref=read_json(rp);rt=REFERENCE/ref['trace_file'];parent=read_json(rt)
    check=read_json(cp)
    for p in (pp,fp):assert sha256(p)==check['files'][str(p.relative_to(SOURCE))]
    paths=[pp,fp,cp,rp,rt,ROOT/'src/videoqa_methods/coarse_eligibility.py',ROOT/'src/videoqa_methods/weighted_seed_support.py',ROOT/'src/videoqa_methods/prefix_coverage_budget.py',Path(__file__)]
    sources={str(p):sha256(p) for p in paths};dest=output/'selections'/(q+'.json')
    if dest.exists():
        saved=read_json(dest);assert saved['sources']==sources and sha256(output/saved['trace_file'])==saved['trace_sha256'];return saved
    pool=read_json(pp);features=np.load(fp,allow_pickle=False)
    args=(pool['candidates'],features,pool['plan'],pool['video'],dict(frame_budget=32),parent['region_budgets'],pool['initial_scores'])
    # 步骤1：旧配额应精确重构父版本输入，确认实际前缀没有改变局部评分。
    ids,trace=select_prefix_coverage(*args)
    assert trace['reference_prefix_selection']==ref['selected_indices']
    assert trace['eligible_indices']==parent['eligible_indices'] and trace['protected_indices']==parent['protected_indices']
    assert select_global_coarse(*(args[:5]+(trace['region_budgets'],pool['initial_scores'])))[0]==ids
    same=trace['region_budgets']==parent['region_budgets'];assert same==(ids==ref['selected_indices'])
    # 步骤2：只取已有精确源帧，候选/特征/ITM/补查均不新增。
    selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in ids]);validate_frames(selection)
    images,hashes=FrameProvider(selection).decode()
    for im in images:im.close()
    tp=output/'traces'/(q+'.json');durable(tp,trace)
    result=dict(question_id=q,method=METHOD,configuration='M32/K32; exp04 pool and local prefixes; exact density-weighted empirical coverage budget DP',
        selection=selection,rgb_sha256=hashes,sources=sources,selected_indices=ids,exp04_indices=ref['selected_indices'],
        removed=sorted(set(ref['selected_indices'])-set(ids)),added=sorted(set(ids)-set(ref['selected_indices'])),
        quotas=trace['region_budgets'],reference_quotas=parent['region_budgets'],
        trace_file=str(tp.relative_to(output)),trace_sha256=sha256(tp),new_model_calls=0)
    durable(dest,result);return result

def main(output,stage):
    """接口：先固定12题，再按同一无新超参规则处理200题；评价事实不进入DP。"""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with exclusive(output/'prepare.lock'):
        data=read_json(MANIFEST)['rows'];ids=[r['question_id'] for r in data];assert len(ids)==len(set(ids))==200
        requested=PILOT if stage=='pilot' else ids
        with ThreadPoolExecutor(max_workers=4) as ex:records=list(ex.map(select_one,[(output,q) for q in requested]))
        summary=dict(stage=stage,questions=len(records),new_model_calls=0,utc=utc(),
            selection_sha256={r['question_id']:sha256(output/'selections'/(r['question_id']+'.json')) for r in records},
            changed_questions=[r['question_id'] for r in records if r['removed']])
        if stage=='all':
            # 步骤3：固定75事实仅用于代理检查；DP目标提高是构造性质，不能当作证据改善。
            facts=[];bank={r['question_id']:r for r in records}
            for fact in read_json(PROXY)['entries']:
                q=fact['question_id'];a=read_json(REFERENCE/'selections'/(q+'.json'));b=bank[q];pts=set(fact['witness_pts'])
                facts.append(dict(**fact,exp04_retained=bool(pts&{f['source_pts'] for f in a['selection']['selected_frames']}),exp07_retained=bool(pts&{f['source_pts'] for f in b['selection']['selected_frames']})))
            assert len(facts)==75 and len({f['question_id'] for f in facts})==42
            durable(output/'proxy_facts.json',facts)
            summary['proxy_gains']=[f for f in facts if f['exp07_retained'] and not f['exp04_retained']]
            summary['proxy_losses']=[f for f in facts if f['exp04_retained'] and not f['exp07_retained']]
            inputs=output/'inputs';inputs.mkdir(exist_ok=True);full={r['question_id']:r for r in rows()};exports={}
            for q in ids:
                p=output/'selections'/(q+'.json');r=bank[q];dest=inputs/(q+'.json')
                durable(dest,dict(r,video_id=full[q]['video_id'],sources=dict(r['sources'],**{str(p):sha256(p)})));exports[q]=sha256(dest)
            durable(inputs/'selection_manifest.json',dict(status='completed',method=METHOD,configuration=records[0]['configuration'],diagnostic=False,
                question_ids=ids,question_count=200,frame_budget=32,dataset_manifest=str(MANIFEST),manifest_sha256=sha256(MANIFEST),
                parent_run=str(SOURCE),parent_protocol_sha256=sha256(SOURCE/'protocol.json'),exports=exports,new_model_calls=0,utc=utc()))
        durable(output/(stage+'_summary.json'),summary)
        print('stage',stage,'questions',len(records),'changed',len(summary['changed_questions']),flush=True)
        if stage=='pilot':
            for r in records:print(r['question_id'],'quotas',r['reference_quotas'],'->',r['quotas'],'removed',r['removed'],'added',r['added'],flush=True)
        else:print('proxy gains',summary['proxy_gains'],'losses',summary['proxy_losses'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--stage',choices=['pilot','all'],required=True)
    args=parser.parse_args();main(args.output,args.stage)
