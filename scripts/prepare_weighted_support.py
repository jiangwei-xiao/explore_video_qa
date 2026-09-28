"""RD-1.2-exp03离线重放与输入冻结：只改变局部支持函数，模型调用0。"""
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
from videoqa_audit.local_distance_scale import select_local_distance
from videoqa_methods.weighted_seed_support import select_weighted_seed_support

SOURCE=ROOT/'outputs/full_videoqa/uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01'
MANIFEST=ROOT/'data/manifests/video_mme_development_200_20260920.json'
PROXY=ROOT/'outputs/methods/followups_clocal_query_videomme50_20260915_r1/proxy_manifest.json'
METHOD='RD-1.2-exp03'

def select_one(job):
    """接口：不接收答案；原M32窗口、候选、保护锚点和第一遍配额保持不变。"""
    output,q,export=job
    pp=SOURCE/'pools'/(q+'.json');fp=SOURCE/'features'/(q+'.npy');cp=SOURCE/'selection_checkpoints'/(q+'.json')
    check=read_json(cp)
    for p in (pp,fp):assert sha256(p)==check['files'][str(p.relative_to(SOURCE))]
    sources={str(p):sha256(p) for p in (pp,fp,cp,ROOT/'src/videoqa_methods/weighted_seed_support.py',Path(__file__))}
    dest=output/'selections'/(q+'.json')
    if dest.exists():
        rec=read_json(dest);assert rec['sources']==sources
        assert sha256(output/rec['trace_file'])==rec['trace_sha256']
        return rec
    pool=read_json(pp);f=np.load(fp,allow_pickle=False);quota=pool['selection_trace']['region_budgets']
    args=(pool['candidates'],f,pool['plan'],pool['video'],dict(frame_budget=32),quota)
    # 步骤1：重放原选择，确认加载的是正式严格FP32池；不修改上游或第一遍配额。
    assert select_local_distance(*args)[0]==pool['selected_indices']
    indices,trace=select_weighted_seed_support(*args,pool['initial_scores'])
    assert trace['region_budgets']==quota
    selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in indices]);validate_frames(selection)
    # 步骤2：先保存零问答选择与完整竞争轨迹；RGB准备只影响后续取帧身份。
    tp=output/'traces'/(q+'.json');durable(tp,trace)
    new=set(indices);old=set(pool['selected_indices'])
    rec=dict(question_id=q,method=METHOD,configuration='M32/K32; frozen RD-1.2 quotas; weighted seed support',
        selected_indices=indices,selection=selection,sources=sources,removed=sorted(old-new),added=sorted(new-old),
        original_selected_indices=pool['selected_indices'],trace_file=str(tp.relative_to(output)),trace_sha256=sha256(tp),
        source_pool_sha256=sha256(pp),new_model_calls=0)
    if export:
        images,hashes=FrameProvider(selection).decode()
        for im in images:im.close()
        rec['rgb_sha256']=hashes
    durable(dest,rec);return rec

def main(output):
    """接口：固定200题并附既有机制案例；事实仅用于评价，不进入选择函数。"""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with exclusive(output/'prepare.lock'):
        dev=read_json(MANIFEST)['rows'];ids=[r['question_id'] for r in dev]
        assert len(ids)==len(set(ids))==200
        extras=[q for q in ('007-3','193-2','004-1','325-3','326-2','619-3') if q not in ids]
        source_fingerprints={str(p):sha256(p) for p in (MANIFEST,PROXY,Path(__file__),ROOT/'src/videoqa_methods/weighted_seed_support.py')}
        completed=output/'offline_summary.json'
        if completed.exists():
            old=read_json(completed);assert old['sources']==source_fingerprints
            for q,h in old['selection_sha256'].items():assert sha256(output/'selections'/(q+'.json'))==h
            print('Offline already complete; model calls 0',flush=True);return
        with ThreadPoolExecutor(max_workers=4) as ex:
            records=list(ex.map(select_one,[(output,q,q in ids) for q in ids+extras]))
        bank={r['question_id']:r for r in records};facts=[];proxy=read_json(PROXY)
        assert proxy['questions']==42 and proxy['facts']==75 and len(proxy['entries'])==75
        # 步骤3：既定75事实只做已列见证代理，不把未标注的新帧当无关。
        for fact in proxy['entries']:
            q=fact['question_id'];r=bank[q];pool=read_json(SOURCE/'pools'/(q+'.json'))
            old={pool['candidates'][i]['source_pts'] for i in r['original_selected_indices']}
            new={f['source_pts'] for f in r['selection']['selected_frames']}
            pts=set(fact['witness_pts'])
            facts.append(dict(**fact,old_retained=bool(old&pts),new_retained=bool(new&pts)))
        perq=[]
        for q in sorted({f['question_id'] for f in facts}):
            fs=[f for f in facts if f['question_id']==q]
            perq.append(dict(question_id=q,n=len(fs),old=sum(f['old_retained'] for f in fs)/len(fs),new=sum(f['new_retained'] for f in fs)/len(fs)))
        durable(output/'proxy_facts.json',facts)
        inputs=output/'inputs';inputs.mkdir(exist_ok=True)
        byid={r['question_id']:r for r in rows()};exports={}
        # 步骤4：导出正式200输入，附加机制案例不进入调用名单。
        for q in ids:
            rec=bank[q];path=output/'selections'/(q+'.json')
            item=dict(rec,video_id=byid[q]['video_id'],sources=dict(rec['sources'],**{str(path):sha256(path)}))
            dest=inputs/(q+'.json');durable(dest,item);exports[q]=sha256(dest)
        durable(inputs/'selection_manifest.json',dict(status='completed',method=METHOD,configuration=records[0]['configuration'],
            diagnostic=False,question_ids=ids,question_count=200,frame_budget=32,dataset_manifest=str(MANIFEST),
            manifest_sha256=sha256(MANIFEST),parent_run=str(SOURCE),parent_protocol_sha256=sha256(SOURCE/'protocol.json'),
            exports=exports,new_model_calls=0,utc=utc()))
        summary=dict(status='completed',new_model_calls=0,questions=200,extra_offline_questions=extras,sources=source_fingerprints,
            changed_questions=[q for q in ids if bank[q]['removed']],proxy_denominator=dict(questions=42,facts=75),
            proxy_old_macro=float(np.mean([r['old'] for r in perq])),proxy_new_macro=float(np.mean([r['new'] for r in perq])),
            proxy_improvements=[f for f in facts if f['new_retained'] and not f['old_retained']],
            proxy_losses=[f for f in facts if f['old_retained'] and not f['new_retained']],
            per_question_proxy=perq,selection_sha256={q:sha256(output/'selections'/(q+'.json')) for q in ids+extras},utc=utc())
        durable(completed,summary)
        print({k:v for k,v in summary.items() if k in ('questions','extra_offline_questions','proxy_old_macro','proxy_new_macro')},flush=True)
        print('changed',len(summary['changed_questions']),'proxy gains',len(summary['proxy_improvements']),'proxy losses',len(summary['proxy_losses']),flush=True)
        for q in extras+['103-1','395-2','383-2']:
            r=bank[q];print(q,'removed',r['removed'],'added',r['added'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)
