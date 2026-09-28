"""exp04分阶段离线输入准备：局部补查不变，只放开现成粗候选的选择资格。"""
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
from videoqa_methods.weighted_seed_support import select_weighted_seed_support

SOURCE=ROOT/'outputs/full_videoqa/uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01'
EXP03=ROOT/'outputs/analysis/rd12_weighted_support_20260928_r1'
MANIFEST=ROOT/'data/manifests/video_mme_development_200_20260920.json'
PROXY=ROOT/'outputs/methods/followups_clocal_query_videomme50_20260915_r1/proxy_manifest.json'
METHOD='RD-1.2-exp04'
PILOT=['584-3','556-1','724-3','781-1','632-3','007-3','103-1','170-2']

def select_one(job):
    """接口：输入仅原池、特征和规则；不读取答案。只由原预算公式重算区域名额。"""
    output,q=job;pp=SOURCE/'pools'/(q+'.json');fp=SOURCE/'features'/(q+'.npy');cp=SOURCE/'selection_checkpoints'/(q+'.json')
    check=read_json(cp)
    for p in (pp,fp):assert sha256(p)==check['files'][str(p.relative_to(SOURCE))]
    paths=[pp,fp,cp,EXP03/'selections'/(q+'.json'),ROOT/'src/videoqa_methods/coarse_eligibility.py',ROOT/'src/videoqa_methods/weighted_seed_support.py',Path(__file__)]
    sources={str(p):sha256(p) for p in paths};dest=output/'selections'/(q+'.json')
    if dest.exists():
        rec=read_json(dest);assert rec['sources']==sources and sha256(output/rec['trace_file'])==rec['trace_sha256'];return rec
    old=read_json(pp);f=np.load(fp,allow_pickle=False);quota=old['selection_trace']['region_budgets']
    args=(old['candidates'],f,old['plan'],old['video'],dict(frame_budget=32),quota,old['initial_scores'])
    # 步骤1：重放exp03局部规则，原输入不变时的参照只在分析层保存。
    baseline,_=select_weighted_seed_support(*args)
    prior=read_json(EXP03/'selections'/(q+'.json'));assert baseline==prior['selected_indices']
    selected,trace=select_global_coarse_two_pass(*args)
    assert sum(trace['region_budgets'].values())==32
    selection=dict(video=old['video'],selected_frames=[old['candidates'][i] for i in selected]);validate_frames(selection)
    # 步骤2：新方法不生成任何细帧/视觉/ITM调用；只精确解码将要送入Qwen的现成帧。
    images,hashes=FrameProvider(selection).decode()
    for im in images:im.close()
    tp=output/'traces'/(q+'.json');durable(tp,trace)
    rec=dict(question_id=q,method=METHOD,configuration='M32/K32; global coarse eligibility in both passes; original local refinement; exp03 scoring',
        selection=selection,rgb_sha256=hashes,sources=sources,selected_indices=selected,exp03_indices=baseline,
        removed=sorted(set(baseline)-set(selected)),added=sorted(set(selected)-set(baseline)),
        selected_outside_original_windows=trace['selected_outside_original_windows'],
        trace_file=str(tp.relative_to(output)),trace_sha256=sha256(tp),new_model_calls=0)
    durable(dest,rec);return rec

def main(output,stage):
    """接口：先固定八题离线验证，再同规则扩展200题；只有all阶段导出问答清单。"""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with exclusive(output/'prepare.lock'):
        data=read_json(MANIFEST)['rows'];ids=[r['question_id'] for r in data]
        assert len(ids)==len(set(ids))==200
        requested=PILOT if stage=='pilot' else ids
        # 步骤1：同一规则、同一缓存身份；不把机制案例名单当算法条件。
        with ThreadPoolExecutor(max_workers=4) as ex:records=list(ex.map(select_one,[(output,q) for q in requested]))
        summary=dict(stage=stage,questions=len(records),new_model_calls=0,utc=utc(),
            selection_sha256={r['question_id']:sha256(output/'selections'/(r['question_id']+'.json')) for r in records},
            changed_questions=[r['question_id'] for r in records if r['removed']],
            outside_selected={r['question_id']:len(r['selected_outside_original_windows']) for r in records})
        if stage=='all':
            # 步骤2：固定旧事实只参与评价，不能进入选择器或按题修改规则。
            proxy=read_json(PROXY);facts=[];bank={r['question_id']:r for r in records}
            for fact in proxy['entries']:
                q=fact['question_id'];prev=read_json(EXP03/'selections'/(q+'.json'));now=bank[q]
                pts=set(fact['witness_pts']);a={f['source_pts'] for f in prev['selection']['selected_frames']};b={f['source_pts'] for f in now['selection']['selected_frames']}
                facts.append(dict(**fact,exp03_retained=bool(a&pts),exp04_retained=bool(b&pts)))
            assert len(facts)==75 and len({r['question_id'] for r in facts})==42
            durable(output/'proxy_facts.json',facts)
            summary['proxy_gains']=[f for f in facts if f['exp04_retained'] and not f['exp03_retained']]
            summary['proxy_losses']=[f for f in facts if f['exp03_retained'] and not f['exp04_retained']]
            byid={r['question_id']:r for r in rows()};inputs=output/'inputs';inputs.mkdir(exist_ok=True);exports={}
            for q in ids:
                rec=bank[q];path=output/'selections'/(q+'.json');item=dict(rec,video_id=byid[q]['video_id'],sources=dict(rec['sources'],**{str(path):sha256(path)}))
                dest=inputs/(q+'.json');durable(dest,item);exports[q]=sha256(dest)
            durable(inputs/'selection_manifest.json',dict(status='completed',method=METHOD,configuration=records[0]['configuration'],diagnostic=False,
                question_ids=ids,question_count=200,frame_budget=32,dataset_manifest=str(MANIFEST),manifest_sha256=sha256(MANIFEST),
                parent_run=str(SOURCE),parent_protocol_sha256=sha256(SOURCE/'protocol.json'),exports=exports,new_model_calls=0,utc=utc()))
        durable(output/(stage+'_summary.json'),summary)
        print('stage',stage,'questions',len(records),'changed',len(summary['changed_questions']),'outside',summary['outside_selected'],flush=True)
        if stage=='all':print('proxy gains',summary['proxy_gains'],'losses',summary['proxy_losses'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--stage',choices=['pilot','all'],required=True)
    args=parser.parse_args();main(args.output,args.stage)
