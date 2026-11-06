#!/usr/bin/env python3
"""接口：从冻结输出生成成本图、历史口径核对和缓存保护证据，不调用模型。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')


def main():
    """接口：已有结果只读，新增复核视图可重算；旧计分/源帧/模型返回保持不变。"""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from videoqa_runtime.common import read_json,sha256
    from videoqa_full.state import durable
    from videoqa_access.trial import METHODS,freeze
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root).resolve()
    assert read_json(root/'independent_completion_audit.json')['status']=='passed'
    report=read_json(root/'reports/videomme.json');data=[r for r in read_json(root/'manifest.json')['rows'] if r['dataset']=='videomme']
    # 步骤1核验全部继承缓存的字节/mtime，证明恢复未重发或覆盖125个已完成批次。
    recovery=read_json(root/'frontend_recovery.json');source=Path(recovery['source_run']);cache=[]
    for rel,h in recovery['imported_cache_sha256'].items():
        current,old=root/rel,source/rel
        if sha256(current)!=h or sha256(old)!=h or current.stat().st_mtime_ns!=old.stat().st_mtime_ns:
            raise ValueError('Imported completed batch changed')
        cache.append(dict(path=rel,sha256=h,mtime_ns=current.stat().st_mtime_ns))
    freeze(root/'final_cache_lineage_audit.json',dict(status='passed',completed_imported_batches=125,
        unchanged_cache_files=len(cache),files=cache,additional_environment_synthetic_blip_forwards=2,new_qa_calls=0))
    # 步骤2按原60题精确引用历史，不混作新协议答案；均匀源/像素/提示相同须复现。
    oldroot=ROOT/'outputs/full_videoqa/llava8__five_methods__videomme2700__20261005_r1'
    historic={};sources={};quality=[];lengths={};tasks={}
    for method in ('BASE-Uniform','BASE-BLIP-TopK','RD-1.2-exp07','RD-1.2-exp12'):
        values=[]
        for row in data:
            q=row['question_id'];path=oldroot/'results'/method/(q+'.json');old=read_json(path);sources[str(path)]=sha256(path)
            entry=dict(question_id=q,historical_correct=old['correct'],historical_prediction=old['answer']['parsed_answer'])
            if method in METHODS:
                new=read_json(root/'qa/videomme/results'/method/(q+'.json'));entry.update(new_correct=new['correct'],new_prediction=new['prediction'])
                if method=='BASE-Uniform':
                    obs=old['answer']['observed'];identity=new['input_identity']
                    same=(old['rgb_sha256']==identity['rgb_sha256'] and obs['pixel_sha256']==identity['pixel_sha256'] and obs['input_token_ids']==identity['input_token_ids'])
                    if not same or old['answer']['parsed_answer']!=new['prediction']:raise ValueError('Uniform historical repeatability differs')
                    entry['complete_input_and_prediction_identical']=True
            values.append(entry)
        historic[method]=dict(n=60,correct=sum(v['historical_correct'] for v in values),records=values)
    for row in data:
        q=row['question_id'];full=read_json(root/'qa/videomme/results/REF-WFS-SB'/(q+'.json'));low=read_json(root/'qa/videomme/results/CTRL-WFS-preview'/(q+'.json'))
        if full['input_identity']['source']!=low['input_identity']['source']:raise ValueError('Quality pair has different sources')
        if full['prediction']!=low['prediction']:
            quality.append(dict(question_id=q,full_prediction=full['prediction'],preview_prediction=low['prediction'],
                full_correct=full['correct'],preview_correct=low['correct'],previously_flagged_ambiguity=q in ('491-2','321-2')))
    for field,dest in [('stratum',lengths),('task_type',tasks)]:
        for group in dict.fromkeys(r[field] for r in data):
            ids={r['question_id'] for r in data if r[field]==group}
            dest[group]={m:dict(n=len(ids),correct=sum(r['correct'] for r in report['records'] if r['method']==m and r['question_id'] in ids)) for m in METHODS}
    freeze(root/'reports/supplemental_checks.json',dict(historical_clear_frontend=historic,historical_sources=sources,
        wfs_same_source_quality_prediction_changes=quality,duration_groups=lengths,task_groups=tasks,
        limitation='Observed first60 panel; original clear-frontend parent also tied Uniform. No full-dataset generalization claim.'))
    # 步骤3只导出实际已测工作点，不插值或虚构预算自适应/独立E2E曲线。
    out=root/'reports/final';out.mkdir(parents=True,exist_ok=True)
    fields=['method','n','correct','accuracy_percent','service_mib','decoded_frames','blip_seconds','selection_seconds','limited_component_seconds','parse_failures']
    csvpath=out/'accuracy_cost.csv'
    with csvpath.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for m in METHODS:
            v=report['metrics'][m];rows=[r for r in report['records'] if r['method']==m]
            writer.writerow(dict(method=m,n=60,correct=v['correct'],accuracy_percent=v['accuracy'],service_mib=v['mean_raw_bytes']/1024**2,
                decoded_frames=v['means']['decoded_frames'],blip_seconds=v['means']['common_blip_seconds']+v['means']['extra_blip_seconds'],selection_seconds=v['means']['selection_seconds'],
                limited_component_seconds=v['components']['limited']['mean'],parse_failures=sum(r['prediction'] is None for r in rows)))
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'access-pilot-v2','font.size':9})
    fig,(ax,costax)=plt.subplots(1,2,figsize=(12.5,4.5),gridspec_kw={'width_ratios':[1,1.35]})
    y=np.arange(len(METHODS));labels=['Uniform','Top-K','WFS (original frames)','WFS (preview only)','Top-K + local','RD-exp12']
    accuracy=np.array([report['metrics'][m]['accuracy'] for m in METHODS]);ci=np.array([report['metrics'][m]['wilson95_percent'] for m in METHODS])
    ax.errorbar(accuracy,y,xerr=np.stack([accuracy-ci[:,0],ci[:,1]-accuracy]),fmt='o',color='#245c91',capsize=3)
    ax.set_yticks(y,labels);ax.set_ylim(len(METHODS)-.4,-.6);ax.set_xlim(30,75);ax.set_xlabel('Accuracy (%) with Wilson 95% interval');ax.grid(axis='x',alpha=.2)
    for x,yy,m in zip(accuracy,y,METHODS):ax.annotate(str(report['metrics'][m]['correct'])+'/60',(x,yy),xytext=(5,-11),textcoords='offset points',fontsize=8)
    parts=[]
    for m in METHODS:
        rows=[r for r in report['records'] if r['method']==m]
        values=[np.mean([r['common_blip_seconds']+r['preview_read_and_check_seconds'] for r in rows]),
                np.mean([r['extra_blip_seconds']+r['selection_seconds'] for r in rows]),
                np.mean([r['raw_access_compute_seconds']+r['network_replay_seconds']['limited'] for r in rows]),
                np.mean([r['qa_generation_seconds']+r['qa_preprocess_seconds'] for r in rows])]
        if abs(sum(values)-report['metrics'][m]['components']['limited']['mean'])>1e-4:raise ValueError('Cost component sum differs')
        parts.append(values)
    parts=np.asarray(parts);left=np.zeros(len(METHODS))
    for j,(label,color) in enumerate(zip(['Shared preview/BLIP','Extra features/selection','Source access + link','QA'],['#527ca6','#95b3d0','#d7954d','#71936b'])):
        costax.barh(y,parts[:,j],left=left,label=label,color=color);left+=parts[:,j]
    costax.set_yticks(y,labels);costax.invert_yaxis();costax.set_xlabel('Mean measured-component estimate (seconds/query)');costax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,fontsize=7,frameon=False);costax.grid(axis='x',alpha=.2)
    fig.suptitle('Video-MME observed pilot: LLaVA-Video K8, BLIP-1, n=60')
    fig.text(.5,.015,'20 Mbps + 50 ms simulated link; shared preview available; excludes index creation/model loading; not direct remote E2E.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(out/'accuracy_cost.svg',metadata={'Date':None});fig.savefig(out/'accuracy_cost.png',dpi=160);plt.close(fig)
    # 步骤4单列可复算的报告代码/来源，旧运行代码快照和旧结果保持原字节。
    snap=root/'report_code/final'/Path(__file__).name;snap.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,snap)
    freeze(root/'report_artifacts_final.json',dict(code_sha256=sha256(Path(__file__)),source_report_sha256=sha256(root/'reports/videomme.json'),
        independent_audit_sha256=sha256(root/'independent_completion_audit.json'),
        exports={str(p.relative_to(root)):sha256(p) for p in [root/'reports/supplemental_checks.json',csvpath,out/'accuracy_cost.svg',out/'accuracy_cost.png',root/'final_cache_lineage_audit.json',snap]},new_model_calls=0))
    print(dict(status='passed',uniform_repeatable=60,quality_changed=len(quality),quality_correctness_changed=sum(r['full_correct']!=r['preview_correct'] for r in quality),outputs=str(out)),flush=True)


if __name__=='__main__':main()
