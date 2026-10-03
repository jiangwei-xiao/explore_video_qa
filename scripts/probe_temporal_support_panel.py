"""固定12题时间支持离线对照，固定配额，不问答、不评分、不改变补查。"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE,FULL
from videoqa_methods.density_selection import source_time,frame_key
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_audit.temporal_support_probe import temporal_support,local_prefix

CURRENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
PANEL=ROOT/'configs/analysis/evidence_state_panel_20261003.json'


def main(run):
    """接口：先冻结单尺度协议，再输出12题选择及来源；不附用历史预测作为新成绩。"""
    sources={str(Path(__file__)):sha256(__file__)}
    module=ROOT/'src/videoqa_audit/temporal_support_probe.py';sources[str(module)]=sha256(module)
    def read(p):sources[str(p)]=sha256(p);return read_json(p)
    panel=read(PANEL)
    protocol=dict(panel_sha256=sha256(PANEL),tau_seconds=4,
        rule='max normalized original coarse ITM times exp(-absolute PTS distance/4) within frozen cell',
        fixed='exp12 quotas, original anchor, sqrt novelty, original full coarse/fine pool',
        all_zero_fallback='uniform support',new_model_calls=0,
        limitation='Single engineering hypothesis borrowing prior window scale; no semantic relation claim or parameter sweep')
    run.mkdir(parents=True,exist_ok=True)
    if (run/'protocol.json').exists():
        if read_json(run/'protocol.json')!=protocol:raise ValueError('Protocol changed')
    else:durable(run/'protocol.json',protocol)
    records=[]
    # 步骤1：使用冻结全部候选的真实PTS，不把新增细帧当成已有ITM评分。
    for case in panel['cases']:
        q=case['question_id'];pool=read(BASE/'pools'/(q+'.json'));old=read(FULL/'traces'/(q+'.json'))
        current=read(CURRENT/'traces'/(q+'.json'));fp=BASE/'features'/(q+'.npy');sources[str(fp)]=sha256(fp)
        f=np.load(fp,allow_pickle=False);c=pool['candidates'];keys=[frame_key(x) for x in c]
        plans={r['region_id']:r for r in pool['plan']['regions']}
        selected=[];regions=[];old_selected=[]
        for r in old['regions']:
            ids=sorted(r['members'],key=lambda i:keys[i]);times=[source_time(c[i],pool['video']) for i in ids]
            scores=[pool['initial_scores'][i] if i<pool['initial_count'] else None for i in ids]
            control,_,_=weighted_support(f,ids,plans[r['region_id']]['seed_indices'],pool['initial_scores'])
            restored,_=local_prefix(f,ids,r['prefix'][0],control,keys,min(32,len(ids)))
            if restored!=r['prefix']:raise ValueError('Original support control failed to reproduce prefix')
            support=temporal_support(times,scores,4.)
            prefix,steps=local_prefix(f,ids,r['prefix'][0],support,keys,min(32,len(ids)))
            quota=current['region_budgets'][str(r['region_id'])]
            selected.extend(prefix[:quota]);old_selected.extend(r['prefix'][:quota])
            regions.append(dict(region_id=r['region_id'],quota=quota,prefix=prefix,steps=steps,
                support_values=dict(zip(map(str,ids),map(float,support)))))
        # 步骤2：配额和预算闭合，选择按源时间恢复；保留替换明细供真实画面复核。
        selected.sort(key=lambda i:keys[i]);old_selected.sort(key=lambda i:keys[i])
        if len(selected)!=len(set(selected)) or len(selected)!=len(old_selected):raise ValueError('Budget changed')
        records.append(dict(question_id=q,regions=regions,selected_indices=selected,old_indices=old_selected,
            selection=dict(video=pool['video'],selected_frames=[c[i] for i in selected]),
            added=sorted(set(selected)-set(old_selected)),removed=sorted(set(old_selected)-set(selected))))
    result=dict(protocol=protocol,sources=sources,records=records,new_model_calls=0,new_answers=0)
    path=run/'selections.json'
    if path.exists():
        if read_json(path)!=result:raise ValueError('Resume input or output changed')
    else:durable(path,result)
    print(dict(questions=len(records),changed=sum(r['selected_indices']!=r['old_indices'] for r in records),new_model_calls=0),flush=True)
    for q,i in [('409-3',80),('556-1',42),('395-2',276),('383-2',944)]:
        r=next(r for r in records if r['question_id']==q)
        region=next(x for x in r['regions'] if str(i) in x['support_values'])
        print(q,i,dict(position=region['prefix'].index(i)+1 if i in region['prefix'] else None,quota=region['quota'],selected=i in r['selected_indices']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(Path(p.parse_args().run))
