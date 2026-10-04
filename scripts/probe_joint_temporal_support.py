"""固定配额下保留外观支持并补充旧时间支持，复用原衰减，不调整半径。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE,FULL
from videoqa_audit.temporal_support_probe import temporal_support,local_prefix
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.density_selection import source_time,frame_key
from probe_direct_support_floor import TARGETS,PARENT


def main(run):
    """接口：13题原完整粗细池；S=max(L,T)，仅重排局部次序，不重分预算。"""
    sources={str(p):sha256(p) for p in [Path(__file__),ROOT/'scripts/probe_direct_support_floor.py',
        ROOT/'src/videoqa_audit/temporal_support_probe.py',ROOT/'src/videoqa_methods/weighted_seed_support.py']}
    records=[]
    # 步骤1：控制重放原exp12前缀及最终选择；使用真实PTS，不用平均FPS。
    for q,targets in TARGETS.items():
        paths=[BASE/'pools'/(q+'.json'),BASE/'features'/(q+'.npy'),FULL/'traces'/(q+'.json'),
               PARENT/'traces'/(q+'.json'),PARENT/'selections/RD-1.2-exp12'/(q+'.json')]
        for p in paths:sources[str(p)]=sha256(p)
        pool=read_json(paths[0]);f=np.load(paths[1],allow_pickle=False)
        old=read_json(paths[2]);current=read_json(paths[3]);actual=read_json(paths[4])
        keys=[frame_key(c) for c in pool['candidates']]
        plans={r['region_id']:r for r in pool['plan']['regions']}
        selected=[];control=[];regions=[]
        for r in old['regions']:
            ids=sorted(r['members'],key=lambda i:keys[i]);plan=plans[r['region_id']]
            appearance,_,_=weighted_support(f,ids,plan['seed_indices'],pool['initial_scores'])
            prefix,_=local_prefix(f,ids,r['prefix'][0],appearance,keys,min(32,len(ids)))
            if prefix!=r['prefix']:raise ValueError('原局部次序未复现')
            # 步骤2：旧时间函数已包含全区粗分数；细帧只继承邻近支持，不造ITM分数。
            times=[source_time(pool['candidates'][i],pool['video']) for i in ids]
            scores=[pool['initial_scores'][i] if i<pool['initial_count'] else None for i in ids]
            temporal=temporal_support(times,scores,4.)
            support=np.maximum(appearance,temporal)
            new,steps=local_prefix(f,ids,r['prefix'][0],support,keys,min(32,len(ids)))
            quota=current['region_budgets'][str(r['region_id'])]
            control.extend(prefix[:quota]);selected.extend(new[:quota])
            regions.append(dict(region_id=r['region_id'],quota=quota,members=ids,old_prefix=prefix,
                prefix=new,appearance=list(map(float,appearance)),temporal=list(map(float,temporal)),steps=steps))
        control.sort(key=lambda i:keys[i]);selected.sort(key=lambda i:keys[i])
        if control!=actual['selected_indices']:raise ValueError('原正式选择未复现')
        if len(selected)!=len(set(selected)) or len(selected)!=len(control):raise ValueError('预算变化')
        # 步骤3：输出精确目标与完整变化，不把目标缺失等同语义证据损失。
        target_rows=[]
        for i in targets:
            r=next(r for r in regions if i in r['members'])
            target_rows.append(dict(candidate_index=i,old_selected=i in control,new_selected=i in selected,
                position=r['prefix'].index(i)+1 if i in r['prefix'] else None,quota=r['quota']))
        records.append(dict(question_id=q,old_selected=control,new_selected=selected,regions=regions,targets=target_rows))
        print(q,target_rows,flush=True)
    result=dict(sources=sources,records=records,new_model_calls=0,new_answers=0,
        rule='max(appearance support, previous all-coarse exponential temporal support tau=4); fixed exp12 quotas',
        limitation='时间相邻不是语义关系；支持不下降也不保证原证据保留')
    path=run/'analysis.json'
    if path.exists():
        if read_json(path)!=result:raise ValueError('恢复来源或结果变化')
    else:run.mkdir(parents=True,exist_ok=True);durable(path,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    main(parser.parse_args().run)
