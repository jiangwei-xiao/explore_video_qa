"""共同粗池上的直接检索支持下界诊断，不改冻结方法、不调用模型。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE
from videoqa_methods.coarse_only_ablation import select_coarse_only
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_methods.density_selection import frame_key

PARENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
TARGETS={'409-3':[77,80],'556-1':[42],'395-2':[276],'584-3':[582],
         '807-2':[1646],'383-2':[944],'538-1':[11],'705-1':[2383],
         '893-1':[699,706],'305-3':[199],'170-2':[10],'191-3':[6], '648-3':[1386,1486]}


def prefix_records(pool,features,baseline,enabled):
    """接口：单一改动S=max(L,r/max_region_r)，零分全区时直接分量为0。

    两路只是不同检索支持代理，不称回答概率。共同粗池避免对新旧来源区别待遇。
    """
    records=[];f=features
    plans={r['region_id']:r for r in pool['plan']['regions']}
    # 步骤1：原粗参考归属、种子、锚点、总预算保持一致。
    for old in baseline['coarse_prefix_trace']['regions']:
        ids=old['members'];plan=plans[old['region_id']]
        support,_,_=weighted_support(f,ids,plan['seed_indices'],pool['initial_scores'])
        raw=np.asarray(pool['initial_scores'],dtype=np.float64)[ids]
        direct=raw/raw.max() if raw.max()>0 else np.zeros_like(raw)
        combined=np.maximum(support,direct) if enabled else support
        selected=[plan['anchor_index']]
        D=np.clip(f[ids]@f[selected[0]],0,1)
        covered=D.copy();curve=[0.,float(covered.mean(dtype=np.float64))];steps=[]
        # 步骤2：保持原sqrt差异与稳定排序，逐步保存直接/传播贡献。
        while len(selected)<min(32,len(ids)):
            proposals=[(float(combined[j])*math.sqrt(1-float(D[j])),j,i)
                       for j,i in enumerate(ids) if i not in selected]
            value,j,i=min(proposals,key=lambda p:(-p[0],*frame_key(pool['candidates'][p[2]])))
            steps.append(dict(candidate_index=i,propagated=float(support[j]),direct=float(direct[j]),
                              support=float(combined[j]),redundancy=float(D[j]),score=value))
            selected.append(i)
            D=np.maximum(D,np.clip(f[ids]@f[i],0,1));covered=D.copy()
            curve.append(float(covered.mean(dtype=np.float64)))
        if not enabled and selected!=old['prefix']:
            raise ValueError('原局部次序未复现')
        records.append(dict(region_id=old['region_id'],members=ids,reference_coarse_indices=ids,
            prefix=selected,coverage_curve=curve,density_weight=old['density_weight'],steps=steps))
    return records


def main(run):
    """接口：原12题加已定位的648-3补充例；无新细帧评分，不生成准确率。"""
    sources={str(Path(__file__)):sha256(__file__)};outputs=[]
    # 步骤1：先复算原共同粗池控制，当前正式混合池仅作为来源而非可比成绩。
    for q,targets in TARGETS.items():
        paths=[BASE/'pools'/(q+'.json'),BASE/'features'/(q+'.npy'),PARENT/'traces'/(q+'.json')]
        for path in paths:sources[str(path)]=sha256(path)
        pool=read_json(paths[0]);full_features=np.load(paths[1],allow_pickle=False);parent=read_json(paths[2])
        parent=dict(parent,protected_indices=[r['anchor_index'] for r in pool['plan']['regions']])
        old_selected,baseline=select_coarse_only(pool,full_features,parent)
        n=pool['initial_count'];coarse=dict(pool,candidates=pool['candidates'][:n]);f=full_features[:n]
        control=prefix_records(coarse,f,baseline,False)
        verify_ref=dict(regions=control,region_budgets=baseline['region_budgets'])
        verified,_=select_relevance_coverage(coarse,f,verify_ref)
        if verified!=old_selected:raise ValueError('共同粗池控制选择不一致')
        new_regions=prefix_records(coarse,f,baseline,True)
        # 步骤2：固定旧配额隔离局部效应；另列同一预算公式对新前缀的自然响应。
        fixed=sorted([i for r in new_regions for i in r['prefix'][:baseline['region_budgets'][str(r['region_id'])]]],key=lambda i:frame_key(coarse['candidates'][i]))
        new,trace=select_relevance_coverage(coarse,f,dict(regions=new_regions,region_budgets=baseline['region_budgets']))
        if not set(baseline['protected_indices'])<=set(new) or len(new)!=len(old_selected):
            raise ValueError('锚点或预算改变')
        target_rows=[]
        for i in targets:
            old=next(r for r in control if i in r['members']);fresh=next(r for r in new_regions if i in r['members'])
            target_rows.append(dict(candidate_index=i,old_position=old['prefix'].index(i)+1 if i in old['prefix'] else None,
                new_position=fresh['prefix'].index(i)+1 if i in fresh['prefix'] else None,
                old_selected=i in old_selected,fixed_quota_selected=i in fixed,new_selected=i in new))
        outputs.append(dict(question_id=q,old_selected=old_selected,fixed_quota_selected=fixed,
            new_selected=new,old_quotas=baseline['region_budgets'],new_quotas=trace['region_budgets'],
            regions=new_regions,targets=target_rows))
        print(q,target_rows,flush=True)
    # 步骤3：只保存离线候选变化，不附用旧预测；恢复一致才可复用。
    result=dict(sources=sources,records=outputs,new_model_calls=0,
        limitation='共同粗池局部支持诊断；不是exp12混合池正式变体，不能复用历史答案')
    path=run/'analysis.json'
    if path.exists():
        if read_json(path)!=result:raise ValueError('恢复来源或输出变化')
    else:
        run.mkdir(parents=True,exist_ok=True);durable(path,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    main(parser.parse_args().run)
