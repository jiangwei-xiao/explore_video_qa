"""只读检查局部首轮选择中支持和去重是否退化为同一个锚点相似度。"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
import argparse
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE,FULL
from videoqa_methods.weighted_seed_support import weighted_support


def first_step(features,region,plan,scores,candidates):
    """接口：独立复算锚点后的首个竞争位置，标记L≈D而不判断画面语义。"""
    # 步骤1：沿用冻结种子和实际区域成员，复现浮点运算与稳定排序。
    ids=region['members'];anchor=plan['anchor_index']
    support,_,_=weighted_support(features,ids,plan['seed_indices'],scores)
    redundancy=np.clip(features[ids]@features[anchor],0,1)
    available=[j for j,i in enumerate(ids) if i!=anchor]
    if not available:return None
    values=[float(support[j])*math.sqrt(1-float(redundancy[j])) for j in range(len(ids))]
    winner=min(available,key=lambda j:(-values[j],candidates[ids[j]]['source_pts'],ids[j]))
    expected=region['local_steps'][0]
    if ids[winner]!=expected['candidate_index'] or abs(values[winner]-expected['local_score'])>1e-6:
        raise ValueError('Frozen local winner or score mismatch')
    # 步骤2：L=D只作数值退化诊断；2/3来自s*sqrt(1-s)的解析极值，不是新阈值。
    coupled=[j for j in available if abs(float(support[j])-float(redundancy[j]))<=1e-6]
    return dict(anchor=anchor,winner=ids[winner],support=float(support[winner]),
        redundancy=float(redundancy[winner]),score=values[winner],
        candidate_count=len(available),coupled_candidate_count=len(coupled),
        winner_coupled=winner in coupled,
        winner_similarity_distance_from_two_thirds=abs(float(redundancy[winner])-2/3),
        top_candidates=[dict(candidate_index=ids[j],source_frame=candidates[ids[j]],
            support=float(support[j]),redundancy=float(redundancy[j]),score=values[j])
            for j in sorted(available,key=lambda j:(-values[j],candidates[ids[j]]['source_pts'],ids[j]))[:5]])


def main(run):
    """接口：冻结开发200题全区域扫描，无问答、无重评分，不改生产选择器。"""
    sources={str(Path(__file__)):sha256(__file__)}
    def read(path):
        sources[str(path)]=sha256(path)
        return read_json(path)
    data=read(ROOT/'data/manifests/video_mme_development_200_20260920.json')['rows']
    if len(data)!=200 or len({r['question_id'] for r in data})!=200:raise ValueError('Manifest changed')
    records=[]
    # 步骤1：逐题仅装载既有池和特征；区域内首轮与exp12共享同一冻结前缀。
    for row in data:
        q=row['question_id'];pool=read(BASE/'pools'/(q+'.json'));trace=read(FULL/'traces'/(q+'.json'))
        path=BASE/'features'/(q+'.npy');sources[str(path)]=sha256(path)
        features=np.load(path,allow_pickle=False)
        plans={r['region_id']:r for r in pool['plan']['regions']}
        for region in trace['regions']:
            first=first_step(features,region,plans[region['region_id']],pool['initial_scores'],pool['candidates'])
            if first is not None:records.append(dict(question_id=q,region_id=region['region_id'],**first))
    # 步骤2：只描述数值出现频率，不把区域视作独立样本或推断这些赢家是错误帧。
    summary=dict(questions=200,nontrivial_regions=len(records),
        coupled_winner_regions=sum(r['winner_coupled'] for r in records),
        questions_with_coupled_winner=len({r['question_id'] for r in records if r['winner_coupled']}),
        candidate_slots=sum(r['candidate_count'] for r in records),
        coupled_candidate_slots=sum(r['coupled_candidate_count'] for r in records),
        analytic_max_similarity=2/3,analytic_max_score=2/(3*math.sqrt(3)),
        new_model_calls=0,scope='First supplemental position after anchor; no semantic quality or accuracy inference')
    result=dict(sources=sources,summary=summary,records=records)
    output=run/'analysis.json'
    if output.exists():
        if read_json(output)!=result:raise ValueError('Resume changed inputs or results')
    else:
        run.mkdir(parents=True,exist_ok=True);durable(output,result)
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    main(Path(parser.parse_args().run))
