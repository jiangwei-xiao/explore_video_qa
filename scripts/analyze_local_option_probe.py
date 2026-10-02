#!/usr/bin/env python3
"""原问题窗口内的已列见证排名，全部选项对称展示，不生成答案或改写最终帧。"""
import argparse
from fractions import Fraction
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_methods.density_selection import source_time
FACTS=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'


def evaluate(run):
    """接口：仅评价固定事实目录与事前列出的宣传册，不将新细帧混入粗见证分母。"""
    summary=read_json(run/'summary.json')
    spec=read_json(run/'protocol.json')
    assert summary['status']=='completed' and summary['qa_calls']==0
    for p,h in summary['artifacts'].items():assert sha256(p)==h,p
    facts=[dict(f,scope='frozen_coarse_witnesses') for f in read_json(FACTS)['entries'] if f['question_id'] in spec['question_ids']]
    # 步骤1：宣传册只依据已冻结精确时间映射，不按新分数反向挑选补充候选。
    item=read_json(run/'inputs/395-2.json')
    pool=read_json(item['pool'])
    pts=[c['source_pts'] for c in pool['candidates'] if source_time(c,pool['video'])==Fraction('272.75')]
    assert len(pts)==1
    facts.append(dict(question_id='395-2',fact_id='ocean_retreat_leaflet',witness_pts=pts,scope='supplemental_fine_partial_clue'))
    records=[]
    for fact in facts:
        q=fact['question_id']
        item=read_json(run/'inputs'/(q+'.json'))
        pool=read_json(item['pool'])
        result=read_json(run/'results'/(q+'.json'))
        candidates=pool['candidates']
        matches={i for i,c in enumerate(candidates) if c['source_pts'] in fact['witness_pts']}
        assert matches
        windows=[]
        # 步骤2：每个窗口两条件使用同一粗细候选集合；窗外事实明确记录为不可在本轮修复。
        for region in pool['plan']['regions']:
            ids=[i for i,c in enumerate(candidates) if Fraction(region['left_fraction'])<=source_time(c,pool['video'])<=Fraction(region['right_fraction'])]
            local=matches.intersection(ids)
            if not local:continue
            comparison={}
            for name,views in result['scores'].items():
                comparison[name]=[]
                for k,values in enumerate(views):
                    order=sorted(ids,key=lambda i:(-values[i],candidates[i]['source_pts'],i))
                    best=min(order.index(i)+1 for i in local)
                    comparison[name].append(dict(view='ABCD'[k],best_rank=best,
                        normalized_rank=(best-1)/max(1,len(ids)-1),top_index=order[0],
                        top_pts=candidates[order[0]]['source_pts'],top_seconds=candidates[order[0]]['source_seconds']))
            windows.append(dict(region_id=region['region_id'],candidate_count=len(ids),matched_indices=sorted(local),comparisons=comparison))
        records.append(dict(question_id=q,fact_id=fact['fact_id'],scope=fact['scope'],in_original_window=bool(windows),windows=windows))
    # 步骤3：仅以最优查询内名次概括潜在可检索性，不选择正确选项或报告问答准确率。
    counts={}
    for name in ['joined','option']:
        eligible=[r for r in records if r['scope']=='frozen_coarse_witnesses' and r['in_original_window']]
        counts[name]=dict(eligible_facts=len(eligible),top1=sum(min(v['best_rank'] for w in r['windows'] for v in w['comparisons'][name])==1 for r in eligible),
            mean_best_normalized_rank=sum(min(v['normalized_rank'] for w in r['windows'] for v in w['comparisons'][name]) for r in eligible)/len(eligible))
    value=dict(records=records,coarse_fact_count=sum(r['scope']=='frozen_coarse_witnesses' for r in records),
        counts=counts,qa_calls=0,sources={str(p):sha256(p) for p in [FACTS,run/'protocol.json',run/'summary.json',Path(__file__)]})
    dest=run/'local_rank_evaluation.json'
    if dest.exists():assert read_json(dest)==value
    else:durable(dest,value)
    print('Local ranking summary',counts,flush=True)
    for r in records:
        ranks={name:min((v['best_rank'] for w in r['windows'] for v in w['comparisons'][name]),default=None) for name in ['joined','option']}
        print(r['question_id'],r['fact_id'],r['scope'],ranks,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',required=True)
    args=p.parse_args()
    import re
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id)
    evaluate(ROOT/'outputs/analysis'/args.run_id)
