"""评分器产物只读归因：固定事实排名/触达/保留和全量区域结构变化。"""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import rows,durable
from videoqa_audit.framework_contraction import BASE,locations
from videoqa_audit.region_scope_runtime import parent_result
from videoqa_methods.density_selection import source_time
from .preparation import METHODS,PROFILES,source_keys

FACTS=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'


def stable_ranks(frames,scores):
    """接口：原题内部稳定排名，分数跨问题/模型不可直接比较为证据概率。"""
    order=sorted(range(len(scores)),key=lambda i:(-scores[i],frames[i]['source_pts'],frames[i]['candidate_index']))
    return {frames[i]['source_pts']:r+1 for r,i in enumerate(order)}


def witness_status(pool,frames,witness_pts):
    """接口：仅已列见证的窗口触达与最终保留，不推断未标注候选的语义价值。"""
    coarse=pool['candidates'][:pool['initial_count']];rank=stable_ranks(coarse,pool['initial_scores'])
    pts=set(witness_pts);selected={f['source_pts'] for f in frames};present={c['source_pts'] for c in coarse}
    if not pts<=present:raise ValueError('Frozen coarse witnesses missing from initial identity')
    best=min(rank[p] for p in pts);windows=pool['plan']['regions']
    reachable=any(any(Fraction(r['left_fraction'])<=source_time(c,pool['video'])<=Fraction(r['right_fraction'])
                      for r in windows) for c in coarse if c['source_pts'] in pts)
    return dict(best_rank=best,normalized_rank=best/len(coarse),window_reachable=reachable,
                selected_exact=bool(pts&selected),selected_witness_pts=sorted(pts&selected))


def selection_analysis(root):
    """接口：5400配置题结构对照，42题75事实固定分母；运行后人工判断另留待审。"""
    root=Path(root);facts=read_json(FACTS)['entries']
    if len(facts)!=75 or len({f['question_id'] for f in facts})!=42:raise ValueError('Proxy denominator changed')
    grouped=defaultdict(list)
    for f in facts:grouped[f['question_id']].append(f)
    summary={}
    for profile,k in PROFILES.items():
        records=[];proxy=[]
        for row in rows():
            q=row['question_id'];oldpath=locations(q,k)[0];oldpool=read_json(oldpath)
            newpath=root/'pools'/profile/(q+'.json');newpool=read_json(newpath)
            oldseeds=set(source_keys([oldpool['candidates'][i] for i in oldpool['plan']['seed_indices']]))
            newseeds=set(source_keys([newpool['candidates'][i] for i in newpool['plan']['seed_indices']]))
            record=dict(question_id=q,old_pool_sha256=sha256(oldpath),new_pool_sha256=sha256(newpath),
                seed_overlap=len(oldseeds&newseeds),seed_counts=[len(oldseeds),len(newseeds)],
                region_counts=[len(oldpool['plan']['regions']),len(newpool['plan']['regions'])],
                window_union_seconds=[float(sum((Fraction(r['right_fraction'])-Fraction(r['left_fraction'])
                    for r in p['plan']['regions']),Fraction(0))) for p in (oldpool,newpool)],
                refinement_candidate_counts=[len(p['candidates'])-p['initial_count'] for p in (oldpool,newpool)],methods={})
            for m in METHODS:
                old=read_json(parent_result(profile,q,m))['selection']['selected_frames']
                new=read_json(root/'selections'/profile/m/(q+'.json'))['selection']['selected_frames']
                a,b=set(source_keys(old)),set(source_keys(new));n=newpool['initial_count']
                coarsekeys=set(source_keys(newpool['candidates'][:n]))
                record['methods'][m]=dict(old_frame_count=len(a),new_frame_count=len(b),same_source_input=source_keys(old)==source_keys(new),
                    overlap=len(a&b),jaccard=len(a&b)/len(a|b),retained_S2_seeds=len(b&newseeds),selected_S2_fine=len(b-coarsekeys))
                for f in grouped[q]:
                    entry=dict(question_id=q,fact_id=f['fact_id'],description=f['description'],role=f['role'],method=m,
                        S1=witness_status(oldpool,old,f['witness_pts']),S2=witness_status(newpool,new,f['witness_pts']))
                    proxy.append(entry)
            durable(root/'selection_analysis'/profile/(q+'.json'),record);records.append(record)
        macro={}
        for m in METHODS:
            byq=defaultdict(list)
            for f in proxy:
                if f['method']==m:byq[f['question_id']].append(f)
            macro[m]={s:float(np.mean([np.mean([f[s]['selected_exact'] for f in fs]) for fs in byq.values()])) for s in ('S1','S2')}
        durable(root/'proxy'/ (profile+'.json'),dict(source_sha256=sha256(FACTS),facts=75,questions=42,
            records=proxy,macro_listed_witness_retention=macro,
            limitation='Exact old witness hits only; unlisted new-frame alternatives not judged absent'))
        summary[profile]=dict(questions=2700,changed_seed_questions=sum(r['seed_overlap']!=r['seed_counts'][0] or r['seed_counts'][0]!=r['seed_counts'][1] for r in records),
            mean_seed_overlap=float(np.mean([r['seed_overlap'] for r in records])),
            changed_input_questions={m:sum(not r['methods'][m]['same_source_input'] for r in records) for m in METHODS},
            proxy_macro=macro,frame_review_status='pending_human_or_AI_actual_frame_review; not inferred from QA')
    durable(root/'selection_analysis/summary.json',dict(status='completed',profiles=summary,
        facts_denominator=dict(questions=42,facts=75),new_model_calls=0))
    return summary
