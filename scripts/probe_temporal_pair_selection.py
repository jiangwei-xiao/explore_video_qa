"""固定200题的时间patch配对候选离线准备，问答与重新编码调用均为0。"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive
from videoqa_audit.query_input_probe import BASE
from videoqa_methods.optional_region_selection import select_optional_regions
from videoqa_methods.temporal_pair_selection import select_temporal_pairs


def main():
    """接口：先重放原K32控制，再计算16代表＋16邻帧，并独立评估旧事实目录。"""
    run=ROOT/'outputs/analysis/temporal_pair_selection200_20261004_r1'
    parent=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
    manifest=ROOT/'data/manifests/video_mme_development_200_20260920.json'
    facts=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'
    with exclusive(run/'run.lock'):
        ids=[r['question_id'] for r in read_json(manifest)['rows']]
        sources={str(p):sha256(p) for p in [Path(__file__),manifest,
            ROOT/'src/videoqa_methods/temporal_pair_selection.py',ROOT/'src/videoqa_methods/optional_region_selection.py']}
        protocol=dict(sources=sources,question_ids=ids,seed_budget=32,source_budget=32,representative_budget=16,
            temporal_patch_size=2,new_model_calls=0,new_answers=0,
            fallback='entire original exp12 input for short pool or infeasible same-region adjacent matching',
            rule='original L*sqrt novelty and weighted coverage for representatives; maximum-coherence direct neighbours')
        pp=run/'protocol.json'
        if pp.exists():
            if read_json(pp)!=protocol:raise ValueError('Offline pair protocol changed')
        else:durable(pp,protocol)
        selections={};rows=[]
        # 步骤1：原K32离散选择必须逐题复现，不附用旧答案为新选择计分。
        for q in ids:
            paths=dict(pool=BASE/'pools'/(q+'.json'),features=BASE/'features'/(q+'.npy'),
                       original=parent/'selections/RD-1.2-exp12'/(q+'.json'))
            fingerprints={str(p):sha256(p) for p in paths.values()}
            dest=run/'selections'/(q+'.json')
            old=read_json(paths['original']);pool=read_json(paths['pool'])
            if dest.exists():
                record=read_json(dest)
                if record['sources']!=fingerprints or record['protocol_sha256']!=sha256(pp):raise ValueError('Offline assets changed')
            else:
                f=np.load(paths['features'],allow_pickle=False)
                control,_=select_optional_regions(pool,f,frame_budget=32)
                if control!=old['selected_indices']:raise ValueError('Original exp12 control changed')
                selected,trace=select_temporal_pairs(pool,f,control)
                record=dict(question_id=q,selected_indices=selected,trace=trace,sources=fingerprints,
                    protocol_sha256=sha256(pp),selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in selected]))
                durable(dest,record)
            selections[q]=(old,record)
            rows.append(dict(question_id=q,fallback=record['trace']['fallback'],
                changed=old['selected_indices']!=record['selected_indices'],
                selected_overlap=len(set(old['selected_indices'])&set(record['selected_indices']))))
            print(q,'fallback',record['trace']['fallback'],'overlap',rows[-1]['selected_overlap'],flush=True)
        # 步骤2：人工目录只在选择完成后用于评价，不传入代表或邻帧选择函数。
        entries=read_json(facts)['entries']
        if len(entries)!=75 or len({x['question_id'] for x in entries})!=42:raise ValueError('Fact denominator changed')
        evidence=[]
        for fact in entries:
            old,new=selections[fact['question_id']]
            hits=[bool(set(fact['witness_pts'])&{f['source_pts'] for f in item['selection']['selected_frames']}) for item in [old,new]]
            evidence.append(dict(question_id=fact['question_id'],fact_id=fact['fact_id'],old_hit=hits[0],new_hit=hits[1]))
        # 步骤3：独立记录失败回退和覆盖代价，不把更相近的配对直接当作更好的证据。
        summary=dict(questions=len(rows),changed=sum(r['changed'] for r in rows),
            fallback_ids=[r['question_id'] for r in rows if r['fallback']],rows=rows,
            fact_source_sha256=sha256(facts),facts=evidence,old_fact_hits=sum(r['old_hit'] for r in evidence),
            new_fact_hits=sum(r['new_hit'] for r in evidence),new_model_calls=0,new_answers=0,
            files={str(run/'selections'/(q+'.json')):sha256(run/'selections'/(q+'.json')) for q in ids})
        path=run/'summary.json'
        if path.exists():
            if read_json(path)!=summary:raise ValueError('Summary changed on resume')
        else:durable(path,summary)
        print({k:v for k,v in summary.items() if k not in ['rows','facts','files']},flush=True)


if __name__=='__main__':main()
