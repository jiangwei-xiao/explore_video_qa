#!/usr/bin/env python3
"""将开发50题扩至200题；按视频盲抽，不改原开发/预留清单或实验结果。"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_methods.followups import durable


def select_new_rows(full,development,reserve):
    """接口：仅用视频ID、题号、时长层确定150题；答案和历史预测不参与排名。"""
    excluded={r['video_id'] for r in development+reserve}
    groups={s:{} for s in ('short','medium','long')}
    # 步骤1：以视频为单位排除旧开发和原预留，避免近重复泄漏。
    for row in full:
        if row['video_id'] not in excluded:
            groups[row['stratum']].setdefault(row['video_id'],[]).append(row)
    def rank(text):return hashlib.sha256(('2027:'+text).encode()).hexdigest()
    chosen={}
    # 步骤2：每层按稳定哈希取50个视频；每视频再以题号哈希选择一道题。
    for stratum,videos in groups.items():
        assert len(videos)>=50
        ids=sorted(videos,key=lambda v:(rank('video:'+stratum+':'+v),v))[:50]
        chosen[stratum]=[min(videos[v],key=lambda r:(rank('question:'+r['question_id']),r['question_id'])) for v in ids]
    return [chosen[s][i] for i in range(50) for s in ('short','medium','long')]


def main():
    """冻结扩展清单及两组基线身份；重复执行只核对，不改变抽样。"""
    base=ROOT/'data/manifests'
    files={k:base/v for k,v in dict(full='video_mme_full_2700.json',development='video_mme_development.json',reserve='video_mme_reserve.json').items()}
    full,old,reserve=[read_json(files[k])['rows'] for k in ('full','development','reserve')]
    added=select_new_rows(full,old,reserve);rows=old+added
    assert len(rows)==200 and len({r['video_id'] for r in rows})==200
    assert Counter(r['stratum'] for r in added)=={'short':50,'medium':50,'long':50}
    assert not ({r['video_id'] for r in rows}&{r['video_id'] for r in reserve})
    # 步骤3：选题后才接入历史结果；不根据成绩改变选择集合。
    merged=ROOT/'outputs/baselines/videomme_full_uniform_topk_20260916_merged/per_question.csv'
    source_map={'r1':'videomme_full_uniform_topk_20260916_r1',
                'completion_r1':'videomme_full_completion_20260917_r1',
                'completion_r2':'videomme_full_completion_20260917_r2'}
    references={};wanted={r['question_id']:r for r in rows}
    for item in csv.DictReader(merged.open()):
        q=item['question_id'];method=item['method']
        if q not in wanted:continue
        label=item['source_run']
        if label not in source_map:raise ValueError('Unknown baseline source label: '+label)
        path=ROOT/'outputs/baselines'/source_map[label]/'results'/method/f'{q}.json'
        record=read_json(path);r=wanted[q]
        assert record['question_id']==q and record['video_id']==r['video_id']
        assert record['question']==r['question'] and record['options']==r['options']
        assert record['reference_answer']=='ABCD'[r['answer_index']]
        assert record['correct']==(item['correct']=='True')
        references.setdefault(q,{})[method]=dict(path=str(path.relative_to(ROOT)),sha256=sha256(path),correct=record['correct'])
    assert set(references)==set(wanted) and all(set(v)=={'uniform','topk'} for v in references.values())
    manifest=dict(schema_version=1,dataset='Video-MME',split='development_expanded_200',seed=2027,
        sampling='Keep original50; add50 unseen-development videos per duration; deterministic SHA256 ranks; one question/video; reserve video exclusion',
        source_hashes={k:sha256(v) for k,v in files.items()},old_question_ids=[r['question_id'] for r in old],
        new_question_ids=[r['question_id'] for r in added],rows=rows,
        strata=dict(Counter(r['stratum'] for r in rows)),
        boundary='All full-dataset baselines have been evaluated; expanded development assessment, not pristine held-out confirmation')
    destination=base/'video_mme_development_200_20260920.json'
    if destination.exists():assert read_json(destination)==manifest
    else:durable(destination,manifest)
    audit=dict(manifest_sha256=sha256(destination),baseline_csv_sha256=sha256(merged),references=references,
        old_videos=50,new_videos=150,unique_videos=200,reserve_overlap=0)
    target=base/'video_mme_development_200_baseline_links_20260920.json'
    if target.exists():assert read_json(target)==audit
    else:durable(target,audit)
    print(json.dumps(dict(total=200,new=150,strata=manifest['strata'],reserve_overlap=0,baseline_records=400,manifest_sha256=sha256(destination)),indent=2))


if __name__=='__main__':main()
