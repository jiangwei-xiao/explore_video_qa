#!/usr/bin/env python3
"""冻结64帧探针：32熟悉案例＋32规则抽取的新视频帧；先渲染、标注，后识别。"""
import os,sys,hashlib,html
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_methods.followups import durable
from videoqa_audit.density_local_validation import render
from videoqa_audit.promotion_probe import INSTRUCTION,OPTIONS
SOURCE=ROOT/'outputs/methods/density_e1_200_20260922_r1'
OUT=ROOT/'outputs/analysis/promotion_probe64_20260922_r1'
KNOWN={'017-2':[1,4,92,95],'348-2':[333,349,355,374,327,336,362,380],
       '591-3':[665,674,678,681],'152-2':[3,43,79],'826-1':[1844,1860,1864],
       '542-2':[859,863,867],'880-3':[2683,2689,2700],'395-2':[273,288],'329-1':[9],'485-1':[411]}
EXCLUDED={'314-3','414-1','821-3','278-1','102-2','736-2','842-3','550-3','591-3','485-1','329-1','737-3','152-2','017-2','348-2','826-1','529-1','190-1','880-3','542-2','724-3','500-3','358-2','881-2'}|set(KNOWN)

def main():
    """步骤1仅按身份/排名/时间选帧；步骤2冻结提示词及清单；步骤3生成无预测审核页。"""
    m=read_json(ROOT/'data/manifests/video_mme_development_200_20260920.json');rows=[];groups={'tail':[],'other':[]}
    def source(q):return read_json(SOURCE/'results/e1'/f'{q}.json')
    def tail_ids(r):
        s=r['selection'];return [i for i in s['plan']['seed_indices'] if s['candidates'][i]['source_seconds']/s['video']['duration_seconds']>=.95]
    for q in m['new_question_ids']:
        if q in EXCLUDED:continue
        r=source(q);groups['tail' if tail_ids(r) else 'other'].append(q)
    fresh={}
    for group,qs in groups.items():
        for q in sorted(qs,key=lambda q:hashlib.sha256(('promo-probe-v1:'+q).encode()).hexdigest())[:8]:
            r=source(q);rank=r['selection']['plan']['seed_indices'];tails=tail_ids(r)
            first=tails[0] if tails else rank[0]
            candidates=[i for i in rank if i!=first and i not in tails]
            second=(candidates[0] if candidates else next(i for i in rank if i!=first)) if tails else rank[-1]
            fresh[q]=[first,second]
    for cohort,items in [('familiar',KNOWN),('new_frames',fresh)]:
        for q,ids in items.items():
            r=source(q);s=r['selection']
            for i in ids:
                assert i<s['initial_count']
                rows.append(dict(sample_id=f'{q}__{i}',cohort=cohort,question_id=q,question=r['question'],
                    candidate=s['candidates'][i],video=s['video'],source_result_sha256=sha256(SOURCE/'results/e1'/f'{q}.json')))
    assert len(rows)==64 and len({r['sample_id'] for r in rows})==64 and len(fresh)==16
    OUT.mkdir(parents=True,exist_ok=True)
    spec=dict(rows=rows,instruction=INSTRUCTION,options=OPTIONS,maximum_auxiliary_calls=64,original_QA_calls=0,
        decision='only A proposes exclusion; B/C/D and parse failures keep; no actual selection changes',
        sample_rule='new150 excluding previously reviewed IDs; hash-ranked8 tail and8 other videos, two seeds each',
        previously_reviewed_ids=sorted(EXCLUDED),limitations='new frames are not an untouched test set; all drawn from observed development200',
        predeclared_gate='Any confirmed useful-frame exclusion blocks integration. Familiar definite promotion recall must reach 8/12. New-frame positives fewer than4 means recall evidence insufficient. No gate authorizes QA.',
        code_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/videoqa_audit/promotion_probe.py']})
    if (OUT/'protocol.json').exists():assert read_json(OUT/'protocol.json')==spec
    else:durable(OUT/'protocol.json',spec)
    from transformers import SiglipImageProcessor
    processor=SiglipImageProcessor.from_pretrained(read_json(ROOT/'configs/local_model_inventory.json')['models']['vision']['path'],local_files_only=True)
    page=['<!doctype html><meta charset="utf-8"><style>body{max-width:1000px;margin:auto}img{max-width:100%}</style><h1>64帧事前标注：不含预测或标准答案</h1>']
    for row in rows:
        p=OUT/'frames'/(row['sample_id']+'.png')
        if not p.exists():render(p,[row['candidate']],row['video'],processor,row['sample_id']+' '+row['cohort'])
        page.append(f'<h2>{row["sample_id"]}</h2><p>{html.escape(row["question"])}</p><img src="frames/{p.name}">')
    (OUT/'annotation.html').write_text(''.join(page));print([(q,v) for q,v in fresh.items()],flush=True)

if __name__=='__main__':main()
