#!/usr/bin/env python3
"""冻结200题区域有效性盘点与固定12题复核页；无模型调用。"""
import argparse,json,hashlib,html,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_audit.region_validity_audit import describe_regions,review_candidates

def main(source,out,annotations):
    """步骤1冻结输入哈希；步骤2盘点/核对固定案例；步骤3发布原图链接及人工标注。"""
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    records={};rows=[];hashes={}
    for path in sorted((source/'results/e1').glob('*.json')):
        r=json.loads(path.read_text());records[r['question_id']]=r;rows+=describe_regions(r)
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(records)==200
    chosen=review_candidates(rows)+['591-3','542-2','329-1','485-1'];assert len(set(chosen))==12
    review=json.loads(Path(annotations).read_text());assert set(review)==set(chosen)
    for q,entry in review.items():
        assert set(entry.get('marked_indices',[]))<=set(records[q]['selection']['selected_indices'])
    tail=[r for r in rows if r['last_five_percent']]
    result=dict(question_count=200,region_count=len(rows),tail_anchor_regions=len(tail),tail_anchor_questions=len({r['question_id'] for r in tail}),
        tail_region_budget_sum=sum(r['quota'] for r in tail),review_case_ids=chosen,new_model_calls=0,
        selection_rule='anchor in last5%, quota descending, qid ascending, 8 distinct questions; +4 controls',
        limitation='No end-region invalidity rate; remaining regions are unreviewed, not valid/invalid by default',regions=rows,review=review,source_sha256=hashes)
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    page=['<!doctype html><meta charset="utf-8"><style>body{font:16px/1.7 sans-serif;max-width:1200px;margin:auto}img{width:100%}pre{white-space:pre-wrap}</style><h1>区域有效性：200题自动盘点＋12题有界复核</h1><p>最后5%只是抽查入口，不是过滤器。AI事后初审，无新增模型调用；没有重选或问答成绩。</p><a href="analysis.json">完整统计与源哈希</a>']
    for q in chosen:
        r=records[q];entry=review[q];page.append(f'<details open id="{q}"><summary>{q}</summary><p>{html.escape(r["question"])}</p><pre>{html.escape(str(r["options"]))}</pre><p>{html.escape(entry["finding"])}</p><pre>{html.escape(str([x for x in rows if x["question_id"]==q]))}</pre><img loading="lazy" src="{os.path.relpath(source/"boards/e1"/(q+".png"),out)}"></details>')
    (out/'index.html').write_text(''.join(page))
    print({k:v for k,v in result.items() if k not in ('regions','review','source_sha256')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--annotations',required=True);a=p.parse_args();main(a.source,a.output,a.annotations)
