#!/usr/bin/env python3
"""七题单次粗候选入口修复重放；标记先冻结，不能据结果修改再跑第二版。"""
import os,sys,html,time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256
from videoqa_methods.followups import durable,utc
from videoqa_audit.entry_repair import replay_coarse
from videoqa_audit.density_local_validation import render
SOURCE=ROOT/'outputs/methods/density_e1_200_20260922_r1'
OUT=ROOT/'outputs/analysis/entry_repair7_20260922_r1'

def main():
    """步骤1核对先前审阅及掩码；步骤2各两种粗选；步骤3并排发布，严禁问答计分。"""
    masks=read_json(OUT/'masks.json');review=read_json(OUT/'review_inventory.json');spec=read_json(OUT/'review_protocol.json')
    assert list(masks['questions'])==spec['question_ids']
    for q,a in masks['questions'].items():
        assert set(a['excluded'])<=set(review[q]['reviewed_candidate_indices'])
        assert not set(a['excluded'])&set(a['uncertain_retained'])
    for name,h in spec['source_files'].items():assert sha256(ROOT/name)==h
    protocol=dict(mask_sha256=sha256(OUT/'masks.json'),review_protocol_sha256=sha256(OUT/'review_protocol.json'),
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/videoqa_audit/entry_repair.py']},
        config=read_json(SOURCE/'protocol.json')['config'],created_at_utc=utc(),
        new_model_calls=0,new_refined_frames=0,comparison='both arms use existing coarse candidates only; no comparison of QA accuracy')
    assert not (OUT/'replay_protocol.json').exists(),'Already frozen: use independent verification, not a second replay/version'
    durable(OUT/'replay_protocol.json',protocol)
    from transformers import SiglipImageProcessor
    processor=SiglipImageProcessor.from_pretrained(read_json(ROOT/'configs/local_model_inventory.json')['models']['vision']['path'],local_files_only=True)
    records={};page=['<!doctype html><meta charset="utf-8"><style>body{font:16px/1.7 sans-serif;max-width:1200px;margin:auto}img{width:100%}pre{white-space:pre-wrap}</style><h1>七题入口人工修复：两组均只用既有粗候选</h1><p>人工定向诊断，不是自动检测器、正式RD-2P输入或新问答成绩。<a href="review.html">标记前候选窗口</a> · <a href="masks.json">事前冻结标记</a></p>']
    started=time.perf_counter()
    for q,annotation in masks['questions'].items():
        r=read_json(SOURCE/'results/e1'/f'{q}.json');s=r['selection'];n=s['initial_count'];rows=s['candidates'][:n]
        f=np.load(SOURCE/r['feature_file'],allow_pickle=False)[:n]
        pair={label:replay_coarse(rows,s['initial_scores'],f,s['video'],protocol['config'],excluded)
            for label,excluded in [('original_coarse',[]),('human_repaired_coarse',annotation['excluded'])]}
        assert pair['original_coarse']['plan']==s['plan']
        for item in pair.values():assert len(item['selected_original_indices'])==16
        if not annotation['excluded']:assert pair['original_coarse']==pair['human_repaired_coarse']
        a,b=pair.values();pair['change']=dict(new_seeds=sorted(set(b['seed_original_indices'])-set(a['seed_original_indices'])),
            added=sorted(set(b['selected_original_indices'])-set(a['selected_original_indices'])),removed=sorted(set(a['selected_original_indices'])-set(b['selected_original_indices'])),
            excluded_selected_before=sorted(set(a['selected_original_indices'])&set(annotation['excluded'])),excluded_selected_after=sorted(set(b['selected_original_indices'])&set(annotation['excluded'])))
        pair['question']=r['question'];pair['options']=r['options'];pair['mask_sha256']=protocol['mask_sha256'];pair['qa_status']='not_run'
        records[q]=pair;durable(OUT/'records'/f'{q}.json',pair)
        page.append(f'<h2 id="{q}">{q}</h2><p>{html.escape(r["question"])}</p><pre>{html.escape(str(r["options"]))}</pre><p>{annotation["reason"]}</p>')
        for label in ('original_coarse','human_repaired_coarse'):
            item=pair[label];page.append(f'<h3>{label}</h3><pre>{html.escape(str(item["regions_original"]))}</pre>')
            for kind,ids in [('seeds',sorted(item['seed_original_indices'])),('selected',item['selected_original_indices'])]:
                target=OUT/'boards'/f'{q}_{label}_{kind}.png';chosen=[rows[i] for i in ids]
                render(target,chosen,s['video'],processor,q+' '+label+' '+kind+'; NO QA')
                page.append(f'<p>{kind}</p><img loading="lazy" src="{target.relative_to(OUT)}">')
    summary=dict(questions=7,conditions=14,new_model_calls=0,new_refined_frames=0,elapsed_seconds=time.perf_counter()-started,
        changes={q:p['change'] for q,p in records.items()},mask_sha256=protocol['mask_sha256'],status='computed_awaiting_visual_review')
    for name,h in spec['source_files'].items():assert sha256(ROOT/name)==h
    assert sha256(OUT/'masks.json')==protocol['mask_sha256']
    durable(OUT/'summary.json',summary);(OUT/'index.html').write_text(''.join(page));print(summary,flush=True)

if __name__=='__main__':main()
