#!/usr/bin/env python3
"""七题推广入口诊断的标注准备：只解码已有1FPS候选，不运行模型或选帧优化。"""
import os,sys,math,html
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_methods.followups import durable
from videoqa_audit.density_local_validation import render
WINDOWS={'017-2':[(0,13),(84,98)],'348-2':[(315,390)],'591-3':[(662,690)],
         '152-2':[(68,90)],'826-1':[(1825,1881)],'542-2':[(850,880)],'880-3':[(2670,2730)]}
SOURCE=ROOT/'outputs/methods/density_e1_200_20260922_r1'
OUT=ROOT/'outputs/analysis/entry_repair7_20260922_r1'

def main():
    """步骤1冻结审核范围与来源；步骤2生成全量窗口粗帧；步骤3提供不含预测的标注页。"""
    from transformers import SiglipImageProcessor
    OUT.mkdir(parents=True,exist_ok=True)
    spec=dict(question_ids=list(WINDOWS),review_windows=WINDOWS,source_protocol_sha256=sha256(SOURCE/'protocol.json'),
        policy='human confirmed promotion excluded from coarse pool; uncertain content retained; both arms coarse-only, no refinement, no QA',
        code_sha256=sha256(Path(__file__)),new_model_calls=0,source_files={})
    for q in WINDOWS:
        p=SOURCE/'results/e1'/f'{q}.json';r=read_json(p)
        for x in (p,SOURCE/r['feature_file']):spec['source_files'][str(x.relative_to(ROOT))]=sha256(x)
        assert sha256(SOURCE/r['feature_file'])==r['feature_sha256']
    if (OUT/'review_protocol.json').exists():assert read_json(OUT/'review_protocol.json')==spec
    else:durable(OUT/'review_protocol.json',spec)
    processor=SiglipImageProcessor.from_pretrained(read_json(ROOT/'configs/local_model_inventory.json')['models']['vision']['path'],local_files_only=True)
    pages=[];inventory={}
    for q,windows in WINDOWS.items():
        r=read_json(SOURCE/'results/e1'/f'{q}.json');s=r['selection'];coarse=s['candidates'][:s['initial_count']]
        rows=[x for x in coarse if any(lo<=x['timestamp_seconds']<=hi for lo,hi in windows)]
        inventory[q]=dict(question=r['question'],options=r['options'],video=s['video'],reviewed_candidate_indices=[x['candidate_index'] for x in rows],pages=[])
        pages.append(f'<h2>{q}</h2><p>{html.escape(r["question"])}</p><pre>{html.escape(str(r["options"]))}</pre>')
        for i in range(0,len(rows),16):
            p=OUT/'review_boards'/f'{q}_{i//16:02}.png'
            if not p.exists():render(p,rows[i:i+16],s['video'],processor,q+' existing coarse candidates; NOT new sampling')
            inventory[q]['pages'].append(str(p.relative_to(OUT)));pages.append(f'<img loading="lazy" src="{p.relative_to(OUT)}">')
    durable(OUT/'review_inventory.json',inventory)
    (OUT/'review.html').write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:1200px;margin:auto}img{width:100%}pre{white-space:pre-wrap}</style><h1>标记前：既有1FPS候选窗口</h1>'+''.join(pages))
    print({q:len(v['reviewed_candidate_indices']) for q,v in inventory.items()},flush=True)

if __name__=='__main__':main()
