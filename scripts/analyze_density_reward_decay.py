#!/usr/bin/env python3
"""D1固定规则全200题离线重放；11题看图，不新增任何模型调用或正式QA。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse
import html
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_methods.followups import durable
from videoqa_audit.density_reward_decay import select_density_decay
from videoqa_audit.density_local_validation import supported_selection,render

SOURCE=ROOT/'outputs/methods/density_extension200_20260920_r1'
CASES=('103-1','314-3','383-2','414-1','821-3','278-1','260-2','004-3','102-2','736-2','842-3')


def main(run_id):
    """冻结来源，原软折中先复现，再只改跨区公式；事后查看不改变规则。"""
    if not run_id.replace('_','').isalnum():raise ValueError('Invalid run id')
    out=ROOT/'outputs/analysis'/run_id;out.mkdir(parents=True,exist_ok=False);(out/'records').mkdir();(out/'boards').mkdir()
    manifest=read_json(ROOT/'data/manifests/video_mme_development_200_20260920.json');old=set(manifest['old_question_ids']);cfg=read_json(SOURCE/'protocol.json')['config']
    frozen=read_json(SOURCE/'protocol.json')
    for name,digest in {**frozen['code_sha256'],**frozen['source_results']}.items():assert sha256(ROOT/name)==digest,name
    protocol=dict(local_rule='L*u',global_rule='(1+(w-1)/n)*L*u',config=cfg,source_protocol_sha256=sha256(SOURCE/'protocol.json'),
        questions=[r['question_id'] for r in manifest['rows']],visual_review=list(CASES),new_model_calls=0,
        code_hashes={str(p.relative_to(ROOT)):sha256(p) for p in (Path(__file__),ROOT/'src/videoqa_audit/density_reward_decay.py')},
        caveat='Density premium w-1 is divided by already-selected count n including anchor; base1 remains; first proposal unchanged; no free exponent/threshold. Attribution heuristic, not claimed globally optimal; no QA or parameter search')
    durable(out/'protocol.json',protocol);hashes={};records={};started=time.perf_counter()
    # 步骤1：复现全部原soft输入，消除环境或特征差异，再计算唯一D1候选。
    for ordinal,row in enumerate(manifest['rows']):
        q=row['question_id'];vp=SOURCE/'combined_results/v1'/f'{q}.json';sp=SOURCE/'combined_results/soft'/f'{q}.json'
        a,b=read_json(vp),read_json(sp);s=a['selection']
        base=ROOT/'outputs/methods/retrieval_density_v1_videomme50_20260920_r1' if q in old else SOURCE
        path=base/a['feature_file'];assert sha256(path)==a['feature_sha256'];f=np.load(path,allow_pickle=False)
        for p in (vp,sp,path):hashes[str(p.relative_to(ROOT))]=sha256(p)
        baseline,_=supported_selection(s['candidates'],f,s['plan'],s['video'],cfg)
        assert baseline==b['selection']['selected_indices'],q
        ids,trace=select_density_decay(s['candidates'],f,s['plan'],s['video'],cfg)
        assert len(ids)==len(set(ids))==16 and set(trace['protected_indices'])<=set(ids)
        record=dict(question_id=q,cohort='old50' if q in old else 'new150',selected_indices=ids,
            selected_frames=[s['candidates'][i] for i in ids],trace=trace,
            changed_vs_soft=len(set(ids)-set(baseline)),changed_vs_v1=len(set(ids)-set(s['selected_indices'])),
            budgets_soft=b['selection']['selection_trace']['region_budgets'],budgets_v1=s['selection_trace']['region_budgets'],
            new_model_calls=0,qa_status='not_run')
        records[q]=record;durable(out/'records'/f'{q}.json',record)
        durable(out/'status.json',dict(status='offline',completed=ordinal+1,total=200))
    # 步骤2：11题正式旧输入和D1输入并排，图像仅CPU处理，不执行视觉编码器。
    from transformers import SiglipImageProcessor
    processor=SiglipImageProcessor.from_pretrained(read_json(ROOT/'configs/local_model_inventory.json')['models']['vision']['path'],local_files_only=True)
    page=['<!doctype html><meta charset="utf-8"><style>body{font:16px/1.7 sans-serif;max-width:1200px;margin:auto}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>D1：区域内L×u，跨区[1+(w−1)/n]×L×u（离线，无问答）</h1><p>上游、锚点与总预算不变。不是正式问答成绩；不得用旧答案给D1计分。</p>']
    for q in CASES:
        a=read_json(SOURCE/'combined_results/v1'/f'{q}.json');r=records[q]
        render(out/'boards'/f'{q}.png',r['selected_frames'],a['selection']['video'],processor,q+' D1 offline; NO QA',{f['source_pts'] for f in r['selected_frames']})
        page.append(f'<details open id="{q}"><summary>{q}：相对soft换{r["changed_vs_soft"]}帧</summary><pre>{html.escape(a["question"])}</pre><p>v1配额{r["budgets_v1"]}；soft配额{r["budgets_soft"]}；D1配额{r["trace"]["region_budgets"]}</p>')
        for v in ('v1','soft'):
            page.append(f'<h3>{v}正式历史输入</h3><img loading="lazy" src="{os.path.relpath(SOURCE/"combined_boards"/v/(q+".png"),out)}">')
        page.append(f'<h3>D1诊断输入，未调用问答</h3><a href="records/{q}.json">全部提议/得分/预算</a><img loading="lazy" src="boards/{q}.png"></details>')
    (out/'index.html').write_text(''.join(page))
    # 步骤3：只汇总输入变化，不把任何历史正确率移植为D1成绩。
    summary=dict(questions=200,original_soft_replays_passed=200,new_model_calls=0,qa_status='not_run',
        changed_vs_soft=sum(r['changed_vs_soft']>0 for r in records.values()),changed_slots_vs_soft=sum(r['changed_vs_soft'] for r in records.values()),
        changed_vs_v1=sum(r['changed_vs_v1']>0 for r in records.values()),changed_slots_vs_v1=sum(r['changed_vs_v1'] for r in records.values()),
        cases={q:{k:records[q][k] for k in ('selected_indices','changed_vs_soft','changed_vs_v1')} for q in CASES},
        elapsed_seconds=time.perf_counter()-started,source_hashes=hashes,
        concentration={label:sum(max(r['trace']['region_budgets'].values())>=12 for r in records.values() if (label=='all' or len(r['trace']['region_budgets'])>1)) for label in ('all','multi_region')})
    for name,digest in hashes.items():assert sha256(ROOT/name)==digest,name
    durable(out/'summary.json',summary);durable(out/'status.json',dict(status='computed_awaiting_review',completed=200,new_model_calls=0))
    print({k:v for k,v in summary.items() if k!='source_hashes'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args();main(args.run_id)
