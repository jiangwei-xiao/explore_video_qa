"""独立完成审计：从真实原始返回和冻结标签重算，不触发任何模型生成。"""
import argparse
import math
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def independent_pair(data,left,right):
    """接口：独立重建四格及分层视频簇区间；只读取已完成原始返回的计分。"""
    import numpy as np
    rng=np.random.default_rng(2027);numerator=np.zeros(10000);denominator=np.zeros(10000)
    cells={k:[] for k in ('both_correct','left_only','right_only','both_wrong')}
    # 步骤1先按冻结题序归入四格；步骤2独立按时长/视频重抽，不使用报告的对错字段。
    for row in data:
        q=row['question_id'];a,b=left[q],right[q]
        key='both_correct' if a and b else 'left_only' if a else 'right_only' if b else 'both_wrong'
        cells[key].append(q)
    for label in ('short','medium','long'):
        clusters={}
        for row in data:
            if row['stratum']==label:
                q=row['question_id'];clusters.setdefault(row['video_id'],[]).append(int(left[q])-int(right[q]))
        if not clusters:continue
        values=list(clusters.values());sums=np.asarray([sum(x) for x in values]);sizes=np.asarray([len(x) for x in values])
        sampled=rng.integers(len(values),size=(10000,len(values)))
        numerator+=sums[sampled].sum(axis=1);denominator+=sizes[sampled].sum(axis=1)
    return dict(n=len(data),cells=cells,difference_pp=100*(len(cells['left_only'])-len(cells['right_only']))/len(data),
        stratified_video_cluster_bootstrap_95=np.percentile(100*numerator/denominator,[2.5,97.5]).tolist())


def independent_wilson(correct,n):
    """接口：从正确数独立复算Wilson区间，单位为0–1。"""
    z=1.959963984540054;p=correct/n;z2=z*z
    center=(p+z2/(2*n))/(1+z2/n)
    width=z*math.sqrt(p*(1-p)/n+z2/(4*n*n))/(1+z2/n)
    return [max(0.,center-width),min(1.,center+width)]


def audit(root,profile):
    """接口：逐后端读取官方Tokenizer、任务解析与物理返回；不加载视觉或语言模型。"""
    import numpy as np
    from transformers import AutoTokenizer
    from videoqa_audit import selection_structure as method
    from videoqa_audit import selection_structure_run as study
    from videoqa_audit import selection_structure_runtime as runtime
    from videoqa_lmms_bridge.common import load_task,task_doc
    root=Path(root);run=root/'qa'/profile
    if not (run/'completion_audit.json').is_file():raise ValueError('Profile has not completed')
    if not (root/'summary.json').is_file():raise ValueError('Global report must be complete before final independent audit')
    spec=study.read(run/'protocol.json');data=study.development_rows();correct={};sources={};returns=0
    inventory=study.read(ROOT/'configs/local_model_inventory.json')
    path=inventory['models']['llava']['path'] if profile=='llava8' else '/home/models/Qwen/Qwen2.5-VL-7B-Instruct'
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    for m in (method.PARENT,*method.CONTROLS):
        correct[m]={}
        for row in data:
            q=row['question_id'];result_path=run/'results'/m/(q+'.json');value=study.read(result_path)
            item=study.read(run/'inputs'/(q+'.json'))['items'][m]
            if value['selection']!=item['selection'] or value['rgb_sha256']!=item['rgb_sha256']:
                raise ValueError('Result/input source identity differs')
            bundle=runtime.physical_sources(result_path,value);raws={}
            for p,h in bundle.items():
                if '/raw_generations/' in p:raws.setdefault(h,study.read(p))
            matching=[v for v in raws.values() if v['raw_generated_text'].strip()==value['answer']['raw_output']
                      and v['observed']['generated_token_ids']==value['answer']['observed']['generated_token_ids']]
            if not matching:raise ValueError('No matching confirmed original return')
            for raw in matching:
                if raw['generation_state']!=dict(started=True,returned=True):raise ValueError('Uncertain raw return')
                tokens=raw['observed']['generated_token_ids']
                if not 1<=len(tokens)<=16 or tokenizer.decode(tokens,skip_special_tokens=True).strip()!=raw['raw_generated_text'].strip():
                    raise ValueError('Original token/text decode differs')
            metric=load_task().videomme_process_results(task_doc(row),[matching[0]['raw_generated_text'].strip()])['videomme_perception_score']
            hit=metric['pred_answer']=='ABCD'[row['answer_index']]
            if hit!=value['correct'] or metric!=value['answer']['framework_metric']:raise ValueError('Independent raw correctness differs')
            correct[m][q]=hit;returns+=1;sources[str(result_path)]=study.sha(result_path);sources.update(bundle)
    value=dict(status='passed',profile=profile,method_results=returns,correct={m:sum(v.values()) for m,v in correct.items()},
        source_hashes=sources,auditor_sha256=study.sha(Path(__file__)),new_model_calls=0)
    # 步骤3全局报告已生成时，再独立核对全部六项配对及总体Wilson；不修改原报告。
    if (root/'summary.json').exists():
        summary=study.read(root/'summary.json');pairs={}
        for a,b in [(method.PARENT,m) for m in method.CONTROLS]+[(method.GLOBAL,method.WFS)]:
            pair=independent_pair(data,correct[a],correct[b]);saved=summary['comparisons'][profile][a+'__minus__'+b]
            if pair['cells']!=saved['cells'] or abs(pair['difference_pp']-saved['difference_pp'])>1e-12 or not np.allclose(
                pair['stratified_video_cluster_bootstrap_95'],saved['stratified_video_cluster_bootstrap_95'],rtol=0,atol=1e-12):
                raise ValueError('Independent paired bootstrap differs')
            pairs[a+'__minus__'+b]=pair
        for m,bank in correct.items():
            saved=summary['profiles'][profile]['metrics'][m]['development200']
            if saved['correct']!=sum(bank.values()) or not np.allclose(independent_wilson(sum(bank.values()),len(data)),saved['wilson_95'],rtol=0,atol=1e-12):
                raise ValueError('Independent Wilson/count differs')
        value['comparisons']=pairs;value['summary_sha256']=study.sha(root/'summary.json')
    study.freeze(root/'independent_audit'/(profile+'.json'),value)
    return {k:v for k,v in value.items() if k not in {'source_hashes','comparisons'}}


def main():
    """接口：只允许两个固定配置的完成审计，不提供生成或选择选项。"""
    import json
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True)
    parser.add_argument('--profile',required=True,choices=['llava8','qwen32']);args=parser.parse_args()
    print(json.dumps(audit(Path(args.run_dir),args.profile),ensure_ascii=False))


if __name__=='__main__':main()
