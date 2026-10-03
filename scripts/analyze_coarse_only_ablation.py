"""开发200题补查消融离线准备；只复用原资产，不运行模型。"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.query_input_probe import BASE,FULL
from videoqa_methods.coarse_only_ablation import select_coarse_only
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
PARENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'


def main(run):
    """接口：所有200题统一移除细帧；不足32不凑帧，同预算子集单列。"""
    with exclusive(run/'run.lock'):
        ids=[r['question_id'] for r in read_json(DATA)['rows']];assert len(ids)==len(set(ids))==200
        sources={str(DATA):sha256(DATA)};inputs={}
        names=['scripts/analyze_coarse_only_ablation.py','src/videoqa_methods/coarse_only_ablation.py',
            'src/videoqa_methods/prefix_coverage_budget.py','src/videoqa_methods/relevance_coverage_budget.py',
            'src/videoqa_methods/weighted_seed_support.py','src/videoqa_methods/coarse_eligibility.py','src/videoqa_methods/density_selection.py']
        # 步骤1：选择前冻结全部来源；不读取任何旧答案决定保留或跳过题目。
        for name in names:
            p=ROOT/name;sources[str(p)]=sha256(p);dest=run/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha256(dest)==sources[str(p)]
            else:shutil.copy2(p,dest)
        for q in ids:
            paths=dict(pool=BASE/'pools'/(q+'.json'),features=BASE/'features'/(q+'.npy'),
                trace=FULL/'traces'/(q+'.json'),parent=PARENT/'selections/RD-1.2-exp12'/(q+'.json'))
            inputs[q]={k:str(p) for k,p in paths.items()}
            for p in paths.values():sources[str(p)]=sha256(p)
        spec=dict(question_ids=ids,sources=sources,inputs=inputs,parent_method='RD-1.2-exp12',new_model_calls=0,
            hypothesis='Only remove refinement frames; retain seeds/windows/coarse scores and features; recompute original prefixes and weighted coverage DP; budget upper bound32',
            primary_denominator=200,expected_common32_count=196)
        pp=run/'protocol.json'
        if pp.exists():assert read_json(pp)==spec
        else:durable(pp,spec)
        records=[]
        # 步骤2：每题原子落盘，断点仅复用协议相同的完整选择。
        for q in ids:
            dest=run/'questions'/(q+'.json')
            if dest.exists():
                record=read_json(dest);assert record['protocol_sha256']==sha256(pp)
            else:
                p=inputs[q];pool=read_json(p['pool']);f=np.load(p['features'],allow_pickle=False);parent=read_json(p['trace'])
                selected,trace=select_coarse_only(pool,f,parent)
                old=read_json(p['parent'])['selected_indices']
                record=dict(question_id=q,protocol_sha256=sha256(pp),old_indices=old,selected_indices=selected,
                    selected_frames=[pool['candidates'][i] for i in selected],trace=trace,new_model_calls=0,qa_prediction=None)
                durable(dest,record)
            records.append(record);durable(run/'status.json',dict(status='running',completed=len(records),total=200,pid=os.getpid(),utc=utc()))
        # 步骤3：精确数量闭合，帧数不足题提前披露，不静默改变预算上限。
        short={r['question_id']:len(r['selected_indices']) for r in records if len(r['selected_indices'])<32}
        assert short=={'143-2':31,'187-1':27,'096-2':21,'138-2':30}
        summary=dict(status='completed',question_count=200,short_pools=short,common32_count=196,
            changed_questions=[r['question_id'] for r in records if r['selected_indices']!=r['old_indices']],
            removed_selected_fine=sum(sum(i>=r['trace']['initial_count'] for i in r['old_indices']) for r in records),new_model_calls=0)
        if (run/'summary.json').exists():assert read_json(run/'summary.json')==summary
        else:durable(run/'summary.json',summary)
        durable(run/'status.json',dict(status='completed',completed=200,total=200,utc=utc()))
        print({k:v for k,v in summary.items() if k!='changed_questions'},'changed',len(summary['changed_questions']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(Path(p.parse_args().run))
