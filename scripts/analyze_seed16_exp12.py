"""M16/K32的exp12单一配置离线准备，M32控制必须精确复现原exp12。"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys
import argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.query_input_probe import BASE
from videoqa_methods.seed_budget_selection import select_seed_budget
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
POOLS=ROOT/'outputs/analysis/seed16_pool200_20261003_r1'
PARENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'


def main(run):
    """接口：严格区分新旧候选ID；对外选择仍导出原M32池ID以兼容既有精确读取。"""
    with exclusive(run/'run.lock'):
        preflight=read_json(POOLS/'preflight.json');assert preflight['status']=='passed'
        for p,h in preflight['sources'].items():assert sha256(p)==h
        ids=[r['question_id'] for r in read_json(DATA)['rows']];assert len(ids)==200
        sources={str(DATA):sha256(DATA),str(POOLS/'preflight.json'):sha256(POOLS/'preflight.json')};inputs={}
        for name in ['scripts/analyze_seed16_exp12.py','src/videoqa_methods/seed_budget_selection.py',
            'src/videoqa_methods/coarse_only_ablation.py','src/videoqa_methods/prefix_coverage_budget.py',
            'src/videoqa_methods/relevance_coverage_budget.py','src/videoqa_methods/weighted_seed_support.py',
            'src/videoqa_methods/coarse_eligibility.py','src/videoqa_methods/density_selection.py']:
            sources[str(ROOT/name)]=sha256(ROOT/name)
        # 步骤1：所有来源先固定，输出源ID与新池内部ID分开，绝不直接跨池比较编号。
        for q in ids:
            paths=dict(pool=BASE/'pools'/(q+'.json'),features=BASE/'features'/(q+'.npy'),new_pool=POOLS/'pools'/(q+'.json'),
                parent=PARENT/'selections/RD-1.2-exp12'/(q+'.json'))
            inputs[q]={k:str(v) for k,v in paths.items()}
            for p in paths.values():sources[str(p)]=sha256(p)
        spec=dict(question_ids=ids,inputs=inputs,sources=sources,new_model_calls=0,
            hypothesis='Only seed budget M32 to M16, K32 fixed; rebuild cores/windows/refinement membership/support/references/density/quotas by unchanged exp12 rules')
        pp=run/'protocol.json'
        if pp.exists():assert read_json(pp)==spec
        else:durable(pp,spec)
        records=[]
        # 步骤2：M32控制重建前缀及配额，与冻结最终源帧顺序完全相同才继续。
        for q in ids:
            dest=run/'questions'/(q+'.json')
            if dest.exists():
                r=read_json(dest);assert r['protocol_sha256']==sha256(pp)
            else:
                p=inputs[q];old=read_json(p['pool']);f=np.load(p['features'],allow_pickle=False);parent=read_json(p['parent'])
                control,_=select_seed_budget(old,f);assert control==parent['selected_indices'],q
                new=read_json(p['new_pool']);mapping=new['original_candidate_indices'];nf=f[mapping]
                selected,trace=select_seed_budget(new,nf)
                original_indices=[mapping[i] for i in selected]
                assert [new['candidates'][i]['source_pts'] for i in selected]==[old['candidates'][i]['source_pts'] for i in original_indices]
                r=dict(question_id=q,protocol_sha256=sha256(pp),old_indices=control,selected_indices=original_indices,
                    selected_frames=[old['candidates'][i] for i in original_indices],new_pool_selected_indices=selected,
                    trace=trace,new_model_calls=0,qa_prediction=None,source_id_scope='selected_indices are original M32 IDs; trace IDs belong to rebuilt M16 pool')
                durable(dest,r)
            records.append(r);durable(run/'status.json',dict(status='running',completed=len(records),total=200,pid=os.getpid(),utc=utc()))
        # 步骤3：完成后只报告结构变化，不能附用旧答案当新配置成绩。
        summary=dict(status='completed',question_count=200,m32_control_exact=200,
            changed_questions=[r['question_id'] for r in records if r['old_indices']!=r['selected_indices']],
            actual_frame_counts=sorted(set(len(r['selected_indices']) for r in records)),new_model_calls=0)
        if (run/'summary.json').exists():assert read_json(run/'summary.json')==summary
        else:durable(run/'summary.json',summary)
        durable(run/'status.json',dict(status='completed',completed=200,utc=utc()))
        print({k:v for k,v in summary.items() if k!='changed_questions'},'changed',len(summary['changed_questions']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    try:main(Path(a.run))
    except BaseException as exc:
        durable(Path(a.run)/'status.json',dict(status='paused',error=repr(exc),utc=utc()));raise
