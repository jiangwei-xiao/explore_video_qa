"""exp31有界文字补充离线入口，不重识别、不扩大帧预算。"""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive
from videoqa_audit.framework_contraction import locations,BASE
from videoqa_audit.text_state_experiments import PILOT
from videoqa_methods.bounded_text_state import select_bounded_text_state
from videoqa_budget32.frames import validate_frames

SOURCE=ROOT/'outputs/methods/rd12_exp29_text_state_20261009_r3'
METHOD='RD-1.2-exp31'


def prepare(root,population):
    """接口：完整父输入→一处有界替换，弱文字信号不能全段连带改帧。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);source=read_json(SOURCE/'protocol.json')
    ids=list(PILOT) if population=='pilot' else source['development_ids']
    files=('src/videoqa_methods/bounded_text_state.py','src/videoqa_methods/text_state_redundancy.py','scripts/run_bounded_text_state.py')
    spec=dict(method=METHOD,parent='RD-1.2-exp07',ocr_source=str(SOURCE),source_protocol_sha256=sha256(SOURCE/'protocol.json'),
              files={n:sha256(ROOT/n) for n in files},pilot_ids=list(PILOT),development_ids=source['development_ids'],
              maximum_replacements=1,raw_ocr=True,real_qa_limit=400,
              limitation='Fixed parent prefix last slot is not guaranteed semantically redundant')
    with exclusive(root/'prepare.lock'):
        p=root/'protocol.json'
        if p.exists() and read_json(p)!=spec:raise ValueError('Bounded text protocol changed')
        if not p.exists():durable(p,spec)
        exports={}
        # 步骤1：同一冻结OCR资产映射两预算，不改帧池、问题或配额。
        for q in ids:
            op=SOURCE/'ocr_banks'/(q+'.json');bank=read_json(op)
            for p,h in bank['calls'].items():
                if sha256(p)!=h:raise ValueError('Raw OCR return changed')
            for profile,k in [('qwen32',32),('llava8',8)]:
                pp,fp,tp,sp=locations(q,k);pool=read_json(pp);f=np.load(fp,allow_pickle=False)
                mapping=list(range(len(f))) if k==32 else pool['original_candidate_indices']
                if [[c['source_frame_index'],c['source_pts']] for c in pool['candidates']]!=[bank['source_keys'][i] for i in mapping]:raise ValueError('Bounded source mapping differs')
                parent=read_json(tp);parent=parent['exp07'] if k==8 else parent
                words=[bank['states'][i] for i in mapping];nf=f if k==32 else f[mapping]
                selected,trace=select_bounded_text_state(pool,nf,parent,words,k)
                selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in selected]);validate_frames(selection,k)
                trp=root/'traces'/profile/(q+'.json');durable(trp,trace)
                value=dict(question_id=q,profile=profile,method=METHOD,selection=selection,selected_indices=selected,
                    actual_frame_count=len(selected),states=words,trace_file=str(trp.relative_to(root)),trace_sha256=sha256(trp),
                    ocr_source_root=str(SOURCE),source_pool=str(pp),source_features=str(fp),
                    sources={str(p):sha256(p) for p in (pp,fp,tp,sp,op)},parent_quotas_fixed=True,
                    changed=trace['winner'] is not None,new_auxiliary_forwards=0,new_qa_calls=0)
                dest=root/'offline'/profile/(q+'.json')
                if dest.exists() and read_json(dest)!=value:raise ValueError('Bounded selection changed on recovery')
                durable(dest,value);exports[str(dest)]=sha256(dest)
        # 步骤2：保存完整比较路径，未出现正增量的输入精确保持父结果。
        durable(root/(population+'_selection_summary.json'),dict(status='passed',questions=len(ids),selections=2*len(ids),
            files=exports,new_auxiliary_forwards=0,new_qa_calls=0))


def main():
    """接口：固定最小一帧干预，无比例、融合系数或逐题路由参数。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--population',choices=('pilot','development'),default='pilot');a=p.parse_args()
    if not a.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    prepare(ROOT/'outputs/methods'/a.run_id,a.population)


if __name__=='__main__':main()
