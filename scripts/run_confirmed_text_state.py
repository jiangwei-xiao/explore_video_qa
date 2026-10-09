"""exp30单项修补：同一OCR银行和配额，只改文字状态的重复观测确认。"""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive
from videoqa_audit.framework_contraction import locations,BASE
from videoqa_audit.text_state_experiments import PILOT,TARGETS
from videoqa_methods.text_state_confirmation import confirm_text_states
from videoqa_methods.text_state_redundancy import select_text_state
from videoqa_budget32.frames import validate_frames

METHOD='RD-1.2-exp30'
SOURCE=ROOT/'outputs/methods/rd12_exp29_text_state_20261009_r3'


def prepare(root,population='pilot'):
    """接口：同源缓存→确认词→固定配额重选；新增OCR/BLIP/QA均0。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);source_spec=read_json(SOURCE/'protocol.json')
    ids=list(PILOT) if population=='pilot' else source_spec['development_ids']
    files=('src/videoqa_methods/text_state_confirmation.py','src/videoqa_methods/text_state_redundancy.py','scripts/run_confirmed_text_state.py')
    spec=dict(method=METHOD,parent='RD-1.2-exp07',ocr_source=str(SOURCE),source_protocol_sha256=sha256(SOURCE/'protocol.json'),
              files={n:sha256(ROOT/n) for n in files},pilot_ids=list(PILOT),development_ids=source_spec['development_ids'],
              rule='Only two unique coarse observations confirm OCR tokens; original visual fallback for unconfirmed text',
              new_auxiliary_forwards=0,real_qa_limit=400)
    with exclusive(root/'prepare.lock'):
        p=root/'protocol.json'
        if p.exists():
            if read_json(p)!=spec:raise ValueError('Confirmed-state protocol changed')
        else:durable(p,spec)
        exported={}
        # 步骤1：原OCR银行中的粗前缀作为共同依据，K8不借用M32额外细帧确认。
        for q in ids:
            source=SOURCE/'ocr_banks'/(q+'.json');bank=read_json(source);pool32=read_json(BASE/'pools'/(q+'.json'))
            if bank['pool_sha256']!=sha256(BASE/'pools'/(q+'.json')):raise ValueError('Frozen OCR source pool differs')
            for path,h in bank['calls'].items():
                if sha256(path)!=h:raise ValueError('Raw OCR result changed')
            states,confirmation=confirm_text_states(pool32['candidates'],pool32['initial_count'],bank['states'])
            for profile,k in [('qwen32',32),('llava8',8)]:
                pp,fp,tp,sp=locations(q,k);pool=read_json(pp);f=np.load(fp,allow_pickle=False)
                mapping=list(range(len(f))) if k==32 else pool['original_candidate_indices']
                if [[c['source_frame_index'],c['source_pts']] for c in pool['candidates']]!=[bank['source_keys'][i] for i in mapping]:raise ValueError('Confirmed source mapping differs')
                parent=read_json(tp);parent=parent['exp07'] if k==8 else parent
                words=[states[i] for i in mapping];nf=f if k==32 else f[mapping]
                selected,trace=select_text_state(pool,nf,parent,words,k)
                selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in selected]);validate_frames(selection,k)
                # 步骤2：无词控制必须与原exp07同源选择一致，不在修补中改视觉或预算。
                control,_=select_text_state(pool,nf,parent,[[] for _ in words],k)
                key=lambda s:[(x['source_frame_index'],x['source_pts']) for x in s['selected_frames']]
                if key(dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in control]))!=key(read_json(sp)['selection']):raise ValueError('Confirmed-state no-text control differs')
                trp=root/'traces'/profile/(q+'.json');durable(trp,trace)
                value=dict(question_id=q,profile=profile,method=METHOD,selection=selection,selected_indices=selected,
                    actual_frame_count=len(selected),states=words,raw_states=[bank['states'][i] for i in mapping],
                    confirmation=[dict(confirmation[i],candidate_index=j) for j,i in enumerate(mapping)],
                    trace_file=str(trp.relative_to(root)),trace_sha256=sha256(trp),source_pool=str(pp),source_features=str(fp),
                    ocr_source_root=str(SOURCE),no_text_control_exact=True,parent_quotas_fixed=True,
                    unchanged_selected_sources=key(selection)==key(read_json(sp)['selection']),
                    sources={str(x):sha256(x) for x in (pp,fp,tp,sp,source)},new_auxiliary_forwards=0,new_qa_calls=0)
                dest=root/'offline'/profile/(q+'.json')
                if dest.exists() and read_json(dest)!=value:raise ValueError('Confirmed selection changed on recovery')
                durable(dest,value);exported[str(dest)]=sha256(dest)
        durable(root/(population+'_selection_summary.json'),dict(status='passed',questions=len(ids),selections=2*len(ids),files=exported,
            new_auxiliary_forwards=0,new_qa_calls=0))


def main():
    """接口：只提供固定确认机制，不暴露时间窗口、频次扫描或逐题路由。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--population',choices=('pilot','development'),default='pilot');a=p.parse_args()
    if not a.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    prepare(ROOT/'outputs/methods'/a.run_id,a.population)


if __name__=='__main__':main()
