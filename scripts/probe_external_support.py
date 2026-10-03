"""原窗外少量分散粗帧支持探针，固定原配额和窗口，零模型调用。"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
from fractions import Fraction
from pathlib import Path
import argparse
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive
from videoqa_audit.query_input_probe import BASE,FULL
from videoqa_audit.temporal_support_probe import local_prefix
from videoqa_methods.weighted_seed_support import weighted_support
from videoqa_methods.density_selection import source_time,frame_key
PARENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
PANEL=ROOT/'configs/analysis/evidence_state_panel_20261003.json'


def propose(pool):
    """接口：只接收粗候选、原分数和窗口；最多4个正分候选，真实时间相距必须大于4秒。"""
    c=pool['candidates'];scores=pool['initial_scores'];video=pool['video'];windows=pool['plan']['regions']
    # 步骤1：窗口端点也算已覆盖，不把原定位种子重复提议为新支持参考。
    eligible=[i for i in range(pool['initial_count']) if scores[i]>0 and not any(
        Fraction(w['left_fraction'])<=source_time(c[i],video)<=Fraction(w['right_fraction']) for w in windows)]
    ordered=sorted(eligible,key=lambda i:(-scores[i],*frame_key(c[i])))
    chosen=[];suppressed=[]
    for i in ordered:
        suppressor=next((s for s in chosen if abs(source_time(c[i],video)-source_time(c[s],video))<=4),None)
        if suppressor is not None:suppressed.append(dict(candidate_index=i,suppressor=suppressor));continue
        chosen.append(i)
        if len(chosen)==4:break
    return chosen,suppressed


def main(run):
    """接口：固定12题统一规则，不选答案、不新增细帧或改变保护集合。"""
    with exclusive(run/'run.lock'):
        sources={str(Path(__file__)):sha256(__file__),str(PANEL):sha256(PANEL)}
        for name in ['src/videoqa_audit/temporal_support_probe.py','src/videoqa_methods/weighted_seed_support.py','src/videoqa_methods/density_selection.py']:
            sources[str(ROOT/name)]=sha256(ROOT/name)
        def read(p):sources[str(p)]=sha256(p);return read_json(p)
        ids=[x['question_id'] for x in read(PANEL)['cases']]
        spec=dict(question_ids=ids,max_support_references=4,suppression_seconds=4,original_windows_unchanged=True,
            quotas='frozen exp12 M32/K32',new_model_calls=0,
            rule='Positive original ITM, outside closed original windows; descending score and stable time tie-break; supplement support only')
        pp=run/'protocol.json'
        if pp.exists():assert read_json(pp)==spec
        else:durable(pp,spec)
        records=[]
        # 步骤2：仅复用通用局部前缀工具，不调用时间支持函数；原支持控制必须重放。
        for q in ids:
            pool=read(BASE/'pools'/(q+'.json'));old=read(FULL/'traces'/(q+'.json'));parent=read(PARENT/'traces'/(q+'.json'))
            fp=BASE/'features'/(q+'.npy');sources[str(fp)]=sha256(fp);f=np.load(fp,allow_pickle=False)
            proposals,suppressed=propose(pool);c=pool['candidates'];keys=[frame_key(x) for x in c]
            plans={r['region_id']:r for r in pool['plan']['regions']};selected=[];before=[];traces=[]
            for region in old['regions']:
                rid=region['region_id'];members=region['members'];seeds=plans[rid]['seed_indices']
                original,_,peak0=weighted_support(f,members,seeds,pool['initial_scores'])
                control,_=local_prefix(f,members,region['prefix'][0],original,keys,min(32,len(members)))
                assert control==region['prefix'],q
                additions=[i for i in proposals if i in members]
                support,weights,peak1=weighted_support(f,members,seeds+additions,pool['initial_scores'])
                assert peak0==peak1 and np.all(support+1e-7>=original)
                prefix,steps=local_prefix(f,members,region['prefix'][0],support,keys,min(32,len(members)))
                quota=parent['region_budgets'][str(rid)];selected.extend(prefix[:quota]);before.extend(control[:quota])
                traces.append(dict(region_id=rid,quota=quota,extra_support=additions,prefix=prefix,steps=steps))
            selected.sort(key=lambda i:keys[i]);before.sort(key=lambda i:keys[i])
            assert len(selected)==len(set(selected))==len(before) and set(old['protected_indices'])<=set(selected)
            records.append(dict(question_id=q,proposals=proposals,proposal_frames=[c[i] for i in proposals],suppressed=suppressed,
                selected_indices=selected,old_indices=before,regions=traces,added=sorted(set(selected)-set(before)),
                removed=sorted(set(before)-set(selected)),selection=dict(video=pool['video'],selected_frames=[c[i] for i in selected])))
        # 步骤3：结果仅为离线选择，不能借用旧预测或把支持单调增加当作信息改善。
        result=dict(protocol=spec,sources=sources,records=records,new_model_calls=0,new_answers=0,
            changed_questions=sum(r['selected_indices']!=r['old_indices'] for r in records))
        out=run/'selections.json'
        if out.exists():assert read_json(out)==result
        else:durable(out,result)
        for r in records:print(r['question_id'],'proposals',r['proposals'],'added',r['added'],'removed',r['removed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(Path(p.parse_args().run))
