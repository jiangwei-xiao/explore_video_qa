"""固定面板的重复竞争诊断；不改变选择顺序、不生成新输入或答案。"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
from pathlib import Path
import argparse
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE,FULL
from videoqa_methods.density_selection import source_time,frame_key

CURRENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
PANEL=ROOT/'configs/analysis/evidence_state_panel_20261003.json'
THRESHOLD=.95


def group_runs(ids,features,candidates,video):
    """接口：时间相邻且与组内所有帧相似度至少0.95才合并，避免传递链越合越远。"""
    # 步骤1：按精确源时间排序，粗细来源平等；大于一秒的空隙不可跨过。
    groups=[]
    for i in sorted(ids,key=lambda j:frame_key(candidates[j])):
        append=bool(groups) and source_time(candidates[i],video)-source_time(candidates[groups[-1][-1]],video)<=1
        if append:
            append=bool(np.all(features[groups[-1]]@features[i]>=THRESHOLD))
        if append:groups[-1].append(i)
        else:groups.append([i])
    return groups


def main(run):
    """接口：压缩原前缀的重复组仅作名次诊断，组命中不等于见证帧命中。"""
    sources={str(Path(__file__)):sha256(__file__)}
    def read(path):
        sources[str(path)]=sha256(path);return read_json(path)
    panel=read(PANEL);rows=[]
    # 步骤1：固定单阈值和12题，不按两题修复结果调相似度或时间跨度。
    for case in panel['cases']:
        q=case['question_id'];pool=read(BASE/'pools'/(q+'.json'));old=read(FULL/'traces'/(q+'.json'))
        new=read(CURRENT/'traces'/(q+'.json'));fp=BASE/'features'/(q+'.npy');sources[str(fp)]=sha256(fp)
        f=np.load(fp,allow_pickle=False);regions=[]
        for r in old['regions']:
            groups=group_runs(r['members'],f,pool['candidates'],pool['video'])
            owner={i:g for g,group in enumerate(groups) for i in group}
            order=list(dict.fromkeys(owner[i] for i in r['prefix']))
            k=new['region_budgets'][str(r['region_id'])]
            regions.append(dict(region_id=r['region_id'],groups=groups,quota=k,
                prefix=r['prefix'],prefix_group_order=order,
                duplicate_groups_in_actual_prefix=k-len({owner[i] for i in r['prefix'][:k]}),
                paths=[dict(candidate_index=i,original_position=r['prefix'].index(i)+1,
                    compressed_group_position=order.index(owner[i])+1,group_members=groups[owner[i]]) for i in r['prefix']]))
        rows.append(dict(question_id=q,regions=regions))
    # 步骤2：只追踪两题已定位见证，不把合并组的其他成员直接当替代证据。
    targets=[]
    for q,i in [('409-3',77),('409-3',80),('556-1',42)]:
        row=next(r for r in rows if r['question_id']==q)
        region=next(r for r in row['regions'] if any(i in g for g in r['groups']))
        group=next(g for g in region['groups'] if i in g);gid=region['groups'].index(group)
        order=region['prefix_group_order'];position=order.index(gid)+1 if gid in order else None
        targets.append(dict(question_id=q,candidate_index=i,group_members=group,
            original_position=region['prefix'].index(i)+1 if i in region['prefix'] else None,
            compressed_group_position=position,quota=region['quota'],
            group_potentially_reached=position is not None and position<=region['quota']))
    result=dict(sources=sources,threshold=THRESHOLD,max_neighbor_gap_seconds=1,rows=rows,targets=targets,
        summary=dict(questions=len(rows),regions=sum(len(r['regions']) for r in rows),
            duplicate_group_slots=sum(g['duplicate_groups_in_actual_prefix'] for r in rows for g in r['regions']),
            new_model_calls=0,new_selections=0),
        limitation='Engineering near-repeat diagnostic, not semantic segmentation; compressing frozen order is not rerunning a grouped selector')
    out=run/'analysis.json'
    if out.exists():
        if read_json(out)!=result:raise ValueError('Changed inputs or analysis')
    else:
        run.mkdir(parents=True,exist_ok=True);durable(out,result)
    print(result['summary'],flush=True);print(targets,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(Path(p.parse_args().run))
