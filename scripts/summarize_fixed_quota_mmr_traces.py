"""只读补充400份源帧替换及Qwen时间patch变化，不重新求解选择。"""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def patch_pairs(frames):
    """接口：按当前官方两帧时间patch记录源身份，奇数末帧只作内部复制。"""
    keys=[[int(f['source_frame_index']),int(f['source_pts'])] for f in frames]
    if len(keys)%2:keys.append(keys[-1])
    return [keys[i:i+2] for i in range(0,len(keys),2)]


def summarize(root):
    """接口：固定父区域/配额验收并导出替换清单；不加载模型、特征或视频。"""
    from videoqa_audit import fixed_quota_local_mmr as m
    from videoqa_audit import fixed_quota_local_mmr_run as s
    root=Path(root);records={};counts={};source_hashes={}
    # 步骤1逐题验证实际配额和源顺序，保留全部改变/未改变，不按对错筛选。
    for profile in s.PROFILES:
        records[profile]={};counter=Counter()
        for row in s.development_rows():
            q=row['question_id'];tp=root/'traces'/profile/m.CONTROL/(q+'.json')
            pp=s.old_path(profile,'traces',m.PARENT,q)
            hp=root/'selections'/profile/m.CONTROL/(q+'.json')
            op=s.old_path(profile,'selections',m.PARENT,q)
            h,p=s.read(tp),s.read(pp);hs,ps=s.read(hp)['selection'],s.read(op)['selection']
            if h['region_budgets']!=p['region_budgets'] or h['dp_calls']!=0:
                raise ValueError('Actual quota intervention scope differs')
            before=set(x['candidate_index'] for x in ps['selected_frames']);after=set(x['candidate_index'] for x in hs['selected_frames'])
            region_records=[]
            for region in h['regions']:
                rid=region['region_id'];old=next(x for x in p['coarse_prefix_trace']['regions'] if str(x['region_id'])==rid)
                if (set(region['members'])!=set(old['members']) or region['anchor']!=old['prefix'][0]
                        or len(region['selection_order'])!=p['region_budgets'][rid]):
                    raise ValueError('Region, anchor, membership or quota changed')
                region_records.append(dict(region_id=rid,quota=region['quota'],anchor=region['anchor'],
                    old_selected=old['prefix'][:region['quota']],new_selected=region['selection_order']))
                counter['regions']+=1
            changed=s.source_identity(hs)!=s.source_identity(ps)
            record=dict(question_id=q,video_id=row['video_id'],stratum=row['stratum'],input_sources_changed=changed,
                removed=[f for f in ps['selected_frames'] if f['candidate_index'] not in after],
                added=[f for f in hs['selected_frames'] if f['candidate_index'] not in before],
                retained=len(before&after),frame_count=len(after),regions=region_records)
            if profile=='qwen32':
                a,b=patch_pairs(ps['selected_frames']),patch_pairs(hs['selected_frames'])
                record['time_patch_pairs']=dict(parent=a,control=b,changed_positions=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y],
                    limitation='Descriptive encoding-partner change, not isolated semantic causality')
                counter['changed_time_patch_positions']+=sum(x!=y for x,y in zip(a,b))
            records[profile][q]=record;counter['questions']+=1;counter['changed_inputs']+=int(changed)
            counter['removed_frames']+=len(before-after)
            source_hashes.update({str(x):s.sha(x) for x in (tp,pp,hp,op)})
        counts[profile]=dict(counter)
    # 步骤2一次发布独立描述表，恢复不重新选择或改写原轨迹。
    value=dict(status='passed',profiles=counts,records=records,source_hashes=source_hashes,
        new_model_calls=0,new_selection_calculations=0,script_sha256=s.sha(Path(__file__)))
    s.freeze(root/'posthoc_structure/source_replacements_and_time_patches.json',value)
    return dict(status='passed',profiles=counts,new_model_calls=0,new_selection_calculations=0)


def main():
    """接口：只接受已冻结运行目录，不开放方法、参数或题目筛选。"""
    import json
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);args=parser.parse_args()
    root=Path(args.run_dir).resolve()
    if not root.is_relative_to((ROOT/'outputs/diagnostics').resolve()):parser.error('Unexpected run root')
    print(json.dumps(summarize(root),ensure_ascii=False))


if __name__=='__main__':main()
