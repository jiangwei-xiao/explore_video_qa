"""核验WFS-SB公开32帧清单与本地冻结Video-MME的对应，不运行模型。"""
from fractions import Fraction
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable
from videoqa_budget32.frames import validate_frames


def main():
    """接口：逐字段及真实PTS索引核验2700题，输出独立参考映射，不导入作者答案计分。"""
    run=ROOT/'outputs/analysis/wfs_release_intake_20261004_r1'
    public=run/'reproduce_videomme_f32.json'
    manifest=ROOT/'data/manifests/video_mme_full_2700.json'
    index=ROOT/'outputs/analysis/lmms_bridge_videomme_20260923_r1'
    rows=read_json(public);local={r['question_id']:r for r in read_json(manifest)['rows']}
    sources={str(p):sha256(p) for p in [public,manifest,Path(__file__)]}
    # 步骤1：所有原始字段必须一致，外部JSON仅作为数据，不执行其内容。
    if len(rows)!=2700 or len({r['question_id'] for r in rows})!=2700 or set(local)!={r['question_id'] for r in rows}:
        raise ValueError('Released question population differs')
    cache={};exports={}
    for row in rows:
        q=row['question_id'];baseline=local[q]
        metadata={k:v for k,v in row.items() if k!='keyframe_indices'}
        if metadata!=baseline['original']:
            raise ValueError(f'Original annotation mismatch: {q}')
        vid=row['videoID']
        # 步骤2：只用完整解码PTS索引映射源编号，不用平均FPS换算。
        if vid not in cache:
            mp=index/'uniform_records'/(vid+'.json');m=read_json(mp);pp=index/m['pts_file']
            if sha256(pp)!=m['pts_sha256']:raise ValueError('PTS identity changed')
            pts=np.load(pp,allow_pickle=False)
            if len(pts)!=m['source_frame_count'] or np.any(np.diff(pts)<=0):raise ValueError('Invalid complete PTS index')
            sources.update({str(mp):sha256(mp),str(pp):m['pts_sha256']});cache[vid]=(m,pts)
        m,pts=cache[vid];ids=row['keyframe_indices']
        if len(ids)!=32 or len(set(ids))!=32 or ids!=sorted(ids) or any(type(i)!=int or not 0<=i<len(pts) for i in ids):
            raise ValueError(f'Released source indices invalid: {q}')
        video=m['video'];tb=Fraction(video['time_base']);frames=[]
        for ordinal,i in enumerate(ids):
            p=int(pts[i]);time=(p-video['start_pts'])*tb
            frames.append(dict(candidate_index=ordinal,source_frame_index=i,source_pts=p,
                               source_seconds=float(p*tb),timestamp_seconds=float(time),requested_seconds=float(time)))
        selection=dict(video=video,selected_frames=frames);validate_frames(selection)
        # 步骤3：只保存选帧来源，不包含标准答案/预测；candidate_index仅为清单序号。
        record=dict(question_id=q,video_id=vid,selection=selection,source_video_sha256=m['source_video_sha256'],
            released_source_sha256=sha256(public),candidate_index_domain='released_selection_ordinal',
            requested_time_domain='mapped_actual_source_time_not_original_sampling_target',
            limitation='作者未提供源视频字节哈希；元数据和本地源范围一致不证明双方视频字节完全相同')
        dest=run/'mapped_selections'/(q+'.json')
        if dest.exists():
            if read_json(dest)!=record:raise ValueError('Mapped reference changed')
        else:durable(dest,record)
        exports[str(dest)]=sha256(dest)
    if len(cache)!=900:raise ValueError('Expected 900 source videos')
    result=dict(status='passed',questions=2700,videos=900,frames_per_question=32,
        annotation_mismatches=0,invalid_source_indices=0,duplicate_indices=0,
        source_url='https://raw.githubusercontent.com/MAC-AutoML/WFS-SB/a424fc4528ecbe57edc93413826a2f2b8bb2c203/datasets/videomme/keyframe_dir/reproduce_videomme_f32.json',
        sources=sources,exports=exports,new_model_calls=0,new_answers=0,
        pending='RGB decode checks and explicit evaluation protocol; not a completed paper reproduction')
    path=run/'audit.json'
    if path.exists():
        if read_json(path)!=result:raise ValueError('Audit fingerprint mismatch')
    else:durable(path,result)
    print('2700 annotations, 900 PTS indexes and 86400 released source positions verified; QA 0',flush=True)


if __name__=='__main__':main()
