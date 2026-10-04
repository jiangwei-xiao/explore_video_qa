"""M64可选区域候选池CPU预检，先精确复现旧M32细帧映射，再冻结新元数据。"""
import argparse
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE
from videoqa_budget32.selection import INDEX,config
from videoqa_methods.optional_region_pool import map_optional_pool


def main(run):
    """接口：固定开发200题，旧粗特征只读、新细帧尚未计算；记录视觉调用上限。"""
    manifest=ROOT/'data/manifests/video_mme_development_200_20260920.json'
    source_files=[Path(__file__),manifest,ROOT/'src/videoqa_methods/optional_region_pool.py',
                  ROOT/'src/videoqa_methods/optional_region_selection.py',ROOT/'configs/retrieval_density_v1.json']
    sources={str(p):sha256(p) for p in source_files};records=[]
    # 步骤1：锁定池、特征、完整索引及源视频身份；不把元数据映射称为已完成解码。
    for row in read_json(manifest)['rows']:
        q=row['question_id'];pool_path=BASE/'pools'/(q+'.json');feature_path=BASE/'features'/(q+'.npy')
        index_path=INDEX/'uniform_records'/(row['video_id']+'.json');index=read_json(index_path)
        pts_path=INDEX/index['pts_file'];old=read_json(pool_path)
        for p in [pool_path,feature_path,index_path,pts_path]:sources[str(p)]=sha256(p)
        if sources[str(pts_path)]!=index['pts_sha256']:raise ValueError('索引哈希不符')
        for k in ['path','start_pts','time_base','duration_fraction']:
            if old['video'][k]!=index['video'][k]:raise ValueError('视频时钟身份不一致')
        video_path=Path(old['video']['path']);sources[str(video_path)]=sha256(video_path)
        if sources[str(video_path)]!=index['source_video_sha256']:raise ValueError('源视频改变')
        pts=np.load(pts_path,allow_pickle=False,mmap_mode='r')
        control=map_optional_pool(old,config(32,32),pts)
        if control['candidates']!=old['candidates'] or control['plan']!=old['plan']:
            raise ValueError('旧M32映射未精确复现: '+q)
        # 步骤2：新池含全部粗帧与新窗口内目标，重新计算全部细帧，不复用旧细帧特征。
        new=map_optional_pool(old,config(32,64),pts)
        features=np.load(feature_path,allow_pickle=False,mmap_mode='r')
        if len(features)!=len(old['candidates']):raise ValueError('特征与旧池错位')
        path=run/'pools'/(q+'.json')
        if path.exists():
            if read_json(path)!=new:raise ValueError('候选池恢复不符')
        else:durable(path,new)
        records.append(dict(question_id=q,video_id=row['video_id'],old_mapping_exact=True,pool_sha256=sha256(path),
            old_fine=old['new_candidate_count'],fine=new['new_candidate_count'],regions=len(new['plan']['regions']),
            visual_batches=(new['new_candidate_count']+15)//16,
            feature_bytes=len(new['candidates'])*features.shape[1]*4))
    # 步骤3：特征输出与检查点预留空间；未调用视觉模型或问答。
    if len(records)!=200:raise ValueError('开发分母变化')
    estimate=2*sum(r['feature_bytes'] for r in records)+256*1024**2
    free=shutil.disk_usage(run).free
    if free<1.5*estimate:raise ValueError('可用空间不足')
    summary=dict(sources=sources,records=records,questions=200,config=config(32,64),new_model_calls=0,
        total_fine=sum(r['fine'] for r in records),maximum_visual_batches=sum(r['visual_batches'] for r in records),
        estimated_new_bytes=estimate,limitation='候选元数据已准备；新细帧视觉特征和正式选择尚未完成')
    path=run/'summary.json'
    if path.exists():
        if read_json(path)!=summary:raise ValueError('预检恢复来源变化')
    else:durable(path,summary)
    print('200 exact M32 mappings; planned M64 fine frames',summary['total_fine'],'batches',summary['maximum_visual_batches'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();args.run.mkdir(parents=True,exist_ok=True);main(args.run)
