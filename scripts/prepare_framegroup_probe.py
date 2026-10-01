#!/usr/bin/env python3
"""两题人工构造帧组能力对照：精确PTS取原片，不改正式候选或问答结果。"""
import argparse
from fractions import Fraction
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc,rows
from videoqa_budget32.frames import FrameProvider
from videoqa_audit.query_input_probe import BASE
INDEX=ROOT/'outputs/analysis/lmms_bridge_videomme_20260923_r1'
STARTS={'395-2':{'anchor_context':'266.25','additional_clue':'268.25'},
        '409-3':{'anchor_context':'90.25','additional_clue':'77.25'}}


def map_targets(pts,video,targets):
    """接口：每个有理数目标映射到不早于它的首个源PTS，禁止平均FPS推算。"""
    records=[]
    for target in targets:
        absolute=Fraction(video['start_pts'])+target/Fraction(video['time_base']);threshold=-(-absolute.numerator//absolute.denominator)
        index=int(np.searchsorted(pts,threshold,side='left'))
        if index>=len(pts):raise ValueError('Window exceeds source video')
        p=int(pts[index]);seconds=(p-video['start_pts'])*Fraction(video['time_base'])
        records.append(dict(candidate_index=index,source_frame_index=index,source_pts=p,source_seconds=float(p*Fraction(video['time_base'])),timestamp_seconds=float(seconds),requested_seconds=float(target)))
    if len({r['source_pts'] for r in records})!=len(records):raise ValueError('Duplicate source frame in diagnostic clip')
    return records


def execute(run):
    """接口：先冻结四个窗口与全部原题选项，手工证据说明单列且不进入模型提示词。"""
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'assets.lock'):
        manifest=run/'assets_manifest.json'
        if manifest.exists():
            for p,h in read_json(manifest)['sources'].items():assert sha256(p)==h
            print('Existing assets verified',flush=True);return
        data={x['question_id']:x for x in rows()};sources={str(Path(__file__).resolve()):sha256(__file__)};cases={};total=0
        # 步骤1：已审计的完整源索引与原视频身份同时核对，诊断网格为固定2 FPS、10唯一帧。
        for q,windows in STARTS.items():
            row=data[q];poolpath=BASE/'pools'/(q+'.json');pool=read_json(poolpath);video=pool['video'];ip=INDEX/'uniform_records'/(str(row['video_id'])+'.json');index=read_json(ip);pp=INDEX/index['pts_file']
            assert sha256(pp)==index['pts_sha256'] and sha256(video['path'])==index['source_video_sha256']
            for p in (poolpath,ip,pp,Path(video['path'])):sources[str(p)]=sha256(p)
            pts=np.load(pp,allow_pickle=False);clips={name:map_targets(pts,video,[Fraction(start)+Fraction(k,2) for k in range(10)]) for name,start in windows.items()}
            union={r['source_pts']:r for group in clips.values() for r in group};frames=[union[p] for p in sorted(union)];images,hashes=FrameProvider(dict(video=video,selected_frames=frames)).decode();records={}
            # 步骤2：原RGB无文字叠加，图板仅供人审，模型读取独立PNG原帧。
            for ordinal,(record,im,h) in enumerate(zip(frames,images,hashes)):
                dest=run/'source_frames'/q/(str(ordinal)+'.png');dest.parent.mkdir(parents=True,exist_ok=True);im.save(dest);im.close();sources[str(dest)]=sha256(dest)
                records[str(record['source_pts'])]=dict(record,file=str(dest),png_sha256=sha256(dest),rgb_sha256=h,bank_index=ordinal)
            for name,group in clips.items():
                canvas=Image.new('RGB',(1600,420),'white');draw=ImageDraw.Draw(canvas)
                for j,record in enumerate(group):
                    with Image.open(records[str(record['source_pts'])]['file']) as im:
                        im=im.convert('RGB');im.thumbnail((316,176));x,y=j%5*320,j//5*210;canvas.paste(im,(x,y));draw.text((x,y+180),f'{record["timestamp_seconds"]:.3f}s',fill='black')
                dest=run/'review'/(q+'_'+name+'.jpg');dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest,quality=94)
            cases[q]=dict(question=row['question'],options=row['options'],video_id=row['video_id'],video=video,frames=records,
                clips={name:[str(x['source_pts']) for x in group] for name,group in clips.items()});total+=len(records)
        # 步骤3：主对照是整组/逐帧聚合，逆序仅测顺序敏感性，不自动视作错误证据。
        assert total==34
        durable(manifest,dict(status='completed',cases=cases,sources=sources,source_frames_per_clip=10,unique_frames=34,encoding_fps=2,
            real_embedding_forwards=52,synthetic_embedding_forwards=3,final_qa_calls=0,
            scope='Human-constructed capability probe, not automatic selected-frame results or an accuracy benchmark',utc=utc()))
        durable(run/'evaluation_notes.json',{'395-2':'两组共享拿红书附近帧，additional_clue增加272.750秒宣传册；并非已经完整呈现后续度假行动。',
            '409-3':'additional_clue包含77/80/81秒持盒、玫瑰及取枪附近状态；anchor_context为已经持枪的后续画面。',
            'limitations':'手工选窗用于能力诊断；题目标准答案不进入输入。逆序不一概判无关；不能将结果当新方法或全量准确率。'})
        print('Prepared',total,'unique frames, 4 clips, 52 real embedding forwards planned',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id)
    execute(ROOT/'outputs/analysis'/a.run_id)
