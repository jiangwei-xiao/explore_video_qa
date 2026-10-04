"""公开WFS源帧的独立顺序解码核验；不运行模型或变更正式输入。"""
import os
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_budget32.frames import FrameProvider

SOURCE=ROOT/'outputs/analysis/wfs_release_intake_20261004_r1'
RUN=ROOT/'outputs/analysis/wfs_release_rgb11_20261004_r1'
IDS=['001-1','001-2','001-3','002-1','002-2','133-1','134-1','409-3','556-1','383-2','819-1']


def verify_video(job):
    """接口：同视频共享顺序扫描，与逐帧seek路径逐RGB比较，不按平均FPS定位。"""
    import av
    vid,items=job
    dest=RUN/'videos'/(vid+'.json')
    sources={str(SOURCE/'mapped_selections'/(r['question_id']+'.json')):
             sha256(SOURCE/'mapped_selections'/(r['question_id']+'.json')) for r in items}
    if dest.exists():
        old=read_json(dest)
        if old['sources']!=sources:raise ValueError('Changed mapping on resume')
        return old
    # 步骤1：确认本地视频字节身份，再用正式精确PTS取帧接口得到像素。
    video=items[0]['selection']['video']
    if sha256(video['path'])!=items[0]['source_video_sha256']:
        raise ValueError('Local source video changed')
    expected={};rows=[]
    for item in items:
        if item['source_video_sha256']!=items[0]['source_video_sha256']:raise ValueError('Shared video hash mismatch')
        images,hashes=FrameProvider(item['selection']).decode()
        for image in images:image.close()
        for f,h in zip(item['selection']['selected_frames'],hashes):
            key=f['source_frame_index'];value=(f['source_pts'],h)
            if key in expected and expected[key]!=value:raise ValueError('Repeated source differs across questions')
            expected[key]=value
        rows.append(dict(question_id=item['question_id'],rgb_sha256=hashes))
    # 步骤2：从文件开头枚举解码源帧号，独立检查公开编号、PTS和RGB的三方对应。
    matched={}
    with av.open(video['path']) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=2
        for index,frame in enumerate(container.decode(stream)):
            if index in expected:
                pts,digest=expected[index]
                if frame.pts!=pts:raise ValueError('Sequential source number maps to different PTS')
                im=frame.to_image();actual=hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest();im.close()
                if actual!=digest:raise ValueError('Sequential and seek pixels differ')
                matched[str(index)]=dict(source_pts=pts,rgb_sha256=actual)
            if index>=max(expected):break
    if len(matched)!=len(expected):raise ValueError('Incomplete sequential decode')
    # 步骤3：保存完整输入RGB身份与核查范围，只声明本地两条解码路径一致。
    record=dict(status='passed',video_id=vid,sources=sources,source_video_sha256=items[0]['source_video_sha256'],
        unique_source_frames=len(matched),sequential_stop_frame=max(expected),matches=matched,questions=rows,
        limitation='同一本地视频的seek与顺序解码；不证明与作者未公布哈希的视频字节相同')
    durable(dest,record);print(vid,len(matched),'RGB verified',flush=True);return record


def main():
    """接口：固定先行7题加4个既有案例，四个CPU任务，模型调用0，恢复不重复解码。"""
    with exclusive(RUN/'run.lock'):
        audit=read_json(SOURCE/'audit.json')
        if audit['status']!='passed' or audit['questions']!=2700:raise ValueError('Metadata acceptance missing')
        groups={}
        # 步骤1：名单在解码前固定，不按是否容易通过动态换题。
        for q in IDS:
            path=SOURCE/'mapped_selections'/(q+'.json')
            if sha256(path)!=audit['exports'][str(path)]:raise ValueError('Mapped selection hash mismatch')
            item=read_json(path);groups.setdefault(item['video_id'],[]).append(item)
        spec=dict(question_ids=IDS,source_audit_sha256=sha256(SOURCE/'audit.json'),code_sha256=sha256(__file__),
                  decode_source_sha256=sha256(ROOT/'src/videoqa_methods/region_context.py'),new_model_calls=0)
        pp=RUN/'protocol.json'
        if pp.exists():
            if read_json(pp)!=spec:raise ValueError('RGB protocol changed')
        else:durable(pp,spec)
        # 步骤2：每视频最多两个解码线程，跨视频并发4，不占用GPU。
        with ThreadPoolExecutor(max_workers=4) as executor:
            records=[]
            for r in executor.map(verify_video,groups.items()):
                records.append(r)
                durable(RUN/'status.json',dict(status='running',videos_done=len(records),videos_total=len(groups),
                                              pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤3：完整核验与未覆盖范围均明确，不将11题外推成全量RGB验收。
        durable(RUN/'summary.json',dict(status='passed',questions=len(IDS),positions=32*len(IDS),videos=len(records),
            unique_source_frames=sum(r['unique_source_frames'] for r in records),new_model_calls=0,
            records={str(RUN/'videos'/(r['video_id']+'.json')):sha256(RUN/'videos'/(r['video_id']+'.json')) for r in records}))
        durable(RUN/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        durable(RUN/'status.json',dict(status='paused',error=repr(exc),utc=utc()));raise
