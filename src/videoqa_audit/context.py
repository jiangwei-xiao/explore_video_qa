"""原片证据窗口：只解码视觉，不调用任何模型或读取音频/外部字幕。"""
from pathlib import Path
from .core import read, write, digest


def render_windows(out, specifications):
    """输入题号、起止秒和采样频率；从文件起点计数，保存真实PTS与源帧号。"""
    import av
    from PIL import Image,ImageDraw,ImageOps
    out=Path(out); questions={q['question_id']:q for q in read(out/'questions.json')}
    # 步骤1：窗口是事后证据核查，不加入原候选池，不改变任何实验输入。
    for spec in specifications:
        q=questions[spec['question_id']];left=spec['left'];right=spec['right'];fps=spec['fps']
        if not (0 <= left < right <= q['video']['duration_seconds']) or not (0 < fps <= 4):
            raise ValueError('无效证据窗口')
        root=out/'context'/q['question_id']/f'{left:g}_{right:g}_{fps:g}'
        if (root/'frames.json').exists():continue
        root.mkdir(parents=True,exist_ok=True)
        if digest(q['video']['path'])!=q['source_video_sha256']:raise ValueError('视频身份不符')
        rows=[];thumbs=[];target=left
        # 步骤2：逐帧从起点恢复编号；仅视频流解码，保留窗口内第一张不早于网格点的帧。
        with av.open(q['video']['path']) as container:
            stream=container.streams.video[0];stream.codec_context.thread_count=2
            for index,frame in enumerate(container.decode(stream)):
                seconds=float((frame.pts-q['video']['start_pts'])*stream.time_base)
                if seconds>right:break
                if seconds+1e-9<target:continue
                image=frame.to_image();image.save(root/f'{frame.pts}.png')
                row=dict(source_pts=frame.pts,source_frame_index=index,timestamp_seconds=seconds)
                rows.append(row)
                thumb=Image.new('RGB',(320,210),'white');small=ImageOps.contain(image,(320,180));thumb.paste(small,(0,0))
                ImageDraw.Draw(thumb).text((3,185),f'{len(rows)} | {seconds:.3f}s | PTS {frame.pts}',fill='black');thumbs.append(thumb)
                while target<=seconds+1e-9:target+=1/fps
        # 步骤3：保存窗口覆盖范围；没有观察到不等于全片不存在。
        for start in range(0,len(thumbs),24):
            page=Image.new('RGB',(1280,1260),'white')
            for i,thumb in enumerate(thumbs[start:start+24]):page.paste(thumb,((i%4)*320,(i//4)*210))
            page.save(root/f'context_{start//24+1:02d}.jpg',quality=94)
        write(root/'frames.json',dict(specification=spec,frames=rows,visual_only=True,source_sha256=q['source_video_sha256']))
        print(f'原片窗口完成 {q["question_id"]} {left}-{right}',flush=True)
