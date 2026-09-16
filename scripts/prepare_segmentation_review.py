"""针对六题目标证据窗口重建密集原片图板，分开保存人工可见性与原算法日志。"""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
from fractions import Fraction
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import av
from PIL import Image,ImageDraw,ImageOps
from videoqa_runtime.common import ROOT,read_json,write_json,sha256

WINDOWS={'103-1':[(3.5,8.5,8),(5.5,7.6,0)],'306-2':[(76,106,1)],'260-2':[(52,66,2)],
         '170-2':[(0,22,1)],'443-1':[(95,145,.5)],'004-3':[(74,97,1)]}


def prepare(out):
    """接口：按明确核查范围读取原帧，0采样率表示范围内每一源帧，禁止对未查看区域作穷尽结论。"""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'outputs/methods/method_v1_videomme50_20260914_r1'
    summary=[]
    # 步骤1：读取冻结B/C的候选、片段和补查窗口，图片只用于事后诊断。
    for qid,windows in WINDOWS.items():
        records={m:read_json(source/'results'/m/f'{qid}.json') for m in ('A','B','C')}
        pool=records['B']['pool'];video=pool['video'];initial={r['source_pts']:r['candidate_index'] for r in pool['candidates']}
        item=dict(question_id=qid,question=records['B']['question'],options=records['B']['options'],scope=pool['scope'],
                  segments=pool['segments'],quotas=pool['quotas'],hotspots=[h for h in records['C']['pool']['hotspot_trace'] if h['selected']],windows=[])
        for wi,(left,right,fps) in enumerate(windows):
            dest=out/qid/f'window_{wi+1}';dest.mkdir(parents=True)
            rows=[];selected=[];next_time=Fraction(str(left));tb=Fraction(video['time_base']);origin=video['start_pts']
            # 步骤2：顺序解码保证源帧编号准确；一边保存完整静音窗口，一边取查看图板。
            with av.open(video['path']) as inp,av.open(str(dest/'source_window.mp4'),'w',options={'movflags':'+faststart'}) as clip:
                stream=inp.streams.video[0];stream.codec_context.thread_count=2
                encoded=clip.add_stream('libx264',rate=stream.average_rate or 25);encoded.width=768;encoded.height=432;encoded.pix_fmt='yuv420p'
                encoded.options={'crf':'18','preset':'fast'};first_pts=None
                for source_index,frame in enumerate(inp.decode(stream)):
                    t=(frame.pts-origin)*tb
                    if t<Fraction(str(left)):continue
                    if t>Fraction(str(right)):break
                    if first_pts is None:first_pts=frame.pts
                    im=frame.to_image();canvas=Image.new('RGB',(768,432),'black');small=ImageOps.contain(im,(768,432));canvas.paste(small,((768-small.width)//2,(432-small.height)//2))
                    vf=av.VideoFrame.from_image(canvas);vf.pts=frame.pts-first_pts;vf.time_base=tb
                    for packet in encoded.encode(vf):clip.mux(packet)
                    if not fps or t>=next_time:
                        row=dict(source_pts=frame.pts,source_frame_index=source_index,seconds=float(t),initial_candidate_index=initial.get(frame.pts))
                        rows.append(row);selected.append(im)
                        im.save(dest/f'{frame.pts}.jpg',quality=95)
                        if fps:
                            step=1/Fraction(str(fps))
                            while next_time<=t:next_time+=step
                for packet in encoded.encode():clip.mux(packet)
            # 步骤3：12帧一页，标真实源PTS/时间，原候选命中单独显示，不把密集新帧当已参与实验。
            for offset in range(0,len(rows),12):
                board=Image.new('RGB',(1280,40+228*((min(12,len(rows)-offset)+3)//4)),'white');draw=ImageDraw.Draw(board)
                draw.text((5,8),f'{qid} source window {left}-{right}s | {fps or "ALL SOURCE"} fps | diagnostic only',fill='black')
                for j,(row,im) in enumerate(zip(rows[offset:offset+12],selected[offset:offset+12])):
                    x=j%4*320;y=j//4*228+40;small=ImageOps.contain(im,(320,190));board.paste(small,(x+(320-small.width)//2,y))
                    draw.text((x+3,y+192),f'{row["seconds"]:.5f}s PTS {row["source_pts"]}',fill='black')
                    draw.text((x+3,y+208),f'source #{row["source_frame_index"]}; initial #{row["initial_candidate_index"]}',fill='black')
                board.save(dest/f'board_{offset//12+1:02}.jpg',quality=95)
            write_json(dest/'frames.json',dict(left=left,right=right,sampling_fps=fps,all_source_frames=fps==0,frames=rows))
            item['windows'].append(dict(path=str(dest.relative_to(out)),left=left,right=right,sampling_fps=fps,frames=len(rows)))
        write_json(out/qid/'algorithm.json',item);summary.append(item)
    write_json(out/'summary.json',dict(cases=summary,model_calls=0,blip_calls=0,scope='指定窗口视觉复核；非六条完整视频穷尽审核'))
    print(f'{len(summary)} cases generated',flush=True)


if __name__=='__main__':prepare(sys.argv[1])
