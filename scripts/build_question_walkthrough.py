"""从冻结实验重建单题逐阶段视频与图板，不问答、不评分、不修改原结果。"""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import shutil
import sys
from fractions import Fraction
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import av
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from videoqa_runtime.common import ROOT,read_json,write_json,sha256,offline_environment
from videoqa_runtime.video import decode_selected
from videoqa_methods.algorithm import membership

HISTORY=ROOT/'outputs/methods/method_v1_videomme50_20260914_r1'
FOLLOWUP=ROOT/'outputs/methods/followups_clocal_query_videomme50_20260915_r1'


def preview_video(path,rows,images,meta):
    """接口：每张候选展示1秒，字幕标真实源时间；不是原片连续运动，不插值、不含音轨。"""
    # 步骤1：使用浏览器可播放H.264/YUV420P，保持画面比例，时间信息单独写在黑边。
    with av.open(str(path),'w',options={'movflags':'+faststart'}) as output:
        stream=output.add_stream('libx264',rate=1)
        stream.width=768;stream.height=480;stream.pix_fmt='yuv420p'
        stream.options={'crf':'18','preset':'fast'}
        for position,row in enumerate(rows):
            pts=row['source_pts'];canvas=Image.new('RGB',(768,480),'black')
            im=ImageOps.contain(images[pts],(768,432));canvas.paste(im,((768-im.width)//2,(432-im.height)//2))
            draw=ImageDraw.Draw(canvas);info=meta[pts]
            draw.text((8,440),f'candidate #{info["index"]} | source {row["timestamp_seconds"]:.5f}s | PTS {pts}',fill='white')
            draw.text((8,457),f'BLIP {info["score"]:.6f} | rank {info["rank"]} | segment {info["segment"]+1} | preview 1 image/sec',fill='white')
            frame=av.VideoFrame.from_image(canvas);frame.pts=position;frame.time_base=Fraction(1)
            for packet in stream.encode(frame):output.mux(packet)
        for packet in stream.encode():output.mux(packet)
    # 步骤2：逐帧解码核对视频确实包含目标数量，不以文件存在代替成功。
    with av.open(str(path)) as video:
        assert sum(1 for _ in video.decode(video=0))==len(rows)


def board(path,rows,views,meta,title):
    """接口：保持384输入视图原尺寸，显示槽位、候选编号、PTS和源时间。"""
    canvas=Image.new('RGB',(1536,40+((len(rows)+3)//4)*416),'white');draw=ImageDraw.Draw(canvas)
    draw.text((8,8),title,fill='black')
    for i,r in enumerate(rows):
        x=i%4*384;y=i//4*416+40;pts=r['source_pts'];m=meta[pts]
        canvas.paste(views[pts],(x,y))
        draw.text((x+3,y+386),f'slot {i+1} | #{m["index"]} | {r["timestamp_seconds"]:.5f}s',fill='black')
        draw.text((x+3,y+401),f'PTS {pts} | score {m["score"]:.6f} | rank {m["rank"]}',fill='black')
    canvas.save(path)


def build(qid,out):
    """核心接口：读取七组冻结结果，重建初始候选/分段/最终帧，保存独立网页和溯源清单。"""
    offline_environment()
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for name in ('images','videos','boards'):(out/name).mkdir()
    records={};sources={}
    for method in ('uniform','topk','A','B','C','D','E'):
        path=(FOLLOWUP/'results/E' if method=='E' else HISTORY/'baseline_reference/results'/method if method in ('uniform','topk') else HISTORY/'results'/method)/f'{qid}.json'
        records[method]=read_json(path);sources[str(path)]=sha256(path)
    pools={m:r.get('pool',r.get('selection')) for m,r in records.items()}
    base=pools['B'];rows=base['candidates'];segments=base['segments'];groups=membership(rows,segments)
    assert pools['topk']['candidates']==rows
    assert pools['topk']['scores']==base['scores']
    # 步骤1：初始候选及七组最终帧取精确PTS并集；不重新建立或更改任何分数。
    all_rows={r['source_pts']:r for r in rows}
    for p in pools.values():all_rows.update({r['source_pts']:r for r in p['selected_frames']})
    ordered=sorted(all_rows.values(),key=lambda r:r['source_pts'])
    originals=dict(zip([r['source_pts'] for r in ordered],decode_selected(base['video'],ordered)))
    from transformers import SiglipImageProcessor
    processor=SiglipImageProcessor.from_pretrained(read_json(ROOT/'configs/local_model_inventory.json')['models']['vision']['path'],local_files_only=True)
    views={}
    for pts,im in originals.items():
        im.save(out/'images'/f'{pts}.original.jpg',quality=95)
        pixel=processor.preprocess(im,return_tensors='np')['pixel_values'][0]
        rgb=pixel.transpose(1,2,0)*np.asarray(processor.image_std)+np.asarray(processor.image_mean)
        view=Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype('uint8'));assert view.size==(384,384)
        view.save(out/'images'/f'{pts}.input.png');views[pts]=view
    rankings=sorted(range(len(rows)),key=lambda i:(-base['scores'][i],rows[i]['source_pts'],i))
    rank={i:j+1 for j,i in enumerate(rankings)}
    meta={r['source_pts']:dict(index=i,score=base['scores'][i],rank=rank[i],segment=int(groups[i]),initial=True) for i,r in enumerate(rows)}
    for pool in (pools['C'],pools['E']):
        pool_groups=membership(pool['candidates'],segments)
        for i,r in enumerate(pool['candidates']):
            if r['source_pts'] not in meta:meta[r['source_pts']]=dict(index=f'fine-{i}',score=pool['scores'][i],rank='fine/not-in-initial',segment=int(pool_groups[i]),initial=False)
    shutil.copy2(base['video']['path'],out/'original.mp4')
    assert sha256(out/'original.mp4')==records['B']['source_video_sha256']
    # 步骤2：候选序列按1张/秒播放，分段视频由同一初始池切片组成。
    preview_video(out/'videos/candidates_1fps.mp4',rows,originals,meta)
    for start in range(0,len(rows),16):board(out/'boards'/f'candidates_{start//16+1:02}.png',rows[start:start+16],views,meta,f'{qid} initial candidates {start}..{min(start+15,len(rows)-1)}')
    for segment in segments:
        sid=segment['segment_id'];part=rows[segment['start']:segment['end']]
        preview_video(out/'videos'/f'segment_{sid+1:02}.mp4',part,originals,meta)
    for method,pool in pools.items():
        board(out/'boards'/f'{method}_16.png',pool['selected_frames'],views,meta,f'{qid} {method}: 16 actual input frames (chronological)')
        preview_video(out/'videos'/f'{method}_16.mp4',pool['selected_frames'],originals,meta)
    # 步骤3：保存分类原始输出、配额、完整候选及每步选择，供用户逐环节复盘。
    payload=dict(question_id=qid,question=records['B']['question'],options=records['B']['options'],reference=records['B']['reference_answer'],
        video=base['video'],scope=base['scope'],lambda_value=base['lambda_value'],segments=segments,
        candidates=[dict(**r,**meta[r['source_pts']]) for r in rows],source_hashes=sources,
        methods={m:dict(prediction=r['answer']['parsed_answer'],correct=r['correct'],selected_frames=pools[m]['selected_frames'],
                      selected_indices=pools[m].get('selected_indices'),quotas=pools[m].get('quotas'),
                      competition_trace=pools[m].get('competition_trace'),local_reselection_trace=pools[m].get('local_reselection_trace'),
                      hotspot_trace=pools[m].get('hotspot_trace')) for m,r in records.items()},
        no_new_qa_or_blip=True,preview_note='1张候选显示1秒；以画面字幕的源时间为准，非原片连续动作',
        manual_segment_descriptions='pending visual review; segmentation itself produces no semantic labels')
    write_json(out/'walkthrough.json',payload)
    fields=['candidate_index','source_frame_index','source_pts','timestamp_seconds','score','rank','segment']
    with (out/'candidates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in payload['candidates']:writer.writerow({k:row[k]+1 if k=='segment' else row[k] for k in fields})
    e=html.escape
    page='<!doctype html><meta charset="utf-8"><title>'+qid+'逐阶段review</title><style>body{font:16px/1.7 sans-serif;max-width:1200px;margin:25px auto;padding:0 20px}video{max-width:100%;width:768px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.grid img{max-width:100%}pre{white-space:pre-wrap}section{border-top:2px solid #ddd;margin-top:30px}summary{cursor:pointer}textarea{width:95%;height:90px}</style>'
    page+=f'<h1>{qid}：从原片到最终16帧</h1><p>仅读取冻结实验重建预览，无新增问答或BLIP。候选编号从0开始；片段编号、输入槽位从1开始。标准答案和预测折叠，便于先自行判断。</p>'
    page+=f'<section><h2>1. 先看原片和原问题</h2><pre>{e(payload["question"])}\n{e(chr(10).join(payload["options"]))}</pre><video controls muted preload="metadata" src="original.mp4"></video><p>原文件完整复制；审计仅使用视觉，默认静音。先记下关键动作时间和所需事实。</p></section>'
    page+=f'<section><h2>2. 模型问题分类</h2><details><summary>展开实际分类结果和提示词</summary><p>原始输出：{e(base["scope"]["raw_output"])}；λ={base["lambda_value"]}；回退={base["scope"]["fallback"]}。仅调整权重，没有生成事件锚点。</p><pre>{e(base["scope"]["prompt"])}</pre></details></section>'
    page+=f'<section><h2>3. 1 FPS初始候选：{len(rows)}帧</h2><video controls muted preload="metadata" src="videos/candidates_1fps.mp4"></video><p>每张候选显示1秒，没有补间运动。源PTS/源时间印在画面下方；预览时间不能替代原时间。<a href="candidates.csv">完整分数/排名CSV</a></p>'
    page+=' '.join(f'<a href="boards/candidates_{i:02}.png">候选图板{i}</a>' for i in range(1,(len(rows)+15)//16+1))+'</section>'
    def frame_grid(selected):
        """生成实际帧网格，点击输入视图可以查看原分辨率，显示唯一PTS。"""
        cells=[]
        for slot,row in enumerate(selected,1):
            pts=row['source_pts'];m=meta[pts]
            cells.append(f'<div><a href="images/{pts}.original.jpg"><img loading="lazy" src="images/{pts}.input.png"></a><br>槽位{slot} / 候选#{m["index"]}<br>{row["timestamp_seconds"]:.5f}s / 段{m["segment"]+1}<br>分数{m["score"]:.6f} / 排名{m["rank"]}</div>')
        return '<div class="grid">'+''.join(cells)+'</div>'
    page+='<section><h2>4. BLIP全局Top-K实际16帧</h2><p>分数降序取16帧，再恢复时间顺序；这里的排名都是同题初始候选排名。</p>'+frame_grid(pools['topk']['selected_frames'])+'</section>'
    page+=f'<section><h2>5. 基于问题分数曲线的分段：{len(segments)}段</h2><p>这是分数波动分段，不是语义事件识别。下方的视频仅包含该段初始候选；内容描述由后续视觉review单独补充。</p><div id="segment-notes"></div>'
    for seg in segments:
        sid=seg['segment_id'];page+=f'<details><summary>段{sid+1}：[{seg["start_seconds"]:.5f}, {seg["end_seconds"]:.5f})秒；候选#{seg["start"]}—#{seg["end"]-1}；B配额{base["quotas"][sid]}</summary><video controls muted preload="none" src="videos/segment_{sid+1:02}.mp4"></video><p>Rₑ={seg["representative_score"]:.6f}。<a href="original.mp4#t={seg["start_seconds"]:.5f},{seg["end_seconds"]:.5f}">查看原视频对应连续时间窗口</a></p></details>'
    page+='</section><section><h2>6. 竞争后的实际16帧与各方法对照</h2>'
    for method in ('B','E','A','C','D','uniform'):
        page+=f'<details><summary>{method}：展开16帧（按源时间）</summary><p><a href="boards/{method}_16.png">384原尺寸图板</a> · <a href="videos/{method}_16.mp4">16帧序列视频</a></p>'+frame_grid(pools[method]['selected_frames'])+'</details>'
    page+='</section><section><h2>7. 最后查看答案与过程日志</h2><details><summary>展开标准答案和实际预测</summary><p>标准：'+e(payload['reference'])+'</p><ul>'+''.join(f'<li>{m}: {e(str(x["prediction"]))}，正确={x["correct"]}</li>' for m,x in payload['methods'].items())+'</ul></details><p><a href="walkthrough.json">完整分类、分段、配额、每步竞争和补查记录</a></p></section>'
    page+='<section><h2>你的逐步记录</h2><p>请记下源时间、候选编号或PTS：①原片关键证据；②初始池是否覆盖；③Top-K是否保留；④分段是否切开动作；⑤竞争为何留下/丢掉；⑥最终16帧是否够回答。可直接把记录发回会话；本页不自动保存输入。</p></section>'
    (out/'index.html').write_text(page)
    manifest={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}
    write_json(out/'artifact_manifest.json',manifest)
    print(dict(output=str(out),candidates=len(rows),segments=len(segments),methods=7,files=len(manifest)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--question-id',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();build(args.question_id,args.out)
