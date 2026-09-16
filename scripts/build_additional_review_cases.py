"""另外三道代表题的逐阶段人工review包；仅视频解码和已有日志复用。"""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import csv
import html
from fractions import Fraction
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import av
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from videoqa_runtime.common import ROOT,read_json,write_json,sha256,offline_environment
from videoqa_runtime.baseline_selection import iter_candidate_frames
from videoqa_runtime.video import decode_selected
from videoqa_methods.algorithm import membership
from build_question_walkthrough import board,preview_video,HISTORY,FOLLOWUP
import prepare_segmentation_review as windows_tool

CASES={
 '395-2':dict(title='收到红书后的后续行为',windows=[(260,285,2),(400,430,1)],question='先定位收到红书的事件，再判断之后行为的证据在哪里；不要把拿书/读书等同于全部后续关系。'),
 '383-2':dict(title='量子计算机与交通工具的类比',windows=[(938,950,4)],question='逐帧比较图示的对象对应关系，检查短暂画面与相邻转场是否提供相同信息。'),
 '544-1':dict(title='摩托车离开时的出门人数',windows=[(20,65,2)],question='先确定摩托车离开的时间与门的位置，再区分从门内出来、路过、重复出现；先不要展开答案。')}


class Writer:
    """有界内存候选视频编码器；每张源候选显示1秒，角标始终给真实源时间。"""
    def __init__(self,path):
        """只建立H.264编码状态，不累积全片RGB。"""
        self.out=av.open(str(path),'w',options={'movflags':'+faststart'});self.stream=self.out.add_stream('libx264',rate=1)
        self.stream.width=768;self.stream.height=480;self.stream.pix_fmt='yuv420p';self.stream.options={'crf':'19','preset':'fast'};self.n=0
    def put(self,im,row,info):
        """写入一个候选的原比例预览及分数/排名，不产生插值画面。"""
        canvas=Image.new('RGB',(768,480),'black');small=ImageOps.contain(im,(768,432));canvas.paste(small,((768-small.width)//2,(432-small.height)//2))
        draw=ImageDraw.Draw(canvas);draw.text((8,439),f'#{row["candidate_index"]} source {row["timestamp_seconds"]:.5f}s PTS {row["source_pts"]}',fill='white')
        draw.text((8,457),f'segment {info["segment"]+1} score {info["score"]:.6f} rank {info["rank"]} | 1 candidate / second',fill='white')
        frame=av.VideoFrame.from_image(canvas);frame.pts=self.n;frame.time_base=Fraction(1);self.n+=1
        for packet in self.stream.encode(frame):self.out.mux(packet)
    def close(self):
        """刷新尾部编码帧并关闭文件，调用方随后逐帧核验数量。"""
        for packet in self.stream.encode():self.out.mux(packet)
        self.out.close()


def build(out):
    """生成三题完整候选/分段预览、原片窗口和七方法16帧；原实验与旧审计只读。"""
    offline_environment();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    from transformers import SiglipImageProcessor
    processor=SiglipImageProcessor.from_pretrained(read_json(ROOT/'configs/local_model_inventory.json')['models']['vision']['path'],local_files_only=True)
    # 步骤1：先生成有明确范围的原片窗口，0次模型推理；此处复用解码工具。
    windows_tool.WINDOWS={q:c['windows'] for q,c in CASES.items()};windows_tool.prepare(out/'windows')
    evidence=[]
    for qid,case in CASES.items():
        dest=out/qid;dest.mkdir();(dest/'videos').mkdir();(dest/'images').mkdir();(dest/'boards').mkdir()
        records={};hashes={}
        for m in ('uniform','topk','A','B','C','D','E'):
            p=(FOLLOWUP/'results/E' if m=='E' else HISTORY/'baseline_reference/results'/m if m in ('uniform','topk') else HISTORY/'results'/m)/f'{qid}.json'
            records[m]=read_json(p);hashes[str(p)]=sha256(p)
        pools={m:r.get('pool',r.get('selection')) for m,r in records.items()};base=pools['B'];rows=base['candidates']
        assert pools['topk']['candidates']==rows and pools['topk']['scores']==base['scores']
        group=membership(rows,base['segments']);ranking=sorted(range(len(rows)),key=lambda i:(-base['scores'][i],rows[i]['source_pts'],i));rank={i:j+1 for j,i in enumerate(ranking)}
        meta={r['source_pts']:dict(index=i,score=base['scores'][i],rank=rank[i],segment=int(group[i])) for i,r in enumerate(rows)}
        # 步骤2：完整初始池流式生成1张/秒视频及分段视频，逐候选核对冻结PTS和编号。
        full=Writer(dest/'videos/candidates_1fps.mp4');writers={i:Writer(dest/'videos'/f'segment_{i+1:02}.mp4') for i in range(len(base['segments']))}
        count=0
        for row,frame in iter_candidate_frames(base['video']['path'],{}):
            assert row==rows[count];im=frame.to_image();full.put(im,row,meta[row['source_pts']]);writers[int(group[count])].put(im,row,meta[row['source_pts']]);count+=1
        full.close()
        for w in writers.values():w.close()
        assert count==len(rows)
        shutil.copy2(base['video']['path'],dest/'original.mp4');assert sha256(dest/'original.mp4')==records['B']['source_video_sha256']
        # 步骤3：只保留七组已选帧并集图片，384使用官方处理器而非模型前向。
        combined={r['source_pts']:r for p in pools.values() for r in p['selected_frames']}
        selected=sorted(combined.values(),key=lambda r:r['source_pts'])
        images=dict(zip([r['source_pts'] for r in selected],decode_selected(base['video'],selected)));views={}
        for p in (pools['C'],pools['E']):
            ids=membership(p['candidates'],base['segments'])
            for i,row in enumerate(p['candidates']):
                if row['source_pts'] not in meta:meta[row['source_pts']]=dict(index=f'fine-{i}',score=p['scores'][i],rank='fine/not-in-initial',segment=int(ids[i]))
        for pts,im in images.items():
            im.save(dest/'images'/f'{pts}.original.jpg',quality=95)
            a=processor.preprocess(im,return_tensors='np')['pixel_values'][0].transpose(1,2,0)
            rgb=a*np.asarray(processor.image_std)+np.asarray(processor.image_mean);view=Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype('uint8'))
            view.save(dest/'images'/f'{pts}.input.png');views[pts]=view
        for m,p in pools.items():
            board(dest/'boards'/f'{m}_16.png',p['selected_frames'],views,meta,f'{qid} {m}: chronological 16 inputs')
            preview_video(dest/'videos'/f'{m}_16.mp4',p['selected_frames'],images,meta)
        payload=dict(question_id=qid,question=records['B']['question'],options=records['B']['options'],video=base['video'],source_hashes=hashes,
            scope=base['scope'],lambda_value=base['lambda_value'],segments=base['segments'],quotas=base['quotas'],
            candidates=[dict(**r,score=base['scores'][i],rank=rank[i],segment=int(group[i])+1) for i,r in enumerate(rows)],
            methods={m:dict(prediction=r['answer']['parsed_answer'],correct=r['correct'],selected_frames=pools[m]['selected_frames'],
                           competition_trace=pools[m].get('competition_trace'),hotspot_trace=pools[m].get('hotspot_trace'),
                           local_reselection_trace=pools[m].get('local_reselection_trace')) for m,r in records.items()},
            reference=records['B']['reference_answer'],new_qa=0,new_blip=0)
        write_json(dest/'walkthrough.json',payload)
        with (dest/'candidates.csv').open('w',newline='') as f:
            fields=['candidate_index','source_pts','timestamp_seconds','score','rank','segment'];w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in payload['candidates'])
        # 步骤4：网页先展示问题/原片，答案和我们的定位提示折叠，便于用户自行review。
        e=html.escape;page='<!doctype html><meta charset="utf-8"><style>body{font:16px/1.7 sans-serif;max-width:1200px;margin:25px auto}video{width:768px;max-width:100%}img{max-width:100%}pre{white-space:pre-wrap}summary{cursor:pointer}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}</style>'
        page+=f'<a href="../index.html">返回三题入口</a><h1>{qid}：{case["title"]}</h1><h2>1. 原问题与原片</h2><pre>{e(payload["question"])}\n{e(chr(10).join(payload["options"]))}</pre><video controls muted preload="metadata" src="original.mp4"></video><p>原视频完整文件，默认静音；本次复核只使用视觉。</p>'
        page+=f'<details><summary>查看review提示</summary><p>{case["question"]}</p></details><h2>2. 重点原片窗口</h2><p>以下保持原片动作时序，非1 FPS预览；窗口是人工诊断范围，不是算法自动定位。</p>'
        for i,(a,b,fps) in enumerate(case['windows'],1):
            uri=f'../windows/{qid}/window_{i}';page+=f'<h3>{a}–{b}秒</h3><video controls muted preload="metadata" src="{uri}/source_window.mp4"></video><details><summary>{fps} FPS辅助核查图板（不是原候选池）</summary>'
            for p in sorted((out/'windows'/qid/f'window_{i}').glob('board_*.jpg')):page+=f'<a href="{uri}/{p.name}"><img loading="lazy" src="{uri}/{p.name}"></a>'
            page+='</details>'
        page+=f'<h2>3. 实际分类及完整1 FPS候选</h2><details><summary>范围分类</summary><p>{base["scope"]["label"]}；λ={base["lambda_value"]}</p><pre>{e(base["scope"]["prompt"])}</pre></details><p>共{len(rows)}帧，每张展示1秒，源时间和分数显示在下方；没有插值，不能用预览时长替代原片时长。</p><video controls muted preload="metadata" src="videos/candidates_1fps.mp4"></video><p><a href="candidates.csv">完整候选分数与排名</a></p>'
        page+='<h2>4. Top-K实际16帧</h2><a href="boards/topk_16.png"><img src="boards/topk_16.png"></a>'
        page+=f'<h2>5. {len(base["segments"])}个分数区间及B名额</h2><p>不是人工事件标签，每段视频只播放其初始候选。</p>'
        for s in base['segments']:
            i=s['segment_id'];page+=f'<details><summary>段{i+1}：{s["start_seconds"]:.3f}–{s["end_seconds"]:.3f}秒；{s["end"]-s["start"]}候选；B名额{base["quotas"][i]}</summary><video controls muted preload="none" src="videos/segment_{i+1:02}.mp4"></video></details>'
        page+='<h2>6. 最终16帧对照</h2>'
        for m in ('B','E','uniform','A','C','D'):
            page+=f'<details><summary>{m}实际16帧</summary><p><a href="videos/{m}_16.mp4">16帧序列视频</a></p><a href="boards/{m}_16.png"><img loading="lazy" src="boards/{m}_16.png"></a></details>'
        page+='<h2>7. 最后查看预测与原始日志</h2><details><summary>标准答案和各方法预测</summary><p>标准：'+payload['reference']+'</p>'+''.join(f'<p>{m}：{r["answer"]["parsed_answer"]}，正确={r["correct"]}</p>' for m,r in records.items())+'</details><p><a href="walkthrough.json">完整分类/分段/配额/竞争/补查记录</a></p>'
        (dest/'index.html').write_text(page)
        evidence.append(dict(question_id=qid,candidates=len(rows),segments=len(base['segments']),source_hashes=hashes))
        print('case complete',qid,flush=True)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>另外三道代表性问题</h1><p>从原片、候选、Top-K、分段到最终输入。无新增问答/评分，答案默认折叠。</p><ul>'+''.join(f'<li><a href="{q}/index.html">{q}：{c["title"]}</a></li>' for q,c in CASES.items())+'</ul>')
    write_json(out/'summary.json',dict(cases=evidence,qa_calls=0,blip_calls=0))
    # 步骤5：解码每个候选预览核对数量，并验证原实验身份不变。
    for record in evidence:
        q=record['question_id'];counts={}
        for p in (out/q/'videos').glob('*.mp4'):
            with av.open(str(p)) as c:counts[p.name]=sum(1 for _ in c.decode(video=0))
        assert counts['candidates_1fps.mp4']==record['candidates']
        assert sum(n for k,n in counts.items() if k.startswith('segment_'))==record['candidates']
        assert all(n==16 for k,n in counts.items() if k.endswith('_16.mp4'))
        assert all(sha256(p)==h for p,h in record['source_hashes'].items())
    write_json(out/'artifact_manifest.json',{str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()})
    print('all three cases verified',flush=True)


if __name__=='__main__':build(sys.argv[1])
