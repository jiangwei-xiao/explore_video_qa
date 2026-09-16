"""只为新旧预测变化题及三条预定机制见证生成实际选帧对照，不调用模型。"""
import html
from pathlib import Path
import sys
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PIL import Image, ImageDraw
from videoqa_runtime.common import ROOT, read_json, write_json, offline_environment
from videoqa_runtime.video import decode_selected
from videoqa_methods.followups import HISTORY


def render(run):
    """接口：生成B/C/E每题16帧和变化帧图板；使用源帧及官方384处理器，不执行视觉前向。"""
    import numpy as np
    from transformers import SiglipImageProcessor
    offline_environment()
    run=Path(run); summary=read_json(run/'summary.json')
    rows=read_json(ROOT/'data/manifests/video_mme_development.json')['rows']
    targets=set(summary['comparisons']['E-C']['prediction_changed'])|set(summary['comparisons']['E-B']['prediction_changed'])|{'004-3','893-1','870-1'}
    targets.update(x['question_id'] for x in summary['E_lost_facts'])
    # 步骤1：只加载官方图像处理配置，不加载任何新模型或生成答案。
    inventory=read_json(ROOT/'configs/local_model_inventory.json')
    # 官方LLaVA使用的视觉塔配置在本地SigLIP模型中；从已核验库存读取。
    siglip=inventory['models']['vision']['path']
    processor=SiglipImageProcessor.from_pretrained(siglip,local_files_only=True)
    index=[]
    for row in rows:
        qid=row['question_id']
        if qid not in targets: continue
        records={m:read_json((run/'results/E' if m=='E' else HISTORY/'results'/m)/f'{qid}.json') for m in ('B','C','E')}
        pools={m:r['pool'] for m,r in records.items()}
        # 步骤2：按三组PTS并集精确取源帧，384视图使用官方预处理逆归一化。
        directory=run/'cases'/qid; directory.mkdir(parents=True,exist_ok=True)
        mapping={r['source_pts']:r for p in pools.values() for r in p['selected_frames']}
        source=sorted(mapping.values(),key=lambda r:r['source_pts'])
        frames=decode_selected(pools['E']['video'],source)
        pixels=processor.preprocess(frames,return_tensors='np')['pixel_values']
        mean=np.asarray(processor.image_mean)[:,None,None]; std=np.asarray(processor.image_std)[:,None,None]
        views={}
        for meta,original,pixel in zip(source,frames,pixels):
            pts=meta['source_pts']; original.save(directory/f'{pts}.original.jpg',quality=95)
            view=Image.fromarray(np.clip(np.rint((pixel*std+mean)*255),0,255).astype(np.uint8).transpose(1,2,0))
            if view.size!=(384,384): raise ValueError('Expected official 384 input')
            view.save(directory/f'{pts}.input.png'); views[pts]=view
        sections=[]
        for method,pool in pools.items():
            board=Image.new('RGB',(4*384,4*412+30),'white'); draw=ImageDraw.Draw(board)
            draw.text((8,8),f'{qid} {method} prediction={records[method]["answer"]["parsed_answer"]} correct={records[method]["correct"]}',fill='black')
            cells=[]
            for i,meta in enumerate(pool['selected_frames']):
                x=(i%4)*384; y=(i//4)*412+30; pts=meta['source_pts']
                board.paste(views[pts],(x,y)); draw.text((x+4,y+386),f'{i+1}: {meta["timestamp_seconds"]:.3f}s PTS={pts}',fill='black')
                cells.append(f'<a href="{pts}.original.jpg"><img src="{pts}.input.png" width="192"><br>{meta["timestamp_seconds"]:.3f}s</a>')
            board.save(directory/f'{method}.png')
            sections.append(f'<h2>{method}: {records[method]["answer"]["parsed_answer"]}</h2><div class="grid">'+''.join(cells)+'</div>')
        # 步骤3：变化帧单独列示，便于在30分钟review预算内核查新旧选择差异。
        before={r['source_pts'] for r in pools['C']['selected_frames']}; after={r['source_pts'] for r in pools['E']['selected_frames']}
        changed=sorted(before^after)
        board=Image.new('RGB',(4*384,((len(changed)+3)//4)*412+30),'white'); draw=ImageDraw.Draw(board)
        draw.text((8,8),f'{qid}: C removed / E added',fill='black')
        for i,pts in enumerate(changed):
            x=(i%4)*384; y=(i//4)*412+30; board.paste(views[pts],(x,y))
            draw.text((x+3,y+386),f'{"E+" if pts in after else "C-"} {mapping[pts]["timestamp_seconds"]:.3f}s PTS={pts}',fill='black')
        board.save(directory/'changes.png')
        page='<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;max-width:1200px;margin:auto}.grid{display:grid;grid-template-columns:repeat(4,1fr)}pre{white-space:pre-wrap}</style>'
        page+=f'<h1>{qid}</h1><pre>{html.escape(row["question"])}\n{html.escape(chr(10).join(row["options"]))}</pre><p>Reference: {"ABCD"[row["answer_index"]]}</p><a href="changes.png">C/E变化帧图板</a>'+''.join(sections)
        (directory/'index.html').write_text(page)
        index.append(dict(question_id=qid,path=f'cases/{qid}/index.html',changed_pts=changed,
                          predictions={m:r['answer']['parsed_answer'] for m,r in records.items()}))
        print('rendered',qid,flush=True)
    write_json(run/'case_index.json',index)
    (run/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>C-local变化题及预定机制对照</h1><ul>'+''.join(
        f'<li><a href="{r["path"]}">{r["question_id"]}</a> {html.escape(str(r["predictions"]))}</li>' for r in index)+'</ul>')


if __name__=='__main__': render(sys.argv[1])
