"""全量证据审计辅助：生成实际384输入图板与像素疑点，不自动赋予语义标签。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def prepare(out):
    """输入既有审计目录，输出384原尺寸图板和只作提示的像素统计。"""
    out=Path(out);root=out/'full_review';root.mkdir(exist_ok=True)
    questions=json.loads((out/'questions.json').read_text());metadata=[]
    # 步骤1：全部使用已恢复的精确PTS图像，禁止模型调用或重新评分。
    for q in questions:
        frames=q['frames'];target=root/'boards'/q['question_id'];target.mkdir(parents=True,exist_ok=True)
        for start in range(0,len(frames),16):
            dest=target/f'{start//16+1:02d}.png'
            if dest.exists():continue
            chunk=frames[start:start+16];board=Image.new('RGB',(1536,410*((len(chunk)+3)//4)),'white');draw=ImageDraw.Draw(board)
            for i,row in enumerate(chunk):
                x=(i%4)*384;y=(i//4)*410
                with Image.open(out/'frames'/q['question_id']/f'{row["source_pts"]}.input.png') as im:board.paste(im,(x,y))
                draw.text((x+3,y+386),f'{start+i+1} | {row["timestamp_seconds"]:.3f}s | {row["source_pts"]}',fill='black')
            board.save(dest)
        # 步骤2：保存原图与实际输入的亮度/动态范围；不把阈值当成过滤规则。
        for i,row in enumerate(frames,1):
            p=out/'frames'/q['question_id']/f'{row["source_pts"]}.original.png'
            with Image.open(p) as im:rgb=np.asarray(im.convert('RGB'))
            metadata.append(dict(question_id=q['question_id'],union_index=i,source_pts=row['source_pts'],
                mean=float(rgb.mean()),std=float(rgb.std()),minimum=int(rgb.min()),maximum=int(rgb.max()),
                near_black_pixel_fraction=float(np.mean(rgb.max(axis=2)<=3)),
                warning='像素指标仅供定位，低照度/黑屏的证据作用必须人工审核'))
        print(q['question_id'],len(frames),flush=True)
    # 步骤3：独立派生产物不覆盖首批审计及原实验记录。
    (root/'pixel_statistics.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);prepare(p.parse_args().out)
