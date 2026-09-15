"""从已保存官方384输入视图生成原尺寸图板，供人工核查，不重新处理视频。"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_audit.core import read


def build(out):
    """首批10题输入视图分页；每张384图不缩放，最多16帧一页。"""
    from PIL import Image,ImageDraw
    out=Path(out)
    # 步骤1：仅读取渲染完成的PNG，按原始联合图板编号对应。
    for q in read(out/'questions.json'):
        if not q['pilot']:continue
        root=out/'frames'/q['question_id']
        for start in range(0,len(q['frames']),16):
            rows=q['frames'][start:start+16];board=Image.new('RGB',(1536,410*((len(rows)+3)//4)),'white');draw=ImageDraw.Draw(board)
            for i,r in enumerate(rows):
                x=(i%4)*384;y=(i//4)*410
                with Image.open(root/f'{r["source_pts"]}.input.png') as image:board.paste(image,(x,y))
                draw.text((x+3,y+386),f'{start+i+1} | {r["timestamp_seconds"]:.3f}s',fill='black')
            # 步骤2：PNG无损保存，保留原预处理像素供审核者放大查看。
            board.save(root/f'input_board_{start//16+1:02d}.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);build(p.parse_args().out)
