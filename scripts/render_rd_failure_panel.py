"""三基线对错分歧的固定六题图板；只精确解码，不产生模型调用。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import html
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,rows
from videoqa_budget32.frames import FrameProvider
from videoqa_budget32.selection import OLD_QWEN
from analyze_rd_budget_failures import SOURCE,METHODS

PANEL=['323-2','568-1','795-1','170-2','278-1','811-2']

def render(job):
    """接口：绘制实际已用源帧；RGB哈希不一致时停止，不能偷换示意图。"""
    directory,q,label,path=job
    item=read_json(path);dest=directory/q/(label+'.jpg')
    dest.parent.mkdir(parents=True,exist_ok=True)
    # 步骤1：恢复只复用来源一致且哈希未变的图板。
    metadata=dest.with_suffix('.json')
    if metadata.exists():
        old=read_json(metadata)
        assert old['source_sha256']==sha256(path) and old['image_sha256']==sha256(dest)
        return old
    from PIL import Image,ImageDraw
    frames=item['selection']['selected_frames'];images,hashes=FrameProvider(item['selection']).decode()
    assert hashes==item['rgb_sha256']
    # 步骤2：原图只做等比缩略；需要判断小文字时另看实际模型尺寸，不冒充模型输入。
    board=Image.new('RGB',(1280,210*((len(images)+3)//4)),'white');draw=ImageDraw.Draw(board)
    for i,(im,f) in enumerate(zip(images,frames)):
        im.thumbnail((316,180));x=(i%4)*320;y=(i//4)*210
        board.paste(im,(x,y));draw.text((x,y+181),f"{i+1}: {f['timestamp_seconds']:.3f}s / src {f['source_frame_index']}",fill='black');im.close()
    board.save(dest,quality=92);board.close()
    rec=dict(question_id=q,label=label,source=str(path),source_sha256=sha256(path),image_sha256=sha256(dest),
        frames=frames,raw_output=item['answer']['raw_output'],correct=item['correct'])
    durable(metadata,rec);return rec


def main(output):
    """接口：先固定三道两基线皆对/RD错、三道RD独对，再复核画面，保留反例。"""
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    # 步骤1：图板名单来自开发200题全部110/001组，没有从新回答中择优。
    durable(directory/'panel.json',dict(question_ids=PANEL,new_model_calls=0,rule='all development200 UTR=110 and UTR=001'))
    data={r['question_id']:r for r in rows()};jobs=[]
    for q in PANEL:
        jobs.append((directory,q,'RD16',OLD_QWEN/'results/RD-1.2'/(q+'.json')))
        jobs += [(directory,q,label,SOURCE/'results'/m/(q+'.json')) for label,m in zip(('Uniform32','TopK32','RD32'),METHODS)]
    with ThreadPoolExecutor(max_workers=4) as executor: records=list(executor.map(render,jobs))
    # 步骤2：预测放入可展开项，页面不根据对错自动赋予语义标签。
    page=['<!doctype html><meta charset="utf-8"><h1>开发200题三方法分歧：六题定位</h1><p>原RGB缩略图，非实际预处理。模型调用0。</p>']
    for q in PANEL:
        row=data[q];page+=['<h2>'+q+'</h2><p>'+html.escape(row['question'])+'</p><p>'+html.escape(' / '.join(row['options']))+'</p>']
        for r in [r for r in records if r['question_id']==q]:
            label=r['label'];page += ['<h3>'+label+'</h3>',f'<a href="{q}/{label}.jpg"><img style="max-width:100%" src="{q}/{label}.jpg"></a>',
                '<details><summary>历史回答</summary>'+html.escape(r['raw_output'])+' / '+str(r['correct'])+'</details>']
    (directory/'index.html').write_text('\n'.join(page),encoding='utf-8')
    print(dict(rendered=len(records),questions=len(PANEL),new_model_calls=0),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)
