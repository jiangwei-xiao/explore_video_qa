"""窗外支持探针的变化图板，按视频和PTS读取，不附用旧答案。"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from PIL import Image,ImageDraw
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_budget32.frames import FrameProvider
from videoqa_audit.query_input_probe import BASE
RUN=ROOT/'outputs/analysis/external_support12_20261003_r1'


def main(run=RUN, question_ids=None):
    """接口：左列移除、右列加入，只是差异展示，不声称逐行有单帧因果关系。"""
    run=Path(run);source=run/'selections.json';data=read_json(source);records=[]
    # 步骤1：仅绘制真实发生变化的题；未知新帧不自动标为有效或无关。
    for r in data['records']:
        if not r['added']:continue
        q=r['question_id'];pool_path=BASE/'pools'/(q+'.json');pool=read_json(pool_path);c=pool['candidates']
        if question_ids is not None and q not in question_ids:continue
        dest=run/'review'/(q+'.jpg');meta=dest.with_suffix('.json')
        if meta.exists():
            old=read_json(meta);assert old['source_sha256']==sha256(source) and old['image_sha256']==sha256(dest)
            records.append(old);continue
        ids=sorted(set(r['added']+r['removed']),key=lambda i:c[i]['source_pts'])
        images,hashes=FrameProvider(dict(video=pool['video'],selected_frames=[c[i] for i in ids])).decode()
        lookup=dict(zip(ids,images));board=Image.new('RGB',(1280,390*max(len(r['added']),len(r['removed']))),'white');draw=ImageDraw.Draw(board)
        # 步骤2：显示的是源RGB等比缩略，绝不作为问答模型的新输入。
        for col,(label,group) in enumerate([('removed',r['removed']),('added',r['added'])]):
            for row,i in enumerate(sorted(group,key=lambda j:c[j]['source_pts'])):
                im=lookup[i];im.thumbnail((630,350));x,y=col*640,row*390;board.paste(im,(x,y))
                draw.text((x,y+355),f'{label} c{i} / {c[i]["timestamp_seconds"]:.3f}s / PTS {c[i]["source_pts"]}',fill='black')
        dest.parent.mkdir(parents=True,exist_ok=True);board.save(dest,quality=95);board.close()
        for im in images:im.close()
        record=dict(question_id=q,source_sha256=sha256(source),pool_sha256=sha256(pool_path),image_sha256=sha256(dest),
            source_frames=[c[i] for i in ids],rgb_sha256=hashes,new_model_calls=0)
        durable(meta,record);records.append(record)
    print(dict(boards=len(records),new_model_calls=0),flush=True)


if __name__=='__main__':main()
