"""固定先导面板的只读图板：保存来源，不自动作语义判断或释放训练门。"""
import argparse
import hashlib
import html
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,utc
from videoqa_set_value.data import METHODS


def draw_bank(record,checkpoint,root,out):
    """接口：从冻结PNG生成八帧缩略图；核对文件和RGB摘要，不解码原片或改模型输入。"""
    root=Path(root);out=Path(out);frames=record['frames']
    if len(frames)!=8:raise ValueError('Review requires the frozen eight-frame bank')
    canvas=Image.new('RGB',(1536,500),'white');draw=ImageDraw.Draw(canvas)
    try:
        # 步骤1：只读取已完成选择的原图；修改过的图像不得用于审核。
        for i,f in enumerate(frames):
            path=Path(f['image']);relative=str(path.relative_to(root))
            if sha256(path)!=checkpoint['files'][relative]:raise ValueError('Frozen review image changed')
            with Image.open(path) as original:
                image=original.convert('RGB')
                if hashlib.sha256(np.ascontiguousarray(np.asarray(image)).tobytes()).hexdigest()!=f['rgb_sha256']:
                    raise ValueError('Frozen review RGB differs')
                image.thumbnail((380,214));x,y=i%4*384,i//4*250;canvas.paste(image,(x,y));image.close()
            draw.text((x+3,y+216),f"#{i+1} {f['timestamp_seconds']:.3f}s | src {f['source_frame_index']}",fill='black')
        # 步骤2：缩略图是显示派生物，不冒充官方384处理后的像素或新增证据。
        out.parent.mkdir(parents=True,exist_ok=True);canvas.save(out,quality=95)
    finally:canvas.close()
    return dict(path=str(out),sha256=sha256(out),source_pts=[f['source_pts'] for f in frames])


def render(root):
    """接口：仅接受原统计程序冻结的最多12题；不写pilot_review.json或任何训练标签。"""
    root=Path(root);panel=read_json(root/'pilot_review_panel.json')
    if panel['pilot_report_sha256']!=sha256(root/'pilot_report.json'):raise ValueError('Pilot panel/report identity differs')
    ids=panel['question_ids']
    if len(ids)>12 or len(ids)!=len(set(ids)):raise ValueError('Fixed review panel changed')
    rows={x['question_id']:x for x in read_json(root/'manifest.json')['rows']}
    from videoqa_set_value.longvideo_runtime import scoring_row
    from videoqa_set_value.longvideo_task import context
    out=root/'pilot_review_assets';cases=[]
    # 步骤1：面板名单已冻结，不能由本工具按对错另选有利案例。
    for q in ids:
        row=rows[q];reference=scoring_row(root,row)['answer_letter'];checkpoint=read_json(root/'selection_checkpoints'/(q+'.json'))
        boards={};answers={};sources={}
        for m in METHODS:
            path=root/'selections'/m/(q+'.json')
            if sha256(path)!=checkpoint['files'][str(path.relative_to(root))]:raise ValueError('Frozen selection changed')
            record=read_json(path);board=draw_bank(record,checkpoint,root,out/q/(m+'.jpg'))
            result=read_json(root/'qa/pilot/results'/m/(q+'.json'))
            boards[m]=dict(board,source=str(path),source_sha256=sha256(path));sources[m]=record['frames']
            answers[m]=dict(raw_output=result['answer']['raw_output'],correct=result['correct'])
        cases.append(dict(question_id=q,video_id=row['video_id'],context=context(row),reference=reference,
            boards=boards,answers=answers,source_frames=sources))
    # 步骤2：可点击原图与回传分开，审核是事后初审，不伪造盲审或用户一致率。
    page=['<!doctype html><meta charset="utf-8"><title>LongVideo固定先导审核</title>',
        '<style>body{font:16px/1.5 sans-serif;max-width:1550px;margin:24px auto}img{max-width:100%}pre{white-space:pre-wrap}</style>',
        '<h1>固定先导面板：事后画面初审</h1><p>源RGB缩略图，非官方384处理图。小字或细节不清时查看原图；本页不自动认定证据充分。</p>']
    for c in cases:
        page+=['<h2>'+html.escape(c['question_id'])+'</h2><pre>'+html.escape(c['context'])+'</pre>']
        for m,b in c['boards'].items():
            rel=os.path.relpath(b['path'],out);page+=['<h3>'+html.escape(m)+'</h3><a href="'+html.escape(rel,quote=True)+'"><img src="'+html.escape(rel,quote=True)+'"></a>']
            links=[]
            for i,f in enumerate(c['source_frames'][m]):
                rel=os.path.relpath(f['image'],out);links.append('<a href="'+html.escape(rel,quote=True)+'">#'+str(i+1)+'</a>')
            page+=['<p>原图：'+' '.join(links)+'</p>']
        page+=['<details><summary>参考与已完成回传（事后审核）</summary><pre>'+html.escape(str(dict(reference=c['reference'],answers=c['answers'])))+'</pre></details>']
    out.mkdir(parents=True,exist_ok=True);(out/'index.html').write_text('\n'.join(page),encoding='utf-8')
    # 步骤3：只归档显示资产，不碰选择、问答账本或语义审核放行文件。
    value=dict(panel_sha256=sha256(root/'pilot_review_panel.json'),cases=cases,semantic_review='pending',
        new_decodes=0,new_model_calls=0,review_gate_written=False,utc=utc())
    durable(out/'manifest.json',value);return value


def main():
    """接口：事前协议面板尚未生成时拒绝执行，不凭空补案例。"""
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True)
    result=render(parser.parse_args().run_root);print(dict(cases=len(result['cases']),semantic_review=result['semantic_review']))


if __name__=='__main__':main()
