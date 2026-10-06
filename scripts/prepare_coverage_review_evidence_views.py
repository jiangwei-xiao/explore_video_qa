"""固定面板补充源见证与有界无声窗口，仅供审核，不扩充实验候选池。"""
import argparse
import html
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_budget32.frames import FrameProvider
from analyze_coverage_objectives import BASE,image_board

CASES={'409-3':([77,80],[(72,82)]),'556-1':([40,41,42,43],[(38,48)]),
       '363-2':([50,51,52,54,55,56,67,71,74,75,76,78,80,84],[(50,60),(71,81)]),
       '065-3':([32,35,53,54,56,57],[(49,59)])}


def prepare(root):
    """接口：只为已冻结四题精确解码，原粗帧编号映射PTS，不推算平均FPS。"""
    root=Path(root);allowed={r['question_id'] for r in read_json(root/'panel.json')['questions']}
    records={}
    # 步骤1：每题最多两个10秒片段，声音丢弃，输出与正式池完全分离。
    for q,(indices,windows) in CASES.items():
        if q not in allowed or len(windows)>2 or any(b-a>10 for a,b in windows):raise ValueError('Review bound violated')
        pool_path=BASE/'pools'/(q+'.json');pool=read_json(pool_path)
        frames=sorted([pool['candidates'][i] for i in indices],key=lambda f:f['source_pts'])
        dest=root/'panel'/q/'evidence';dest.mkdir(parents=True,exist_ok=True)
        images,hashes=FrameProvider(dict(video=pool['video'],selected_frames=frames)).decode()
        image_board(images,frames,dest/'coarse_witnesses.jpg')
        for i,image in enumerate(images):image.save(dest/f'witness_{indices[i]}.png');image.close()
        clips=[]
        for a,b in windows:
            out=dest/f'window_{a}_{b}.mp4'
            command=[str(Path(sys.executable).parent/'ffmpeg'),'-nostdin','-v','error','-y','-ss',str(a),'-i',pool['video']['path'],
                '-t',str(b-a),'-an','-c:v','libx264','-threads','2','-preset','fast','-crf','23','-vf','scale=1280:-2',str(out)]
            subprocess.run(command,check=True);clips.append(dict(left=a,right=b,file=out.name,sha256=sha256(out),audio=False))
        record=dict(pool_source=str(pool_path),pool_sha256=sha256(pool_path),frames=frames,rgb_sha256=hashes,
            clips=clips,experiment_pool_mutated=False,new_model_calls=0)
        durable(dest/'manifest.json',record);records[q]=record
        page=['<!doctype html><meta charset="utf-8"><h1>'+q+' 已定位粗见证与原片窗口</h1>',
            '<p>仅为审核解码；这些画面没有新增到实验池或重新问答。</p><img style="max-width:100%" src="coarse_witnesses.jpg">']
        for clip in clips:page += [f'<p>{clip["left"]}–{clip["right"]}秒，无声</p><video controls width="960" src="{clip["file"]}"></video>']
        page += ['<pre>'+html.escape(str([(f['source_frame_index'],f['source_pts']) for f in frames]))+'</pre>']
        (dest/'index.html').write_text('\n'.join(page),encoding='utf-8')
    durable(root/'evidence_views.json',dict(records=records,new_model_calls=0,maximum_window_seconds_per_question=20))


def review_page(root):
    """接口：将AI初审和争议写到独立页面，保留匿名画面页供用户修改判断。"""
    root=Path(root);review=read_json(root/'review.json');panel=read_json(root/'panel.json')
    page=['<!doctype html><meta charset="utf-8"><style>body{max-width:1100px;margin:auto;font:16px sans-serif}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style>',
        '<h1>exp07/exp12固定面板初审</h1><p>事后AI审核：已看标准答案和旧报告，非完全盲审；模型调用0。</p><p><a href="index.html">先看匿名画面</a> · <a href="review.json">完整判断JSON</a></p>']
    for case in review['cases']:
        q=case['question_id'];selected=next(c for c in panel['questions'] if c['question_id']==q)
        page += [f'<h2>{q}</h2>', '<p>'+html.escape(case['required'])+'</p>',
            '<p>'+html.escape(case['finding'])+'</p>', '<p>依据：'+html.escape('；'.join(case['basis']))+'</p>',
            '<p>归因：'+html.escape(' / '.join(case['causes']))+'；确信度：'+html.escape(case['confidence'])+'</p>',
            '<p>争议：'+html.escape(case['dispute'])+'</p>',
            ' '.join(f'<a href="panel/{q}/{p}/index.html">{p}画面和得分</a>' for p in selected['profiles'])]
        if q in CASES:page.append(f'<a href="panel/{q}/evidence/index.html">已定位源见证/无声窗口</a>')
        page.append('<p><label>用户复核：<textarea rows="3" cols="100" oninput="localStorage.setItem(\'coverage_review_'+q+'\',this.value)"></textarea></label></p>')
    page.append('<p>网页留言仅存在当前浏览器localStorage，需导出后归档；未收到用户逐题复核前，不报告一致率。</p>')
    page.append('<button onclick="const a=document.createElement(\'a\');a.href=URL.createObjectURL(new Blob([JSON.stringify(Object.fromEntries(Object.keys(localStorage).filter(k=>k.startsWith(\'coverage_review_\')).map(k=>[k,localStorage[k]])),null,2)],{type:\'application/json\'}));a.download=\'user_review.json\';a.click()">导出复核记录</button>')
    (root/'review.html').write_text('\n'.join(page),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--review-page',action='store_true');args=parser.parse_args()
    root=ROOT/'outputs/analysis'/args.run_id
    if args.review_page:review_page(root)
    else:prepare(root)
