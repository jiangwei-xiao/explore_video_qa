"""定位输入探针图板与排名曲线；展示的是Top32种子，不是最终问答输入。"""
import argparse
from pathlib import Path
import sys
import html
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_budget32.frames import FrameProvider
from videoqa_audit.query_input_probe import CONDITIONS


def render_one(job):
    """接口：精确解码种子帧并对齐评分时RGB，原比例图板不冒充模型预处理视图。"""
    run,q=job;item=read_json(run/'inputs'/(q+'.json'));result=read_json(run/'results'/(q+'.json'));pool=read_json(item['pool_path'])
    directory=run/'review'/q;directory.mkdir(parents=True,exist_ok=True)
    for condition in CONDITIONS:
        ids=sorted(result['plans'][condition]['seed_indices'],key=lambda i:pool['candidates'][i]['source_pts'])
        fs=[pool['candidates'][i] for i in ids];images,hashes=FrameProvider(dict(video=pool['video'],selected_frames=fs)).decode()
        assert hashes==[result['rgb_sha256'][i] for i in ids]
        rank={i:j+1 for j,i in enumerate(result['ranking'][condition])}
        board=Image.new('RGB',(1280,210*((len(fs)+3)//4)),'white');draw=ImageDraw.Draw(board)
        # 步骤1：全32个入口按源时间展示，同时标原排名；不据答案裁图选例。
        for j,(im,f,i) in enumerate(zip(images,fs,ids)):
            x,y=j%4*320,j//4*210;im.thumbnail((316,178));board.paste(im,(x,y));im.close()
            draw.text((x,y+180),f"{f['timestamp_seconds']:.3f}s c{i} rank{rank[i]}",fill='black')
        path=directory/(condition+'.jpg');board.save(path,quality=93);board.close()
        durable(directory/(condition+'.json'),dict(frames=fs,source_rgb_sha256=hashes,image_sha256=sha256(path),role='Top32 localization seeds, NOT final QA frames'))


def main(run):
    """接口：完整评分后生成固定9题对照页；不新增评分、问答或细采样。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);summary=read_json(run/'summary.json');assert summary['status']=='completed'
    questions=[x['question_id'] for x in summary['questions']]
    with ThreadPoolExecutor(max_workers=4) as executor:list(executor.map(render_one,[(run,q) for q in questions]))
    page=['<!doctype html><meta charset="utf-8"><title>定位文本对照</title><h1>原题与全部选项的定位输入对照</h1>',
        '<p>展示Top32定位种子和拟议窗口，不是最终输入，没有新增问答。所有选项同等输入，不用标准答案；已列见证不穷尽全部证据。</p>']
    # 步骤2：排名曲线仅在同题比较，避免把跨查询绝对ITM值解释为证据置信度。
    for q in questions:
        item=read_json(run/'inputs'/(q+'.json'));result=read_json(run/'results'/(q+'.json'));pool=read_json(item['pool_path'])
        n=pool['initial_count'];ts=[f['timestamp_seconds'] for f in pool['candidates'][:n]]
        fig,ax=plt.subplots(figsize=(13,3.2))
        for condition,color in zip(CONDITIONS,['tab:blue','tab:orange']):
            percentile=np.empty(n)
            percentile[result['ranking'][condition]]=np.arange(n)/max(1,n-1)
            ax.plot(ts,percentile,label=condition,color=color,alpha=.75,linewidth=.7)
        ax.invert_yaxis();ax.set(xlabel='Source time (seconds)',ylabel='Normalized rank (0=best)',title=q);ax.legend();fig.tight_layout()
        fig.savefig(run/'review'/q/'ranks.png',dpi=120);plt.close(fig)
        page += [f'<h2>{q}</h2><p>{html.escape(item["question"])}</p><p>{html.escape(" / ".join(item["options"]))}</p>',
            f'<img style="max-width:100%" src="{q}/ranks.png">',
            '<table border="1" cellpadding="5"><tr><th>已列线索</th><th>条件</th><th>最佳排名</th><th>Top32</th><th>窗口可达</th></tr>']
        for w in summary['witnesses']:
            if w['question_id']==q:page.append(f'<tr><td>{html.escape(w["meaning"])}</td><td>{w["condition"]}</td><td>{w["best_rank"]}</td><td>{w["seed_hit"]}</td><td>{w["window_hit"]}</td></tr>')
        page.append('</table>')
        for condition in CONDITIONS:page.append(f'<h3>{condition} 定位种子</h3><a href="{q}/{condition}.jpg"><img style="max-width:100%" src="{q}/{condition}.jpg"></a>')
    (run/'review/index.html').write_text('\n'.join(page),encoding='utf-8')
    durable(run/'review/assets.json',dict(status='rendered_pending_review',questions=questions,source_boards=18,new_model_calls=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);main(parser.parse_args().run)
