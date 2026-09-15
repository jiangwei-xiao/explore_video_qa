import html
import textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont,ImageOps
from videoqa_runtime.common import read_json,write_json
from videoqa_runtime.video import decode_selected
from .report import load_all,chosen,predicted,METHODS
from .algorithm import membership


def render_cases(run):
    """可视化接口：只对预测变化题读取已选源帧，绘制分数/边界/配额曲线和每组16帧图板；严格按冻结题序输出，不追加模型调用。"""
    run=Path(run)
    # 步骤1：选取所有预测变化题，顺序沿用冻结名单，不按案例是否有利筛选。
    rows,records=load_all(run)
    summary=read_json(run/'summary.json')
    selected_ids=summary['changed_prediction_cases']
    font_file=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    font=ImageFont.truetype(str(font_file),14)
    case_index=[]
    for row in rows:
        qid=row['question_id']
        if qid not in selected_ids: continue
        directory=run/'cases'/qid
        directory.mkdir(parents=True,exist_ok=True)
        pool=records['C'][qid]['pool']
        # 步骤2：合并该题各组已选PTS，一次读取图板所需画面；此操作属于事后分析。
        all_rows={r['source_pts']:r for m in METHODS for r in chosen(records[m][qid])}
        source_rows=sorted(all_rows.values(),key=lambda r:r['source_pts'])
        images=decode_selected(pool['video'],source_rows)
        image_map={r['source_pts']:im for r,im in zip(source_rows,images)}
        # 步骤3：画相关性曲线、边界/热点、各组时间位置和片段配额。
        figure,axes=plt.subplots(3,1,figsize=(15,10),constrained_layout=True)
        n=pool['initial_count']; c=pool['candidates']; scores=pool['scores']
        times=[r['timestamp_seconds'] for r in c]
        axes[0].plot(times[:n],scores[:n],lw=.7,label='Initial 1 FPS ITM')
        if len(c)>n: axes[0].scatter(times[n:],scores[n:],s=18,c='red',marker='x',label='Fine candidates')
        for segment in pool['segments'][1:]: axes[0].axvline(segment['start_seconds'],color='gray',alpha=.35,lw=.6)
        for h in pool['hotspot_trace']:
            if h['selected']: axes[0].axvspan(h['left_seconds'],h['right_seconds'],color='orange',alpha=.25)
        axes[0].set_ylim(-.03,1.03); axes[0].set_ylabel('ITM probability'); axes[0].legend(loc='upper right')
        axes[0].set_title(qid+': '+textwrap.fill(row['question'],100),fontsize=10)
        width=.8/len(METHODS)
        for i,m in enumerate(METHODS):
            selected=chosen(records[m][qid]); ts=[r['timestamp_seconds'] for r in selected]
            axes[1].scatter(ts,[i]*16,marker='|',s=130,label=m)
            positions={r['source_pts']:j for j,r in enumerate(c)}
            groups=membership(c,pool['segments'])[[positions[r['source_pts']] for r in selected]]
            counts=np.bincount(groups,minlength=len(pool['segments']))
            axes[2].bar(np.arange(len(counts))+(i-2.5)*width,counts,width=width,label=m)
        axes[1].set_yticks(range(len(METHODS)),METHODS); axes[1].set_xlabel('Source time (seconds)')
        axes[2].set_xlabel('Initial segment ID'); axes[2].set_ylabel('Selected frame quota'); axes[2].legend(ncol=6)
        figure.savefig(directory/'selection.png',dpi=130); plt.close(figure)
        # 步骤4：为每组生成16帧图板，并附上原问题、参考答案与实际预测。
        sections=[]
        for m in METHODS:
            selected=chosen(records[m][qid]); record=records[m][qid]
            board=Image.new('RGB',(960,4*166+40),'white'); draw=ImageDraw.Draw(board)
            draw.text((8,8),f'{qid} | {m} | prediction={predicted(record)} | correct={record["correct"]}',font=font,fill='black')
            for i,r in enumerate(selected):
                x=(i%4)*240; y=(i//4)*166+40
                image=ImageOps.contain(image_map[r['source_pts']],(236,132))
                board.paste(image,(x+(240-image.width)//2,y))
                draw.text((x+3,y+135),f'{i+1}: {r["timestamp_seconds"]:.2f}s / #{r["source_frame_index"]}',font=font,fill='black')
            board.save(directory/f'{m}.jpg',quality=88)
            scope=record['pool']['scope']['label'] if m in ('B','C') else 'n/a'
            sections.append(f'<h2>{m}: {html.escape(str(predicted(record)))}; scope={scope}</h2><img src="{m}.jpg" alt="{m} selected frames">')
        page='<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;max-width:1400px;margin:25px auto}img{max-width:100%}pre{white-space:pre-wrap}</style>'
        page+=f'<h1>{qid}</h1><pre>{html.escape(row["question"])}\n{html.escape(chr(10).join(row["options"]))}</pre><p>Reference: {"ABCD"[row["answer_index"]]}</p><img src="selection.png" alt="score, boundaries, selections, quotas">'+''.join(sections)
        (directory/'index.html').write_text(page)
        case_index.append(dict(question_id=qid,stratum=row['stratum'],path=f'cases/{qid}/index.html',
                               predictions={m:predicted(records[m][qid]) for m in METHODS}))
        print(f'Case rendered: {qid} ({len(case_index)}/{len(selected_ids)})',flush=True)
    # 步骤5：保存完整案例索引，供正文按固定题序选择正负案例。
    write_json(run/'case_index.json',case_index)
    links=''.join(f'<li><a href="{x["path"]}">{x["question_id"]}</a> — {html.escape(str(x["predictions"]))}</li>' for x in case_index)
    (run/'cases.html').write_text('<!doctype html><meta charset="utf-8"><h1>All changed-prediction cases, in frozen sample order</h1><ul>'+links+'</ul>')
    return case_index
