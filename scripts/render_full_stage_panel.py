#!/usr/bin/env python3
"""固定复核面板：CPU精确解码/官方预处理/曲线与可点击图板，不运行模型。"""
import argparse
import hashlib
import html
import os
import sys
from pathlib import Path
from fractions import Fraction
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import durable,utc
from videoqa_lmms_bridge.frames import FrameProvider
from videoqa_lmms_baseline.methods import UNIFORM,SOURCE,METHODS
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor


def render(run,panel_file='review_panel.json'):
    """接口：固定题单的原帧与384输入视图逐一对照，输出仅供人/AI审核。"""
    panel=read_json(run/panel_file);rows={r['question_id']:r for r in read_json(run/'per_question.json')}
    exp=Path(read_json(run/'protocol.json')['experiment']);processor=SigLipImageProcessor();torch.set_num_threads(2)
    records=[];parts=['<!doctype html><meta charset="utf-8"><title>2700题阶段定位复核</title>',
        '<style>body{font-family:sans-serif;max-width:1500px;margin:30px auto}img{max-width:100%}.methods{display:flex;gap:12px}.method{width:33%}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}</style>',
        f'<h1>全量统计后的固定{len(panel)}题复核</h1><p>仅CPU解码与图像预处理，新增模型调用0。已知预测后的事后审核；未填写的判断不是“证据不足”。点击图板下帧号可查看384输入及原帧。区域不是已确认的语义事件。图板c为各方法自己的记录编号，均匀c不是1 FPS候选编号；跨方法请按真实秒数和源帧身份对应。</p>']
    for item in panel:
        q=item['question_id'];row=rows[q];dest=run/'cases'/q;dest.mkdir(parents=True,exist_ok=True)
        pool=read_json(SOURCE/'llava/pools'/(q+'.json'))
        parts+=['<hr>',f'<h2 id="q{q}">{q} | {item["stratum"]} | {item["cell"]}</h2>',
            '<p>'+html.escape(row['question'])+'</p><p>'+'<br>'.join(html.escape(o) for o in row['options'])+'</p>',
            '<details><summary>实际预测与计分（不代表证据充分性）</summary>'+html.escape(str(row['predictions']))+'; 标准 '+row['answer']+'</details>']
        # 步骤1：原候选评分、窗口和最终源时间可视化，新增帧没有ITM分数不补造。
        fig,ax=plt.subplots(figsize=(13,3));c=pool['candidates'];n=pool['initial_count']
        ts=[f['timestamp_seconds'] for f in c[:n]];scores=pool['initial_scores'];ax.plot(ts,scores,lw=.7)
        for g in pool['plan']['regions']:
            l,r=float(Fraction(g['left_fraction'])),float(Fraction(g['right_fraction']))
            ax.axvspan(l,r,alpha=.15,color='orange');ax.text((l+r)/2,1.01,f"R{g['region_id']} q={row['region_budgets'][str(g['region_id'])]}",ha='center',fontsize=7)
        for level,(name,source) in enumerate([('uniform',UNIFORM/'results'),('topk',exp/'results'/METHODS[0]),('rd',exp/'results'/METHODS[1])]):
            rec=read_json(source/(q+'.json'));t=[f['timestamp_seconds'] for f in rec['selection']['selected_frames']]
            ax.scatter(t,[-.08*(level+1)]*len(t),s=12,label=name)
        ax.set_ylim(-.3,1.13);ax.set_xlabel('True source seconds');ax.set_ylabel('Initial BLIP ITM score');ax.legend(loc='upper right');fig.tight_layout();fig.savefig(dest/'timeline.png');plt.close(fig)
        parts.append(f'<img src="cases/{q}/timeline.png"><div class="methods">')
        # 步骤2：实际RGB哈希与BF16像素Tensor哈希均对照历史输入；不加载权重。
        for name,source in [('uniform',UNIFORM/'results'),('topk',exp/'results'/METHODS[0]),('rd',exp/'results'/METHODS[1])]:
            rec=read_json(source/(q+'.json'));images,hashes=FrameProvider(rec['selection']).decode();assert hashes==rec['rgb_sha256']
            pixels=processor.preprocess(images,return_tensors='pt')['pixel_values'].to(torch.bfloat16).contiguous()
            prefix=(str(pixels.dtype)+str(list(pixels.shape))).encode();ph=hashlib.sha256(prefix+pixels.view(torch.uint8).numpy().tobytes()).hexdigest()
            assert ph==rec['result']['observed']['pixel_sha256']
            board=Image.new('RGB',(896,992),'white');draw=ImageDraw.Draw(board);links=[]
            try:
                for j,(im,f) in enumerate(zip(images,rec['selection']['selected_frames'])):
                    im.save(dest/f'{name}_{j:02d}_source.jpg',quality=95)
                    view=Image.fromarray(np.clip(np.rint((pixels[j].float().permute(1,2,0).numpy()*.5+.5)*255),0,255).astype('uint8'))
                    view.save(dest/f'{name}_{j:02d}_384.png');x,y=(j%4)*224,(j//4)*248
                    board.paste(view.resize((224,224)),(x,y));draw.text((x+2,y+226),f"{j+1} t={f['timestamp_seconds']:.3f} c={f['candidate_index']}",fill='black')
                    links.append(f'<a href="cases/{q}/{name}_{j:02d}_384.png">{j+1}: {f["timestamp_seconds"]:.3f}s</a> (<a href="cases/{q}/{name}_{j:02d}_source.jpg">原帧</a>)')
                board.save(dest/f'{name}_board.jpg',quality=92)
            finally:
                for im in images:im.close()
            parts.append(f'<div class="method"><h3>{name}</h3><a href="cases/{q}/{name}_board.jpg"><img src="cases/{q}/{name}_board.jpg"></a>'+''.join('<p>'+x+'</p>' for x in links)+'</div>')
            records.append(dict(question_id=q,method=name,rgb_identity_verified=True,pixel_sha256=ph))
        parts.append('</div><details><summary>窗口与区域预算</summary><pre>'+html.escape(str(row['regions']))+'\n'+html.escape(str(row['region_budgets']))+'</pre></details>')
        print('Rendered',q,flush=True)
    # 步骤3：输出静态页及验收记录，不伪造人工审核标签。
    suffix='' if panel_file=='review_panel.json' else '_extra'
    (run/f'index{suffix}.html').write_text('\n'.join(parts));durable(run/f'panel_render_audit{suffix}.json',dict(status='passed',records=records,new_model_calls=0,utc=utc()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--panel-file',default='review_panel.json');a=p.parse_args()
    assert Path(a.run_id).name==a.run_id and a.run_id not in ('.','..')
    assert Path(a.panel_file).name==a.panel_file
    render(ROOT/'outputs/analysis'/a.run_id,a.panel_file)
