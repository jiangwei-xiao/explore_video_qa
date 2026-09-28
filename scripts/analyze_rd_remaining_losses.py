"""exp03相对均匀退化的窗口资格检查；帧是否在窗内不等于是否为证据。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import html
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,rows
from videoqa_methods.density_selection import source_time
from videoqa_budget32.frames import FrameProvider

BASE=ROOT/'outputs/full_videoqa/uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01'
EXP=ROOT/'outputs/diagnostics/rd12_exp03__qwen__dev200__k32__20260928_r1'
PANEL=['584-3','724-3','541-1','556-1','453-2','840-2','632-3','781-1']

def eligibility(frames,pool):
    """接口：精确有理数窗口归属与到窗口距离；不按平均FPS推断源时间。"""
    regions=pool['plan']['regions'];video=pool['video'];out=[]
    for frame in frames:
        t=source_time(frame,video)
        inside=[r['region_id'] for r in regions if Fraction(r['left_fraction'])<=t<=Fraction(r['right_fraction'])]
        distances=[max(Fraction(r['left_fraction'])-t,t-Fraction(r['right_fraction']),Fraction(0)) for r in regions]
        nearest=min(range(pool['initial_count']),key=lambda i:(abs(source_time(pool['candidates'][i],video)-t),pool['candidates'][i]['source_pts']))
        out.append(dict(frame=frame,region_ids=inside,distance_to_window_seconds=float(min(distances)),
            nearest_coarse_index=nearest,nearest_coarse_seconds=pool['candidates'][nearest]['timestamp_seconds']))
    return out

def render(job):
    """接口：固定八题双图板；红框仅表示窗外，不能自动标为无关或缺失证据。"""
    directory,q,kind=job
    path=BASE/'results/BASE-Uniform'/(q+'.json') if kind=='uniform' else EXP/'results/RD-1.2-exp03'/(q+'.json')
    pool=read_json(BASE/'pools'/(q+'.json'));result=read_json(path)
    target=directory/q/(kind+'.jpg');target.parent.mkdir(parents=True,exist_ok=True)
    metadata=target.with_suffix('.json');source_hashes={str(path):sha256(path),str(BASE/'pools'/(q+'.json')):sha256(BASE/'pools'/(q+'.json'))}
    if metadata.exists():
        record=read_json(metadata);assert record['sources']==source_hashes and sha256(target)==record['image_sha256'];return
    from PIL import Image,ImageDraw
    images,hashes=FrameProvider(result['selection']).decode();assert hashes==result['rgb_sha256']
    marked=eligibility(result['selection']['selected_frames'],pool)
    # 步骤1：全部已用源图保留；附资格标记方便定位而不替代画面判断。
    board=Image.new('RGB',(1280,210*8),'white');draw=ImageDraw.Draw(board)
    for i,(im,m) in enumerate(zip(images,marked)):
        im.thumbnail((312,178));x=i%4*320;y=i//4*210;board.paste(im,(x+2,y+2));im.close()
        color='green' if m['region_ids'] else 'red';draw.rectangle((x,y,x+315,y+181),outline=color,width=3)
        f=m['frame'];draw.text((x,y+182),f"{f['timestamp_seconds']:.3f}s src{f['source_frame_index']} {'IN' if m['region_ids'] else 'OUT'}",fill=color)
    board.save(target,quality=92);board.close()
    durable(metadata,dict(question_id=q,kind=kind,sources=source_hashes,image_sha256=sha256(target),frames=marked,new_model_calls=0))

def main(output):
    """接口：全18题自动核对，明确排除2题歧义机制归因；先固定8题语义复核名单。"""
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    sp=EXP/'summary.json';summary=read_json(sp)
    qs=summary['comparisons']['K32__BASE-Uniform']['cells']['right_only'];assert len(qs)==18
    data={r['question_id']:r for r in rows()};records=[]
    # 步骤1：输出全部窗口资格与最近粗候选，不能只保留方便解释的案例。
    for q in qs:
        p=BASE/'pools'/(q+'.json');pool=read_json(p);u=read_json(BASE/'results/BASE-Uniform'/(q+'.json'))
        fs=eligibility(u['selection']['selected_frames'],pool)
        records.append(dict(question_id=q,question=data[q]['question'],task_type=data[q]['task_type'],
            known_ambiguity=q in ('321-2','491-2'),uniform_frames=fs,outside_count=sum(not f['region_ids'] for f in fs),
            quotas=pool['selection_trace']['region_budgets'],regions=pool['plan']['regions'],pool_sha256=sha256(p)))
    durable(directory/'window_eligibility.json',dict(records=records,source_sha256=sha256(sp),new_model_calls=0))
    assert set(PANEL)<=set(qs)
    durable(directory/'panel.json',dict(question_ids=PANEL,rule='predefined eight explicit time/object/title cases among remaining losses',new_model_calls=0))
    # 步骤2：精确解码图板复用；旧预测只展示，不参与任何帧资格或方法选择。
    with ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(render,[(directory,q,k) for q in PANEL for k in ('uniform','exp03')]))
    page=['<!doctype html><meta charset="utf-8"><h1>剩余退化：窗口资格与源图</h1><p>红框=不在RD窗口；绿框=窗内。颜色不代表画面有无答题信息。</p>']
    for q in PANEL:
        r=data[q];page+=['<h2>'+q+'</h2><p>'+html.escape(r['question'])+'</p><p>'+html.escape(' / '.join(r['options']))+'</p>']
        for k in ('uniform','exp03'):page.append(f'<h3>{k}</h3><a href="{q}/{k}.jpg"><img style="max-width:100%" src="{q}/{k}.jpg"></a>')
    (directory/'index.html').write_text('\n'.join(page),encoding='utf-8')
    print([(r['question_id'],r['outside_count']) for r in records],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)
