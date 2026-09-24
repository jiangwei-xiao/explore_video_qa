"""预算工作点的三环节轨迹对照；人工事实只在评价层读取。"""
from fractions import Fraction
import html
from pathlib import Path
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import rows,durable,utc
from .selection import SOURCE,OLD_QWEN,METHODS
from .frames import FrameProvider

PANEL=['193-2','004-1','037-1','080-1','007-3','419-2','325-3','303-1','326-2','328-2',
       '616-3','619-3','629-3','606-1','623-3','001-3','177-1','094-1']


def identity(frame):
    """接口：跨池匹配采用真实源编号与PTS，不依赖变化的候选ID。"""
    return (frame['source_frame_index'],frame['source_pts'])


def region_overlap(old,new):
    """接口：按闭窗口交叠建立关系；不将同名区域ID当同一预算单元。"""
    links=[]
    for a in old:
        for b in new:
            left=max(Fraction(a['left_fraction']),Fraction(b['left_fraction']))
            right=min(Fraction(a['right_fraction']),Fraction(b['right_fraction']))
            if left<=right:links.append(dict(old_region=a['region_id'],new_region=b['region_id'],overlap_seconds=float(right-left)))
    return links


def stage_summary(pool):
    """接口：输出定位/补查/配额代理信息，不自动判定语义有效性。"""
    region=pool['plan']['regions'];count=pool['initial_count']
    seeds=set(i for r in region for i in r['seed_indices'])
    picked=set(pool['selected_indices'])
    return dict(seed_count=len(seeds),region_count=len(region),
        window_seconds=sum(float(Fraction(r['right_fraction'])-Fraction(r['left_fraction'])) for r in region),
        refinement_regions=sum(r['refine'] for r in region),new_candidate_count=pool['new_candidate_count'],
        selected_new_count=sum(i>=count for i in picked),seeds_retained=len(seeds&picked),
        quotas=pool['budget_trace']['region_budgets'],regions=region,
        final_frames=pool['selected_frames'])


def analyze(run,render=True):
    """接口：全量轨迹对照与最多24题差异图板，无模型调用。"""
    from PIL import Image,ImageDraw
    run=Path(run);directory=run/'stage_analysis';directory.mkdir(exist_ok=True)
    data=rows();all_records=[];gains={};losses={}
    # 步骤1：依次对齐旧完整池、固定旧池32诊断和正式新池。
    for row in data:
        q=row['question_id'];old=read_json(SOURCE/'llava/pools'/(q+'.json'));new=read_json(run/'pools'/(q+'.json'))
        diag=read_json(run/'diagnostics'/(q+'.json'))
        a={identity(f) for f in old['selected_frames']};b={identity(f) for f in new['selected_frames']}
        d={identity(f) for f in diag['selected_frames']}
        old_answer=read_json(OLD_QWEN/'results'/METHODS[2]/(q+'.json'))
        new_answer=read_json(run/'results'/METHODS[2]/(q+'.json'))
        if new_answer['correct'] and not old_answer['correct']:gains.setdefault(row['stratum'],[]).append(q)
        if old_answer['correct'] and not new_answer['correct']:losses.setdefault(row['stratum'],[]).append(q)
        rec=dict(question_id=q,video_id=row['video_id'],task_type=row['task_type'],stratum=row['stratum'],
            old=stage_summary(old),new=stage_summary(new),window_links=region_overlap(old['plan']['regions'],new['plan']['regions']),
            removed=[list(x) for x in sorted(a-b)],added=[list(x) for x in sorted(b-a)],
            old_retained_in_fixed_pool32=len(a&d),old_retained_in_full32=len(a&b),
            fixed_pool32_final_frames=diag['selected_frames'],fixed_pool32_quotas=diag['budget_trace']['region_budgets'],
            old_correct=old_answer['correct'],new_correct=new_answer['correct'])
        all_records.append(rec)
    durable(directory/'per_question.json',all_records)
    # 步骤2：追加规则事先固定，缺少某类时留空、不跨类别补齐。
    panel=list(PANEL)
    for layer in ('short','medium','long'):
        for changes in (gains,losses):
            extra=next((q for q in changes.get(layer,[]) if q not in panel),None)
            if extra:panel.append(extra)
    assert len(panel)<=24
    durable(directory/'panel.json',dict(question_ids=panel,new_model_calls=0,manual_review_status='pending',
        rule='existing18 + first unseen gain/loss per duration in frozen order',utc=utc()))
    if not render:return
    # 步骤3：原RGB缩略图只辅助复核；不是实际Qwen动态预处理Tensor。
    page=['<!doctype html><meta charset="utf-8"><title>Qwen32差异复核</title>',
        '<h1>Qwen32差异复核</h1><p>原RGB等比缩略图；不代表实际Qwen预处理。人工初审待填写，不自动作证据标签。</p>']
    byid={r['question_id']:r for r in data}
    for q in panel:
        row=byid[q];page.append('<h2>'+html.escape(q+' '+row['question'])+'</h2><p>'+html.escape(' / '.join(row['options']))+'</p>')
        case=directory/q;case.mkdir(exist_ok=True)
        for label,path in [('RD16',OLD_QWEN/'results'/METHODS[2]/(q+'.json'))]+[(m+'32',run/'results'/m/(q+'.json')) for m in METHODS]:
            item=read_json(path);dest=case/(label+'.jpg')
            if not dest.exists():
                images,hashes=FrameProvider(item['selection']).decode();assert hashes==item['rgb_sha256']
                board=Image.new('RGB',(4*320,((len(images)+3)//4)*210),'white');draw=ImageDraw.Draw(board)
                for i,(im,f) in enumerate(zip(images,item['selection']['selected_frames'])):
                    im.thumbnail((316,180));x=(i%4)*320;y=(i//4)*210
                    board.paste(im,(x,y));draw.text((x,y+181),f"{i+1}: {f['timestamp_seconds']:.3f}s / src {f['source_frame_index']}",fill='black');im.close()
                board.save(dest,quality=90);board.close()
            page.append(f'<h3>{html.escape(label)}: {html.escape(item["answer"]["raw_output"])} / correct={item["correct"]}</h3>')
            page.append(f'<a href="{q}/{label}.jpg"><img style="max-width:100%" src="{q}/{label}.jpg"></a>')
    (directory/'index.html').write_text('\n'.join(page),encoding='utf-8')
