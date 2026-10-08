"""exp27离线证据对照与固定画面复核页；只读取已有事实和源视频。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import html
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import rows,durable,utc
from videoqa_audit.framework_contraction import RUN8,RUN32
from videoqa_audit.region_scope_experiments import FACTS,METHOD
from videoqa_budget32.frames import FrameProvider

PANEL=('103-1','383-2','395-2','409-3','035-2','844-1')


def witness_report(root):
    """接口：冻结42题75事实的精确见证对照，范围外未标帧不判无关。"""
    root=Path(root);facts=read_json(FACTS)['entries']
    if len(facts)!=75 or len({x['question_id'] for x in facts})!=42:raise ValueError('Fact denominator changed')
    result={}
    # 步骤1：标准答案与人工事实不进入选择器；本接口只评价已保存输出。
    for profile,parent in [('qwen32',RUN32),('llava8',RUN8)]:
        entries=[]
        for fact in facts:
            q=fact['question_id'];old_path=parent/'selections/RD-1.2-exp07'/(q+'.json');new_path=root/'offline'/profile/(q+'.json')
            old=read_json(old_path)['selection']['selected_frames'];new=read_json(new_path)['selection']['selected_frames']
            pts=set(fact['witness_pts']);a=bool(pts & {f['source_pts'] for f in old});b=bool(pts & {f['source_pts'] for f in new})
            entries.append(dict(fact,old_exact_hit=a,new_exact_hit=b,
                sources={str(old_path):sha256(old_path),str(new_path):sha256(new_path)}))
        # 步骤2：按题平均后宏平均，分母固定；保留全部得失而非只显示净值。
        def macro(key):
            return sum(sum(e[key] for e in entries if e['question_id']==q)/sum(e['question_id']==q for e in entries)
                       for q in {e['question_id'] for e in entries})/42
        result[profile]=dict(facts=75,questions=42,old_hits=sum(x['old_exact_hit'] for x in entries),
            new_hits=sum(x['new_exact_hit'] for x in entries),old_macro=macro('old_exact_hit'),new_macro=macro('new_exact_hit'),
            gains=[x for x in entries if x['new_exact_hit'] and not x['old_exact_hit']],
            losses=[x for x in entries if x['old_exact_hit'] and not x['new_exact_hit']],entries=entries)
    value=dict(facts_source=str(FACTS),facts_sha256=sha256(FACTS),profiles=result,new_model_calls=0,
               limitation='Exact listed coarse witnesses only; not semantic sufficiency',utc=utc())
    durable(root/'witness_report.json',value);return value


def draw_board(job):
    """接口：精确源帧图板与RGB身份落盘；缩略图不是官方处理器视图。"""
    root,q,profile,label,path=Path(job[0]),job[1],job[2],job[3],Path(job[4])
    out=root/'review'/profile/q/(label+'.jpg');meta=out.with_suffix('.json')
    if meta.exists():
        old=read_json(meta)
        if old['source_sha256']!=sha256(path) or old['image_sha256']!=sha256(out):raise ValueError('Review artifact changed')
        return old
    from PIL import Image,ImageDraw
    item=read_json(path);selection=item['selection'];images,rgb=FrameProvider(selection).decode()
    if 'rgb_sha256' in item and rgb!=item['rgb_sha256']:raise ValueError('Historical review RGB changed')
    frames=selection['selected_frames'];canvas=Image.new('RGB',(1280,210*((len(frames)+3)//4)),'white');draw=ImageDraw.Draw(canvas)
    # 步骤1：等比缩略原RGB；必要小文字另查原帧或既有模型尺寸视图。
    for i,(im,f) in enumerate(zip(images,frames)):
        im.thumbnail((316,180));x,y=i%4*320,i//4*210;canvas.paste(im,(x,y));im.close()
        draw.text((x,y+181),f"{i+1}: {f['timestamp_seconds']:.3f}s src {f['source_frame_index']}",fill='black')
    out.parent.mkdir(parents=True,exist_ok=True);canvas.save(out,quality=94);canvas.close()
    # 步骤2：画面元数据与问答分开，不写虚构预测或把观察当作真值。
    value=dict(question_id=q,profile=profile,label=label,source=str(path),source_sha256=sha256(path),
        image_sha256=sha256(out),frames=frames,rgb_sha256=rgb,new_model_calls=0)
    durable(meta,value);return value


def render_review(root):
    """接口：固定六题加至多两道按冻结题序出现的见证损失疑点，总复核不超过8题。"""
    root=Path(root);w=witness_report(root);data=rows();byid={r['question_id']:r for r in data}
    loss_ids={x['question_id'] for p in w['profiles'].values() for x in p['losses']}
    extra=[r['question_id'] for r in data if r['question_id'] in loss_ids and r['question_id'] not in PANEL][:2]
    panel=list(PANEL)+extra;durable(root/'review/panel.json',dict(question_ids=panel,maximum=8,review_minutes_limit=45))
    jobs=[]
    for profile,parent in [('qwen32',RUN32),('llava8',RUN8)]:
        for q in panel:
            jobs.extend([(str(root),q,profile,'exp07',str(parent/'selections/RD-1.2-exp07'/(q+'.json'))),
                         (str(root),q,profile,'exp27',str(root/'offline'/profile/(q+'.json')))])
    # 步骤1：只绘制冻结选择，图板由界限确定，不按新回答选有利题。
    with ThreadPoolExecutor(max_workers=4) as executor:records=list(executor.map(draw_board,jobs))
    page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>exp27区域职责对照</title><style>body{font:16px/1.6 sans-serif;margin:24px auto;max-width:1300px}img{max-width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}</style><h1>exp27区域职责对照</h1><p>原RGB缩略图。左exp07，右exp27；尚未进行新问答，图板本身不证明证据充分。</p>']
    for q in panel:
        page+=['<h2>'+q+'</h2><p>'+html.escape(byid[q]['question'])+'</p><p>'+html.escape(' / '.join(byid[q]['options']))+'</p>']
        for profile in ('qwen32','llava8'):
            page+=['<h3>'+profile+'</h3><div class="pair">']
            for label in ('exp07','exp27'):
                url=profile+'/'+q+'/'+label+'.jpg';page+=['<div>'+label+'<a href="'+url+'"><img src="'+url+'"></a></div>']
            page+=['</div>']
    page+=['</html>'];(root/'review/index.html').write_text('\n'.join(page),encoding='utf-8')
    result=dict(question_ids=panel,boards=len(records),records=records,new_model_calls=0,semantic_review='pending')
    durable(root/'review/manifest.json',result);return result
