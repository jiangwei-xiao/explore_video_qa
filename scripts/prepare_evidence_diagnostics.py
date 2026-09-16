"""从已冻结候选池构建原输入/人工修复输入；仅CPU预览，确认前不调用问答。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_audit.core import read,digest,selected
from videoqa_runtime.common import write_json
from videoqa_runtime.video import decode_selected


def validate_case(case, pool):
    """核对16帧唯一、有序、同候选池，且实际改动为1至4张；不接受额外图像。"""
    old=case['original_frames'];new=case['repaired_frames'];valid={r['source_pts']:r for r in pool}
    for rows in (old,new):
        pts=[r['source_pts'] for r in rows]
        if len(pts)!=16 or len(set(pts))!=16 or pts!=sorted(pts):raise ValueError('不是16张有序唯一帧')
        if any(r!=valid.get(r['source_pts']) for r in rows):raise ValueError('帧不属于原候选池')
    replaced=len({r['source_pts'] for r in old}-{r['source_pts'] for r in new})
    if not 1<=replaced<=4:raise ValueError('替换必须为1至4帧')


def prepare(source,out,spec):
    """把人工提案解析成精确PTS与可视预览，冻结前保留proposed状态。"""
    from PIL import Image,ImageDraw
    from transformers import SiglipImageProcessor
    import numpy as np
    source=Path(source).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('已有冻结清单，禁止改写')
    processor=SiglipImageProcessor.from_pretrained('/home/models/google/siglip-so400m-patch14-384',local_files_only=True)
    cases=[]
    # 步骤1：读历史结果，不利用任何新问答结果选案例或挑替换帧。
    for item in spec['cases']:
        qid=item['question_id'];m=item['method']
        historical={}
        for method in ('uniform','topk','A','B','C','D'):
            folder=source/('baseline_reference/results' if method in ('uniform','topk') else 'results')/method
            historical[method]=read(folder/f'{qid}.json')
        flags=[r['correct'] for r in historical.values()]
        category='all_correct' if all(flags) else 'all_wrong' if not any(flags) else 'discordant'
        expected='B' if category=='all_correct' or not historical['B']['correct'] else next(x for x in ('C','topk','uniform','A','D') if not historical[x]['correct'])
        if category!=item['stratum'] or expected!=m:raise ValueError('未遵守预定类别或方法选择顺序')
        path=source/('baseline_reference/results' if m in ('uniform','topk') else 'results')/m/f'{qid}.json'
        r=read(path);pool=r.get('pool',r.get('selection'));old=selected(r)
        removed=[old[i-1] for i in item['remove_slots']];added=[pool['candidates'][i] for i in item['add_candidates']]
        new=sorted([f for f in old if f not in removed]+added,key=lambda f:f['source_pts'])
        case=dict(item,order_index=r['order_index'],question=r['question'],options=r['options'],reference=r['reference_answer'],
                  historical_prediction=r['answer']['parsed_answer'],historical_correct=r['correct'],historical_prompt=r['answer']['prompt'],
                  video=pool['video'],video_sha256=r['source_video_sha256'],source_result=str(path),source_result_sha256=digest(path),
                  original_frames=old,repaired_frames=new,removed_frames=removed,added_frames=added)
        validate_case(case,pool['candidates'])
        # 步骤2：输出实际官方384预处理的替换前后画面，供人工确认新增证据可读。
        root=out/'preview'/qid;root.mkdir(parents=True,exist_ok=True)
        rows=sorted({f['source_pts']:f for f in removed+added}.values(),key=lambda f:f['source_pts'])
        images=dict(zip((f['source_pts'] for f in rows),decode_selected(pool['video'],rows)))
        board=Image.new('RGB',(768,410*len(removed)),'white');draw=ImageDraw.Draw(board)
        for j,pair in enumerate(zip(removed,added)):
            for col,f in enumerate(pair):
                im=images[f['source_pts']];im.save(root/f'{f["source_pts"]}.original.png')
                pixels=processor(images=im,return_tensors='np')['pixel_values'][0]
                rgb=pixels.transpose(1,2,0)*np.asarray(processor.image_std)+np.asarray(processor.image_mean)
                view=Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype('uint8'));view.save(root/f'{f["source_pts"]}.input.png')
                board.paste(view,(384*col,410*j));draw.text((384*col+3,410*j+386),f'{"REMOVE" if col==0 else "ADD"} {f["timestamp_seconds"]:.3f}s',fill='black')
        board.save(root/'changes.png');cases.append(case)
    # 步骤3：提案和筛选理由先落盘；manifest须在看过全部预览后另行冻结。
    write_json(out/'proposed_manifest.json',dict(version='evidence-repair-v1',status='proposed',maximum_started_calls=20,
               source=str(source),cases=sorted(cases,key=lambda c:c['order_index']),screening_notes=spec['screening_notes'],
               planned_calls=2*len(cases),algorithm_evaluation=False,scope_calls=0,blip_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--spec',required=True)
    a=p.parse_args();prepare(a.source,a.out,read(a.spec))
