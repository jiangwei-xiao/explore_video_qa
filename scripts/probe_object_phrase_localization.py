"""固定颜色问句的对象短语定位探针；全粗池评分，不运行最终问答。"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import re
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.query_input_probe import BASE
from videoqa_runtime.baseline_selection import BlipItmScorer,iter_candidate_frames

RUN=ROOT/'outputs/diagnostics/object_phrase_blip5_20261004_r1'
MODEL=Path('/home/models/Salesforce/blip-itm-base-coco')
PATTERN=re.compile(r'^What colo[u]?r (?:is|are) (.+?)\?$',re.I)


def object_phrase(question):
    """接口：仅处理完整匹配的颜色问句，保留所有对象限定语，不猜答案或补充实体。"""
    match=PATTERN.fullmatch(question)
    return match.group(1) if match else None


def invoke(scorer,text,images,path,fingerprint):
    """接口：每批不超过16，调用前记账；恢复不重复完成项，未知前向即停止。"""
    if path.exists():
        r=read_json(path)
        if r['status']!='completed' or r['fingerprint']!=fingerprint or r['text']!=text or r['frames']!=len(images):
            raise ValueError('Unsafe batch recovery')
        return r['scores']
    encoded=scorer.tokenize(text)
    durable(path,dict(status='started',fingerprint=fingerprint,text=text,frames=len(images),utc=utc()))
    values,pre,forward=scorer.score_batch(encoded,images)
    durable(path,dict(status='completed',fingerprint=fingerprint,text=text,frames=len(images),scores=values,
        tokens=int(encoded['input_ids'].shape[1]),preprocess_seconds=pre,forward_seconds=forward,utc=utc()))
    return values


def main():
    """接口：原名单自动匹配5题，1657短语评分＋80原题控制＋32合成，QA为0。"""
    from videoqa_budget32.selection import configure_visual_precision
    from videoqa_runtime.baseline_run import available_gpus
    with exclusive(RUN/'run.lock'):
        manifest=ROOT/'data/manifests/video_mme_development_200_20260920.json'
        rows=[r for r in read_json(manifest)['rows'] if object_phrase(r['question']) is not None]
        ids=[r['question_id'] for r in rows]
        if ids!=['384-3','438-3','541-1','205-3','241-1']:
            raise ValueError('Fixed question-template population changed')
        pools={q:read_json(BASE/'pools'/(q+'.json')) for q in ids}
        if len(ids)!=5 or sum(p['initial_count'] for p in pools.values())!=1657:raise ValueError('Score budget changed')
        # 步骤1：问题文本规则先于历史预测，查询中没有选项或未知颜色。
        paths=[Path(__file__),manifest,ROOT/'src/videoqa_runtime/baseline_selection.py']
        paths += [BASE/'pools'/(q+'.json') for q in ids]+[p for p in MODEL.iterdir() if p.is_file()]
        protocol=dict(sources={str(p):sha256(p) for p in paths},
            queries=[dict(question_id=r['question_id'],original=r['question'],query=object_phrase(r['question'])) for r in rows],
            max_real_phrase_pairs=1657,max_real_control_pairs=80,max_synthetic_pairs=32,new_final_qa=0,
            batch_size=16,precision='FP32 both TF32 off',candidate_protocol='1FPS phase 0.25 actual PTS',
            boundary='仅此文本模板的定位能力；不声称短语保留完整问答需求，不按结果扩展规则')
        pp=RUN/'protocol.json'
        if pp.exists():
            if read_json(pp)!=protocol:raise ValueError('Probe protocol changed')
        else:durable(pp,protocol)
        fp=sha256(pp)
        for path in (RUN/'calls').rglob('*.json'):
            if read_json(path)['status']!='completed':raise ValueError('Unknown prior forward')
        if (RUN/'summary.json').exists():
            for p,h in read_json(RUN/'summary.json')['assets'].items():
                if sha256(p)!=h:raise ValueError('Completed assets changed')
            print('Completed resume; new forwards 0',flush=True);return
        free,_=available_gpus(['1'])
        if '1' not in free:raise RuntimeError('GPU1 not idle')
        offline_environment()
        import torch
        from PIL import Image
        configure_visual_precision(torch);torch.set_num_threads(2);torch.set_num_interop_threads(1)
        scorer=BlipItmScorer(MODEL)
        # 步骤2：合成单列，首个原题16帧批次必须复现旧评分，再允许该题短语评分。
        images=[Image.new('RGB',(1280,720),'gray') for _ in range(16)]
        for k in range(2):invoke(scorer,'Which synthetic color is visible?',images,RUN/'calls/synthetic'/f'{k}.json',fp)
        for im in images:im.close()
        for row in rows:
            q=row['question_id'];pool=pools[q];values=[];images=[];control_error=None;count=0
            def flush():
                nonlocal images,control_error
                if not values:
                    original=invoke(scorer,row['question'],images,RUN/'calls/control'/(q+'.json'),fp)
                    control_error=max(abs(a-b) for a,b in zip(original,pool['initial_scores'][:len(images)]))
                    if len(images)!=16 or control_error>1e-6:raise ValueError('Original ITM batch mismatch')
                scores=invoke(scorer,object_phrase(row['question']),images,RUN/'calls'/q/(str(len(values))+'.json'),fp)
                values.extend(scores)
                for im in images:im.close()
                images=[]
            metadata={}
            for i,(frame,raw) in enumerate(iter_candidate_frames(pool['video']['path'],metadata)):
                if i>=pool['initial_count']:raise ValueError('Unexpected coarse frame')
                old=pool['candidates'][i]
                if (frame['source_pts'],frame['source_frame_index'])!=(old['source_pts'],old['source_frame_index']):
                    raise ValueError('Coarse source identity differs')
                images.append(raw.to_image());count+=1
                if len(images)==16:flush()
            if images:flush()
            if count!=pool['initial_count'] or len(values)!=count:raise ValueError('Incomplete full coarse scoring')
            order=sorted(range(count),key=lambda i:(-values[i],pool['candidates'][i]['source_pts'],i))
            old_order=sorted(range(count),key=lambda i:(-pool['initial_scores'][i],pool['candidates'][i]['source_pts'],i))
            # 步骤3：只报告排名与入口重合，不生成新的最终选择或沿用旧预测。
            durable(RUN/'results'/(q+'.json'),dict(question_id=q,query=object_phrase(row['question']),scores=values,
                rank_order=order,original_rank_order=old_order,top32_overlap=len(set(order[:32])&set(old_order[:32])),
                control_max_error=control_error,new_final_qa=0))
            durable(RUN/'status.json',dict(status='running',question_id=q,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            print(q,'scored',count,'top32 overlap',len(set(order[:32])&set(old_order[:32])),flush=True)
        assets={str(p):sha256(p) for d in ['calls','results'] for p in (RUN/d).rglob('*.json')}
        durable(RUN/'summary.json',dict(status='completed',questions=5,phrase_pairs=1657,control_pairs=80,
            synthetic_pairs=32,new_final_qa=0,assets=assets,utc=utc()))
        durable(RUN/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        durable(RUN/'status.json',dict(status='paused',error=repr(exc),utc=utc()));raise
