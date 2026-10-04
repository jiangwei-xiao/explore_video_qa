"""两题固定空间尺度诊断：统一五裁剪，不圈定答案物体，不改变正式选帧。"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.query_input_probe import BASE
from videoqa_budget32.frames import FrameProvider
from videoqa_runtime.baseline_selection import BlipItmScorer

CASES={'409-3':[69,77,80,94,99],'556-1':[16,17,42,47,48]}
MODEL=Path('/home/models/Salesforce/blip-itm-base-coco')


def crop_boxes(width,height):
    """接口：固定半宽半高四角和中心五窗；坐标与问题、答案及目标物体无关。"""
    if min(width,height)<2:raise ValueError('图像太小')
    w,h=width//2,height//2;x,y=(width-w)//2,(height-h)//2
    return [(0,0,w,h),(width-w,0,width,h),(0,height-h,w,height),
            (width-w,height-h,width,height),(x,y,x+w,y+h)]


def invoke(scorer,text,images,path,fingerprint):
    """接口：批次前记账，未知前向不重试；原题不截断，最多16图。"""
    if path.exists():
        r=read_json(path)
        if r['status']!='completed' or r['fingerprint']!=fingerprint:raise ValueError('不安全的恢复')
        return r
    encoded=scorer.tokenize(text)
    durable(path,dict(status='started',fingerprint=fingerprint,count=len(images),utc=utc()))
    scores,pre,forward=scorer.score_batch(encoded,images)
    result=dict(status='completed',fingerprint=fingerprint,count=len(images),scores=scores,
                preprocessing_seconds=pre,forward_seconds=forward,token_count=int(encoded['input_ids'].shape[1]))
    durable(path,result);return result


def main(run):
    """接口：96张历史批次重放＋50裁剪评分，2批合成另计，问答0。"""
    from PIL import Image
    import torch
    from videoqa_runtime.baseline_run import available_gpus
    with exclusive(run/'run.lock'):
        # 步骤1：冻结两题及负对照，模型仅接收原问题，不接收选项或答案。
        manifest=ROOT/'data/manifests/video_mme_full_2700.json'
        questions={r['question_id']:r['question'] for r in read_json(manifest)['rows']}
        paths={q:BASE/'pools'/(q+'.json') for q in CASES}
        pools={q:read_json(p) for q,p in paths.items()}
        sources={str(p):sha256(p) for p in [Path(__file__),manifest,*paths.values(),ROOT/'src/videoqa_runtime/baseline_selection.py']}
        for p in MODEL.iterdir():
            if p.is_file():sources[str(p)]=sha256(p)
        spec=dict(sources=sources,cases=CASES,questions={q:questions[q] for q in CASES},
            crop_rule='half-width half-height four corners plus centered; integer floor',
            control_rule='exact historical coarse batches of 16',batch_size=16,precision='float32',
            matmul_allow_tf32=False,cudnn_allow_tf32=False,
            max_real_image_text_pairs=146,max_synthetic_pairs=32,max_qa_calls=0,
            scope='事后固定目标及干扰对照；最大裁剪分数仅用于诊断，不作为新选择规则')
        pp=run/'protocol.json'
        if pp.exists():
            if read_json(pp)!=spec:raise ValueError('协议变化')
        else:durable(pp,spec)
        fp=sha256(pp)
        if (run/'summary.json').exists():
            summary=read_json(run/'summary.json')
            for path,digest in summary['call_hashes'].items():
                if sha256(path)!=digest:raise ValueError('完成调用记录变化')
            print('completed; no new forward');return
        free,_=available_gpus(['1'])
        if '1' not in free:raise RuntimeError('GPU1不空闲')
        offline_environment();torch.set_num_threads(2);torch.manual_seed(2027)
        # 原批次关闭TF32；独立加载BLIP不能依赖先加载Qwen时留下的全局设置。
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        scorer=BlipItmScorer(str(MODEL))
        durable(run/'status.json',dict(status='running',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤2：合成预热单列，真实控制必须重现原批次概率，容差1e-6不放宽。
        synthetic=Image.new('RGB',(1280,720),'gray')
        for i in range(2):invoke(scorer,'Which synthetic color is visible?', [synthetic]*16,run/'synthetic'/f'{i}.json',fp)
        records=[]
        for q,ids in CASES.items():
            pool=pools[q];full_scores={}
            for start in sorted({i//16*16 for i in ids}):
                frames=pool['candidates'][start:min(start+16,pool['initial_count'])]
                images,_=FrameProvider(dict(video=pool['video'],selected_frames=frames)).decode()
                values=invoke(scorer,questions[q],images,run/'controls'/f'{q}_{start}.json',fp)['scores']
                for j,value in enumerate(values):
                    if abs(value-pool['initial_scores'][start+j])>1e-6:raise ValueError('历史整帧评分未复现')
                    full_scores[start+j]=value
                for im in images:im.close()
            crops=[];metadata=[]
            for i in ids:
                images,hashes=FrameProvider(dict(video=pool['video'],selected_frames=[pool['candidates'][i]])).decode()
                for k,box in enumerate(crop_boxes(*images[0].size)):
                    crop=images[0].crop(box);path=run/'views'/f'{q}_{i}_{k}.png'
                    path.parent.mkdir(parents=True,exist_ok=True);crop.save(path)
                    crops.append(crop);metadata.append(dict(question_id=q,candidate_index=i,view=k,box=list(box),
                        source_frame=pool['candidates'][i],original_rgb_sha256=hashes[0],crop_sha256=sha256(path)))
                images[0].close()
            # 步骤3：所有固定裁剪同样评分；不根据输出追加尺度或手工微调裁剪。
            for start in range(0,len(crops),16):
                values=invoke(scorer,questions[q],crops[start:start+16],run/'crops'/f'{q}_{start}.json',fp)['scores']
                for row,value in zip(metadata[start:start+16],values):records.append(dict(row,score=value,full_score=full_scores[row['candidate_index']]))
            for image in crops:image.close()
        calls=list((run/'controls').glob('*.json'))+list((run/'crops').glob('*.json'))+list((run/'synthetic').glob('*.json'))
        counts={group:sum(read_json(p)['count'] for p in (run/group).glob('*.json')) for group in ['controls','crops','synthetic']}
        if counts!={'controls':96,'crops':50,'synthetic':32}:raise ValueError('调用规模不符')
        durable(run/'summary.json',dict(records=records,counts=counts,qa_calls=0,call_hashes={str(p):sha256(p) for p in calls}))
        durable(run/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        for q,ids in CASES.items():
            for i in ids:
                rr=[r for r in records if r['question_id']==q and r['candidate_index']==i]
                print(q,i,rr[0]['full_score'],[(r['view'],round(r['score'],6)) for r in rr],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    main(parser.parse_args().run)
