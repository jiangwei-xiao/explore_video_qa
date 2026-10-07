"""固定四题的开放词汇物体定位能力检查；不生成答案、不修改当前方法。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys
import argparse
import time
import hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,rows,utc,save_array
from videoqa_budget32.frames import FrameProvider

MODEL=ROOT/'outputs/model_cache/grounding_dino_tiny_a2bb814'
BASE=ROOT/'outputs/full_videoqa/uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01'
CASES={
    '409-3':dict(indices=[77,80,94],caption='a man with sunglasses. a gun. a motorcycle. a jacket. a box of roses. a person.'),
    '556-1':dict(indices=[42,61],caption='a person. a bicycle. a skateboard. a car.'),
    '383-2':dict(indices=[944,887],caption='a quantum computer. a horse. a boat. a person. a car.'),
    '103-1':dict(indices=[64,7],caption='a person. a fork. chopsticks. a box of noodles. a spoon. a trash can.')
}


def prepare(root):
    """接口：固定9真实图与4空白控制、全部选项物体查询，标签只用于能力评价。"""
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():return verify(root)
    source={str(Path(__file__)):sha256(__file__)};inputs={};data={r['question_id']:r for r in rows()}
    from PIL import Image
    for q,case in CASES.items():
        pp=BASE/'pools'/(q+'.json');source[str(pp)]=sha256(pp);pool=read_json(pp)
        frames=sorted([pool['candidates'][i] for i in case['indices']],key=lambda f:f['source_pts'])
        images,hashes=FrameProvider(dict(video=pool['video'],selected_frames=frames)).decode()
        files=[]
        # 步骤1：保留完整源图，不手工裁到答案；所有备选对象都列入同一个查询。
        for i,image in enumerate(images):
            path=root/'inputs'/q/f'{i}.png';path.parent.mkdir(parents=True,exist_ok=True);image.save(path);image.close()
            source[str(path)]=sha256(path);files.append(str(path))
        blank=root/'inputs'/q/'blank.png';Image.new('RGB',(1280,720),'white').save(blank)
        source[str(blank)]=sha256(blank)
        inputs[q]=dict(question=data[q]['question'],options=data[q]['options'],frames=frames,
            rgb_sha256=hashes,files=files,blank=str(blank),caption=case['caption'])
    # 步骤2：官方固定提交和LFS摘要校验；只写项目缓存，不碰共享模型或依赖。
    download=read_json(MODEL/'download_manifest.json')
    if download['revision']!='a2bb814dd30d776dcf7e30523b00659f4f141c71':raise ValueError('Detector checkpoint revision differs')
    for name,entry in download['files'].items():
        path=MODEL/name
        if sha256(path)!=entry['sha256']:raise ValueError('Downloaded model bytes differ')
        source[str(path)]=entry['sha256']
    source[str(MODEL/'download_manifest.json')]=sha256(MODEL/'download_manifest.json')
    import inspect
    from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoForObjectDetection
    from transformers.models.grounding_dino.processing_grounding_dino import GroundingDinoProcessor
    for cls in (GroundingDinoForObjectDetection,GroundingDinoProcessor):
        path=inspect.getfile(cls);source[path]=sha256(path)
    spec=dict(inputs=inputs,sources=source,real_image_limit=9,synthetic_blank_limit=4,qa_calls=0,
        box_threshold=.4,text_threshold=.3,dtype='float32',batch_size=1,disable_custom_kernels=True,
        scope='Manual diagnostic noun normalization from original question/all alternatives; not an automatic query extractor or answer predictor',
        gates='At least two videos localize visible needed small objects with boxes, no confident hallucinated target on blanks; do not require every object type',utc=utc())
    import shutil
    dest=root/'code_snapshot'/Path(__file__).name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(__file__,dest)
    durable(root/'protocol.json',spec);return spec


def verify(root):
    """接口：模型、图像和查询全部绑定，未知返回不重复推理，不扫描阈值。"""
    spec=read_json(root/'protocol.json')
    for p,h in spec['sources'].items():
        if sha256(p)!=h:raise ValueError('Frozen grounded object source changed: '+p)
    return spec


def invoke(root,name,image,caption,processor,model,spec):
    """接口：调用前记账、完整logits/框先落盘；输出阈值沿用官方示例，保留原始结果。"""
    import torch
    path=root/'calls'/(name+'.json');raw=root/'calls'/(name+'.npz')
    identity=dict(caption=caption,rgb_sha256=hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),protocol_sha256=sha256(root/'protocol.json'))
    if path.exists():
        old=read_json(path)
        if old['identity']!=identity or old['status']!='completed' or sha256(raw)!=old['raw_sha256']:
            raise ValueError('Uncertain detection; repetition forbidden')
        return old
    inputs=processor(images=image,text=caption,return_tensors='pt')
    tokens=inputs['input_ids'][0].tolist()
    if len(tokens)>256:raise ValueError('Object query exceeds detector text limit')
    inputs=inputs.to('cuda')
    # 步骤1：模型只返回物体框，不使用VLM生成或数据集标准答案。
    durable(path,dict(status='started_may_forward',identity=identity,token_ids=tokens,utc=utc()))
    torch.cuda.synchronize();started=time.perf_counter()
    with torch.inference_mode():output=model(**inputs)
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    # 官方以-inf屏蔽未使用文本位置，sigmoid为0；不能把正常padding当作模型失败。
    if torch.isnan(output.logits).any() or torch.isposinf(output.logits).any() or not torch.isfinite(output.pred_boxes).all():
        raise ValueError('Invalid detection return')
    valid_mask=inputs['attention_mask'].bool()
    valid_logits=output.logits[0,:,:valid_mask.shape[1]][:,valid_mask[0]]
    if not torch.isfinite(valid_logits).all():raise ValueError('Non-finite valid text logits')
    raw.parent.mkdir(parents=True,exist_ok=True);temporary=raw.with_suffix('.tmp.npz')
    np.savez(temporary,logits=output.logits.cpu().numpy(),boxes=output.pred_boxes.cpu().numpy());os.replace(temporary,raw)
    # 步骤2：4.50接口参数名是threshold；0.4/0.3固定，不按漏检改变阈值。
    detections=processor.post_process_grounded_object_detection(output,inputs.input_ids,
        threshold=spec['box_threshold'],text_threshold=spec['text_threshold'],target_sizes=[image.size[::-1]])[0]
    result=dict(status='completed',identity=identity,raw_sha256=sha256(raw),token_ids=tokens,
        boxes=detections['boxes'].cpu().tolist(),scores=detections['scores'].cpu().tolist(),labels=detections['text_labels'],
        source_size=list(image.size),processed_size=list(inputs['pixel_values'].shape),seconds=elapsed,qa_calls=0,utc=utc())
    durable(path,result);return result


def execute(root,gpu):
    """接口：单卡固定13次识别，空白控制按题前后交替；不改变选帧或问答后端。"""
    offline_environment();spec=verify(root)
    if (root/'summary.json').exists():
        saved=read_json(root/'summary.json')
        for p,h in saved['artifacts'].items():
            if sha256(p)!=h:raise ValueError('Completed grounding artifacts differ')
        return saved
    import torch
    from PIL import Image,ImageDraw
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
    torch.set_num_threads(2);torch.manual_seed(2027);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with exclusive(root/'run.lock'):
        processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        model=AutoModelForZeroShotObjectDetection.from_pretrained(MODEL,local_files_only=True,
            torch_dtype=torch.float32,disable_custom_kernels=True).cuda().eval().requires_grad_(False)
        durable(root/'environment.json',dict(torch=torch.__version__,transformers=__import__('transformers').__version__,gpu=gpu,
            pid=os.getpid(),sid=os.getsid(0),model_dtype=str(model.dtype),custom_kernels_disabled=model.config.disable_custom_kernels,utc=utc()))
        results={}
        # 步骤1：固定图像顺序和查询，不从识别输出补样本或改成只查正确选项。
        for ordinal,(q,item) in enumerate(spec['inputs'].items()):
            names=[(str(i),p) for i,p in enumerate(item['files'])]
            names=([('blank',item['blank'])]+names) if ordinal%2==0 else (names+[('blank',item['blank'])])
            results[q]={}
            for label,path in names:
                with Image.open(path) as opened:image=opened.convert('RGB')
                r=invoke(root,q+'__'+label,image,item['caption'],processor,model,spec);results[q][label]=r
                # 步骤2：仅绘制返回框供审核，不裁剪输入、不改变模型的框或置信度。
                draw=ImageDraw.Draw(image)
                for box,score,text in zip(r['boxes'],r['scores'],r['labels']):
                    draw.rectangle(box,outline='red',width=3);draw.text((box[0],max(0,box[1]-14)),f'{text}: {score:.3f}',fill='red')
                dest=root/'views'/q/(label+'.jpg');dest.parent.mkdir(parents=True,exist_ok=True);image.save(dest,quality=95);image.close()
                print(q,label,[(text,round(score,3)) for text,score in zip(r['labels'],r['scores'])],flush=True)
        # 步骤3：能力判断独立于答题得分，全部框和空白反例保留。
        calls=list((root/'calls').glob('*.json'))
        if len(calls)!=13:raise ValueError('Detection budget population differs')
        summary=dict(status='completed',results=results,real_image_forwards=9,synthetic_blank_forwards=4,qa_calls=0,
            review_required=True,artifacts={str(p):sha256(p) for p in (root/'calls').iterdir()},utc=utc())
        durable(root/'summary.json',summary);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--stage',choices=('prepare','run'),required=True);parser.add_argument('--gpu',default='7')
    args=parser.parse_args()
    if not args.run_id.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run ID')
    root=ROOT/'outputs/diagnostics'/args.run_id
    try:prepare(root) if args.stage=='prepare' else execute(root,args.gpu)
    except BaseException as exc:
        durable(root/('failure_'+str(time.time_ns())+'.json'),dict(error=repr(exc),utc=utc()));raise
