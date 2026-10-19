"""新数据四冻结策略准备及图文融合表示；价值特征不反馈改变选帧。"""
from contextlib import contextmanager
from pathlib import Path
import copy
import fcntl
import hashlib
import os
import time
import numpy as np
from PIL import Image

from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import durable,digest,save_array,utc
from videoqa_methods.density_pipeline import DensityScorer,scan_original,decode_regions
from videoqa_methods.density_selection import build_regions
from videoqa_methods.budget_contract import budget_config,select_exp07
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_runtime.baseline_selection import select_topk
from videoqa_budget32.frames import FrameProvider,native_frames,validate_frames
from videoqa_set_value.data import METHODS
from videoqa_set_value.learner import extract_set_features

BLIP=Path('/home/models/Salesforce/blip-itm-base-coco')


def save_npz(path,values):
    """接口：紧凑数值资产原子保存，禁止pickle和模型文字混入特征。"""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.npz.part')
    with temp.open('wb') as stream:np.savez(stream,**values)
    temp.replace(path)


@contextmanager
def batch_lock(path):
    """接口：共享同输入缓存可以等待；进程异常留下的未确认前向不自动重复。"""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


class ValueScorer(DensityScorer):
    """接口：冻结FP32 BLIP同次输出ITM、视觉CLS和问题融合CLS，分别用于选择/学习。"""
    def __init__(self,root):
        super().__init__(BLIP);self.root=Path(root);self.fused_blocks=[]

    def _forward(self,encoded,images,kind):
        """步骤1输入与账本；步骤2一次前向；步骤3原始数值检查点先于下游选帧。"""
        if not 1<=len(images)<=16:raise ValueError('Frozen BLIP batch16 exceeded')
        torch=self.torch
        rgb=[hashlib.sha256(np.ascontiguousarray(np.asarray(im.convert('RGB'))).tobytes()).hexdigest() for im in images]
        text={k:v.tolist() for k,v in (encoded or {}).items() if k in ('input_ids','attention_mask')}
        key=digest(dict(kind=kind,rgb=rgb,text=text));cp=self.root/'blip_calls'/kind/(key+'.json');array=cp.with_suffix('.npz')
        with batch_lock(cp.with_suffix('.lock')):
            if cp.exists():
                old=read_json(cp)
                if old['status']!='completed' or sha256(array)!=old['array_sha256']:
                    raise RuntimeError('Uncertain BLIP forward; inspect saved batch before reuse')
                data=np.load(array,allow_pickle=False)
                return {k:data[k].copy() for k in data.files},0.,0.
            torch.cuda.synchronize();start=time.perf_counter()
            pixels=self.processor(images=images,return_tensors='pt')['pixel_values'].to('cuda',torch.float32)
            inputs={k:v.repeat(len(images),1).cuda() for k,v in (encoded or {}).items() if k in ('input_ids','attention_mask')}
            torch.cuda.synchronize();pre=time.perf_counter()-start
            durable(cp,dict(status='started',kind=kind,source_rgb_sha256=rgb,text_tokens=text,pid=os.getpid(),utc=utc()))
            start=time.perf_counter()
            with torch.inference_mode():
                if kind=='itm':
                    output=self.model(pixel_values=pixels,**inputs,use_itm_head=True,return_dict=True)
                    cls=output.last_hidden_state[:,0,:].float();fused=output.question_embeds[:,0,:].float()
                    values=dict(scores=output.itm_score.softmax(-1)[:,1].cpu().numpy(),
                        visual=torch.nn.functional.normalize(cls,dim=-1).cpu().numpy(),
                        fused=torch.nn.functional.normalize(fused,dim=-1).cpu().numpy())
                else:
                    cls=self.model.vision_model(pixel_values=pixels,return_dict=True).last_hidden_state[:,0,:].float()
                    values=dict(visual=torch.nn.functional.normalize(cls,dim=-1).cpu().numpy())
            torch.cuda.synchronize();fw=time.perf_counter()-start
            if any(not np.isfinite(v).all() for v in values.values()) or values['visual'].shape!=(len(images),768):raise ValueError('Invalid BLIP representation')
            if kind=='itm' and (values['fused'].shape!=(len(images),768) or np.any((values['scores']<0)|(values['scores']>1))):raise ValueError('Invalid ITM/fused shape')
            save_npz(array,values)
            durable(cp,dict(status='completed',kind=kind,source_rgb_sha256=rgb,text_tokens=text,
                array_sha256=sha256(array),preprocess_seconds=pre,forward_seconds=fw,frames=len(images),utc=utc()))
            return values,pre,fw

    def score_batch_features(self,encoded,images):
        """接口：原选择器仍只得到原分数/纯视觉CLS；融合CLS旁路保存供学习。"""
        values,pre,fw=self._forward(encoded,images,'itm');self.fused_blocks.append(values['fused'])
        return values['scores'].tolist(),values['visual'],pre,fw

    def visual_batch(self,images):
        """接口：补查的选择特征仍只计算纯视觉，不构造选项/答案或新增选择分数。"""
        values,pre,fw=self._forward(None,images,'visual');return values['visual'],pre,fw


def prepare_one(root,row,scorer):
    """接口：原问题与原视频→四个K8输入、原轨迹及独立价值特征，不使用标准答案。"""
    root=Path(root);q=row['question_id'];cp=root/'selection_checkpoints'/(q+'.json')
    if cp.exists():
        old=read_json(cp)
        for path,fp in old['files'].items():
            if sha256(root/path)!=fp:raise ValueError('Completed selection asset changed')
        return old
    began=time.perf_counter();scorer.fused_blocks=[]
    # 步骤1：全片原1FPS网格与原FP32批次；同次融合向量只旁路记录。
    coarse,visual,timing=scan_original(row['video_path'],row['question'],scorer)
    fused=np.concatenate(scorer.fused_blocks);video=coarse['video'];cfg=budget_config(8)
    index=read_json(row['index_path']);pts=np.load(index['pts_path'],allow_pickle=False)
    if video!=index['video'] or len(coarse['candidates'])<8:raise ValueError('Frozen source/coarse eligibility mismatch')
    plan=build_regions(coarse['candidates'],coarse['scores'],video,cfg)
    temporary=root/'temporary_refinement'/q
    added,assets,refinement,refine_time=decode_regions(video,coarse['candidates'],plan,cfg,temporary)
    blocks=[visual]
    for offset in range(0,len(assets),16):
        ims=[]
        for asset in assets[offset:offset+16]:
            with Image.open(asset['path']) as im:ims.append(im.convert('RGB'))
        try:features,_,_=scorer.visual_batch(ims);blocks.append(features)
        finally:
            for im in ims:im.close()
    all_visual=np.concatenate(blocks)
    pool=dict(video=video,candidates=coarse['candidates']+added,initial_count=len(coarse['candidates']),
        initial_scores=coarse['scores'],plan=plan,refinement=refinement,new_frames_have_itm_scores=False)
    a,trace=select_exp07(pool,all_visual,8);b,weighted=select_relevance_coverage(pool,all_visual,trace)
    chosen={METHODS[0]:native_frames(pts,video,8),METHODS[1]:select_topk(coarse['candidates'],coarse['scores'],8),
        METHODS[2]:[pool['candidates'][i] for i in a],METHODS[3]:[pool['candidates'][i] for i in b]}
    for frames in chosen.values():
        if len(frames)!=8:raise ValueError('Every candidate set must have eight unique frames')
        validate_frames(dict(video=video,selected_frames=frames),8)
    # 步骤2：只对最终四组的源帧并集存PNG，候选池不保存全量图片。
    source_key=lambda f:(f['source_pts'],f['source_frame_index'])
    union={source_key(f):f for frames in chosen.values() for f in frames};ordered=sorted(union)
    images,rgb=FrameProvider(dict(video=video,selected_frames=[union[k] for k in ordered])).decode()
    image_map={};value_map={source_key(f):(fused[i],coarse['scores'][i]) for i,f in enumerate(coarse['candidates'])}
    try:
        for k,im,rgb_sha in zip(ordered,images,rgb):
            path=root/'selected_rgb'/q/(str(k[0])+'_'+str(k[1])+'.png');path.parent.mkdir(parents=True,exist_ok=True);im.save(path)
            image_map[k]=dict(image=str(path),rgb_sha256=rgb_sha)
        missing=[i for i,k in enumerate(ordered) if k not in value_map];encoded=scorer.tokenize(row['question'])
        for offset in range(0,len(missing),16):
            ids=missing[offset:offset+16];values,_,_=scorer._forward(encoded,[images[i] for i in ids],'itm')
            for j,i in enumerate(ids):value_map[ordered[i]]=(values['fused'][j],float(values['scores'][j]))
    finally:
        for im in images:im.close()
    # 步骤3：新增ITM仅进入独立价值特征；四个原选择和轨迹不反馈重算。
    value={};records={};files=[]
    for method,frames in chosen.items():
        source=[source_key(f) for f in frames]
        value[method]=extract_set_features([value_map[k][0] for k in source],[value_map[k][1] for k in source],
            [f['timestamp_seconds'] for f in frames],video['duration_seconds'])
        record=dict(question_id=q,method=method,selection=dict(video=video,selected_frames=frames),
            frames=[dict(copy.deepcopy(f),**image_map[source_key(f)]) for f in frames])
        path=root/'selections'/method/(q+'.json');durable(path,record);files.append(path)
        records[method]=dict(path=str(path),sha256=sha256(path))
    pp=root/'pools'/(q+'.json');fp=root/'visual_features'/(q+'.npy');tp=root/'traces'/(q+'.json');vp=root/'value_features'/(q+'.npz')
    durable(pp,pool);save_array(fp,all_visual);durable(tp,dict(exp07=trace,exp12=weighted));save_npz(vp,value)
    files.extend([pp,fp,tp,vp]);files.extend(Path(v['image']) for v in image_map.values())
    result=dict(question_id=q,files={str(p.relative_to(root)):sha256(p) for p in files},selections=records,
        timing=dict(**timing,refinement=refine_time,wall_seconds=time.perf_counter()-began),utc=utc())
    durable(cp,result)
    # 步骤4：确认是本题自己创建的暂存文件后清理；失败现场不会走到此处。
    for asset in assets:
        p=Path(asset['path'])
        if p.parent!=temporary or sha256(p)!=asset['sha256']:raise ValueError('Unexpected transient path')
        p.unlink()
    return result
