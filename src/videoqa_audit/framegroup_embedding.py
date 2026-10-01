"""辅助帧组嵌入探针；不接入正式问答或正式选帧，禁止隐式采样和NULL回退。"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from videoqa_runtime.common import sha256

QUERY_INSTRUCTION='Retrieve visual evidence relevant to the query.'
DOCUMENT_INSTRUCTION="Represent the user's input."
MAX_PIXELS=768*32*32
MIN_PIXELS=4*32*32


def unpack_pixels(pixels,grid,patch=16,merge=2,temporal=2):
    """接口：还原官方补丁布局，验证每个实际送入的RGB视图，不修改模型输入。"""
    shape=np.asarray(grid).reshape(-1,3)
    if len(shape)!=1:raise ValueError('Probe accepts one media item per call')
    t,h,w=map(int,shape[0]);x=np.asarray(pixels)
    if min(t,h,w)<=0 or h%merge or w%merge or x.shape!=(t*h*w,3*temporal*patch*patch):raise ValueError('Unexpected Qwen3 patch layout')
    # 步骤1：展开合并块和时间补丁；步骤2：恢复按输入顺序排列的HWC帧。
    x=x.reshape(t,h//merge,w//merge,merge,merge,3,temporal,patch,patch)
    return x.transpose(0,6,1,3,7,2,4,8,5).reshape(t*temporal,h*patch,w*patch,3)


def prepare_view(source_path,destination):
    """接口：一次性等比缩放到官方32像素网格，同一PNG用于单图与帧组两种条件。"""
    from qwen_vl_utils.vision_process import smart_resize
    with Image.open(source_path) as source:
        rgb=source.convert('RGB');h,w=smart_resize(rgb.height,rgb.width,factor=32,min_pixels=MIN_PIXELS,max_pixels=MAX_PIXELS)
        view=rgb.resize((w,h),Image.Resampling.BICUBIC);destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            with Image.open(destination) as old:assert np.array_equal(np.asarray(old.convert('RGB')),np.asarray(view))
        else:view.save(destination)
        return dict(path=str(destination),sha256=sha256(destination),rgb_sha256=hashlib.sha256(np.asarray(view).tobytes()).hexdigest(),height=h,width=w)


class StrictEmbeddingBackend:
    def __init__(self,model_path,expected_source_sha):
        """接口：只读加载随模型的官方前向类，输入层使用独立严格适配，不调用其回退/重采样。"""
        import torch
        from transformers import Qwen3VLProcessor
        sys.dont_write_bytecode=True;model_path=Path(model_path);source=model_path/'scripts/qwen3_vl_embedding.py'
        if sha256(source)!=expected_source_sha:raise ValueError('Embedding implementation changed')
        # 步骤1：已完整review的本地封装只提供模型类；不执行其自动抽帧、截断或NULL处理。
        name='videoqa_frozen_qwen3_embedding';spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        started=time.perf_counter()
        self.model,info=module.Qwen3VLForEmbedding.from_pretrained(str(model_path),local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
        if info.get('missing_keys') or info.get('mismatched_keys') or set(info.get('unexpected_keys',[]))-{'lm_head.weight'}:raise ValueError(('Unexpected weight loading',info))
        self.model=self.model.to('cuda').eval().requires_grad_(False);self.model.config.use_cache=False
        self.processor=Qwen3VLProcessor.from_pretrained(str(model_path),local_files_only=True,padding_side='right')
        self.normalized_images={}
        # 步骤2：精度、注意力和全部加载差异落盘，不宣称和原Qwen2.5问答协议相同。
        impls=dict(text=getattr(self.model.config.text_config,'_attn_implementation',None),vision=getattr(self.model.config.vision_config,'_attn_implementation',None))
        if set(impls.values())!={'sdpa'}:raise ValueError(('Unexpected attention implementation',impls))
        if self.model.dtype!=torch.bfloat16:raise ValueError('Expected BF16 embedding weights')
        self.load_report=dict(seconds=time.perf_counter()-started,loading_info=info,attention=impls,dtype=str(self.model.dtype),
            parameters=sum(p.numel() for p in self.model.parameters()),model_allocated_bytes=torch.cuda.memory_allocated(),source_sha256=expected_source_sha,device_name=torch.cuda.get_device_name(),device_total_bytes=torch.cuda.get_device_properties(0).total_memory)

    def preprocess(self,job,views):
        """接口：固定文本/单图/十帧组输入，禁止截断与二次采样，核验实际补丁和源像素。"""
        import torch
        kind=job['kind'];instruction=QUERY_INSTRUCTION if kind=='text' else DOCUMENT_INSTRUCTION;content=[];images=[]
        for fid in job.get('frame_ids',[]):
            view=views[fid]
            if sha256(view['path'])!=view['sha256']:raise ValueError('Prepared view changed')
            with Image.open(view['path']) as im:images.append(im.convert('RGB'))
        kwargs={}
        # 步骤1：仅使用固定媒体占位，帧组名称、人工判断和标准答案不进入提示词。
        if kind=='text':content=[dict(type='text',text=job['text'])]
        elif kind=='image':
            if len(images)!=1:raise ValueError('Expected exactly one source image')
            content=[dict(type='image',image=images[0])];kwargs['images']=images
        elif kind=='video':
            if len(images)!=10:raise ValueError('Expected ten source images')
            arrays=[np.array(im,dtype=np.uint8,copy=True) for im in images]
            if len({a.shape for a in arrays})!=1:raise ValueError('Video view dimensions differ')
            content=[dict(type='video',video=images)];kwargs['videos']=[torch.stack([torch.from_numpy(a).permute(2,0,1) for a in arrays])]
            kwargs['video_metadata']=[dict(total_num_frames=10,fps=2.,frames_indices=list(range(10)))];kwargs['do_sample_frames']=False
        else:raise ValueError('Unknown modality')
        conversation=[dict(role='system',content=[dict(type='text',text=instruction)]),dict(role='user',content=content)]
        prompt=self.processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=False)
        inputs=self.processor(text=[prompt],padding=True,truncation=False,do_resize=False,return_tensors='pt',**kwargs)
        if inputs['input_ids'].shape[0]!=1 or inputs['input_ids'].shape[1]>8192:raise ValueError('No input truncation permitted')
        ids=inputs['input_ids'][0].tolist();mask=inputs['attention_mask'][0].tolist();active=sum(mask)
        if active<1 or mask!=[1]*active+[0]*(len(mask)-active):raise ValueError('Expected right-padded active-token mask')
        visual_tokens=0;grid=None;pixel_sha=None;max_error=0.
        # 步骤2：还原真实处理器输出，检查源帧数量/顺序及两条件看到的像素一致。
        if kind!='text':
            pixel_key='pixel_values' if kind=='image' else 'pixel_values_videos';grid_key='image_grid_thw' if kind=='image' else 'video_grid_thw'
            pixels=inputs[pixel_key].cpu().numpy();grid=inputs[grid_key].cpu().numpy();restored=unpack_pixels(pixels,grid)
            token=self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>' if kind=='image' else '<|video_pad|>')
            visual_tokens=ids.count(token);assert visual_tokens==int(np.prod(grid[0])//4)
            if kind=='image':
                if len(restored)!=2 or not np.array_equal(restored[0],restored[1]):raise ValueError('Unexpected single-image temporal padding')
                restored=restored[:1]
            elif len(restored)!=10:raise ValueError('Video was resampled or padded unexpectedly')
            mean=np.asarray(self.processor.image_processor.image_mean,dtype=np.float32);std=np.asarray(self.processor.image_processor.image_std,dtype=np.float32)
            if kind=='video':
                assert np.array_equal(mean,np.asarray(self.processor.video_processor.image_mean,dtype=np.float32))
                assert np.array_equal(std,np.asarray(self.processor.video_processor.image_std,dtype=np.float32))
            for fid,im,actual in zip(job['frame_ids'],images,restored):
                expected=(np.asarray(im,dtype=np.float32)*np.float32(self.processor.image_processor.rescale_factor)-mean)/std
                error=float(np.max(np.abs(actual-expected)));max_error=max(max_error,error)
                if error>1e-6:raise ValueError(('Processed pixels differ from frozen view',fid,error))
                if fid in self.normalized_images:
                    error=float(np.max(np.abs(actual-self.normalized_images[fid])));max_error=max(max_error,error)
                    if error>1e-6:raise ValueError(('Image/video visibility differs',fid,error))
                if kind=='image':self.normalized_images[fid]=actual.copy()
            pixel_sha=hashlib.sha256(pixels.tobytes()).hexdigest()
        for im in images:im.close()
        meta=dict(kind=kind,prompt=self.processor.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),template_prompt=prompt,
            input_token_ids=ids,attention_mask=mask,input_tokens=len(ids),last_valid_token_id=ids[active-1],last_valid_token_text=self.processor.tokenizer.decode([ids[active-1]]),
            source_frame_ids=job.get('frame_ids',[]),source_unique_frames=len(set(job.get('frame_ids',[]))),visual_tokens=visual_tokens,
            grid=grid.tolist() if grid is not None else None,pixel_tensor_sha256=pixel_sha,maximum_pixel_error=max_error,
            source_prepared_rgb_hashes=[views[k]['rgb_sha256'] for k in job.get('frame_ids',[])],truncation=False,resampling=False)
        return inputs,meta

    def forward(self,inputs):
        """接口：仅一次嵌入前向，按官方最后有效Token池化，FP32归一化；从不调用generate。"""
        import torch
        # 步骤1：原生BF16模型计算；步骤2：单独记录同步墙钟与显存，输出不是回答字母。
        inputs={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in inputs.items()};torch.cuda.synchronize()
        resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        with torch.inference_mode():
            output=self.model(**inputs,use_cache=False);last=int(inputs['attention_mask'][0].sum().item())-1
            vector=torch.nn.functional.normalize(output.last_hidden_state[0,last,:].float(),p=2,dim=-1).cpu().numpy()
        torch.cuda.synchronize();metrics=dict(forward_seconds=time.perf_counter()-started,allocated_before_forward_bytes=resident,peak_allocated_bytes=torch.cuda.max_memory_allocated(),incremental_peak_bytes=torch.cuda.max_memory_allocated()-resident)
        if vector.shape!=(4096,) or not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector))-1)>1e-5:raise ValueError('Invalid normalized embedding')
        return vector,metrics
