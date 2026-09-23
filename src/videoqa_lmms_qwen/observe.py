"""实际Qwen生成观测及原始答案检查点；不替代框架生成算法。"""
import copy
import hashlib
import time
import numpy as np
from videoqa_full.state import durable
from videoqa_full.qwen_backend import MIN_PIXELS, MAX_PIXELS, padded_count
from videoqa_lmms_bridge.common import load_task, task_doc, POST_PROMPT


def tensor_hash(value):
    """接口：按实际dtype、形状和连续字节校验处理器与模型之间的输入。"""
    import torch
    x=value.detach().cpu().contiguous()
    return hashlib.sha256((str(x.dtype)+str(list(x.shape))).encode()+x.view(torch.uint8).numpy().tobytes()).hexdigest()


def clean_raw(raw):
    """仅重放固定框架的停止字符串与输出清理，不调用模型或改变解析规则。"""
    from lmms_eval.models.model_utils.reasoning_model_utils import parse_reasoning_model_answer
    text=raw['raw_generated_text']
    for term in raw['stop_terms']:
        if term: text=text.split(term)[0]
    return parse_reasoning_model_answer(text)


def result_from_raw(raw,row):
    """接口：原始生成已落盘后，可安全恢复任务计分；不重新生成。"""
    assert raw['generation_state']=={'started':True,'returned':True}
    text=clean_raw(raw)
    metric=load_task().videomme_process_results(task_doc(row),[text])['videomme_perception_score']
    return dict(raw_output=text,parsed_answer=metric['pred_answer'] or None,framework_metric=metric,
        observed=raw['observed'],raw_generated_text=raw['raw_generated_text'],
        framework_generate='Qwen2_5_VL.generate_until',media_reads=dict(first_frame_reads=1,video_reads=1))


def observe(backend,adapter,frames,row,state,raw_path=None,identity=None):
    """接口：一次框架问答，核验视频Tensor/Token，生成返回后立即保存可恢复原文。"""
    import torch
    n=len(frames);base_processor=backend.processor;model=backend.model
    old_generate=model.generate;before_config=copy.deepcopy(model.generation_config.to_dict())
    measured=dict(generate_calls=0,prefill_calls=0,checked_logit_steps=0,source_frames=n)
    native=load_task();context=native.videomme_doc_to_text(task_doc(row),{'post_prompt':POST_PROMPT})
    expected_rgb=np.stack([np.asarray(im.convert('RGB')) for im in frames])
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    class Processor:
        """步骤1：截取官方处理器真实产物；不重复预处理、不改变输入数据。"""
        def __getattr__(self,name):return getattr(base_processor,name)
        def __call__(self,*args,**kwargs):
            assert kwargs['truncation'] is False and kwargs['fps']==2.0
            assert kwargs.get('images') is None and np.array_equal(np.asarray(kwargs['videos'][0]),expected_rgb)
            assert kwargs['images_kwargs']['size']==dict(shortest_edge=MIN_PIXELS,longest_edge=MAX_PIXELS)
            result=base_processor(*args,**kwargs);grid=result['video_grid_thw'].tolist();assert len(grid)==1
            h,w=grid[0][1]*14,grid[0][2]*14;visual=int(np.prod(grid[0])//4)
            assert h%28==w%28==0 and MIN_PIXELS<=h*w<=MAX_PIXELS
            assert grid[0][0]*2==padded_count(n) and list(result['second_per_grid_ts'])==[1.0]
            ids=result['input_ids'][0];assert int((ids==model.config.video_token_id).sum())==visual
            assert int((ids==model.config.image_token_id).sum())==0
            assert len(ids)+16<=model.config.max_position_embeddings
            assert context in kwargs['text'][0]
            measured.update(processor_pixel_sha256=tensor_hash(result['pixel_values_videos']),
                input_token_ids=ids.tolist(),prefill_tokens=len(ids),video_grid_thw=grid,
                visual_tokens=visual,processed_size=[h,w],encoded_frames=2*grid[0][0],
                internal_padding_frames=padded_count(n)-n,encoding_fps=2.0,second_per_grid_ts=[1.0],
                prompt=kwargs['text'][0],context=context,pixel_shape=list(result['pixel_values_videos'].shape),
                pixel_dtype=str(result['pixel_values_videos'].dtype))
            return result
    def generate(*args,**kwargs):
        """步骤2：记录真实generate边界；不确定生成不允许自动再次进入。"""
        measured['generate_calls']+=1;assert measured['generate_calls']==1
        assert kwargs['max_new_tokens']==16 and kwargs['do_sample'] is False and kwargs['num_beams']==1
        assert kwargs['temperature'] is None and kwargs['top_p'] is None and kwargs['use_cache'] is True
        assert kwargs['input_ids'][0].tolist()==measured['input_token_ids']
        measured['model_pixel_sha256']=tensor_hash(kwargs['pixel_values_videos'])
        assert measured['model_pixel_sha256']==measured['processor_pixel_sha256']
        assert kwargs['video_grid_thw'].tolist()==measured['video_grid_thw']
        measured['generation_kwargs']={k:kwargs[k] for k in ('max_new_tokens','do_sample','num_beams','temperature','top_p','eos_token_id','pad_token_id','use_cache')}
        measured['model_generation_config']=before_config
        torch.cuda.synchronize();t=time.perf_counter();state['started']=True
        with torch.inference_mode():output=old_generate(*args,**kwargs)
        state['returned']=True;torch.cuda.synchronize()
        measured['generation_seconds']=time.perf_counter()-t
        length=kwargs['input_ids'].shape[1]
        assert torch.equal(output[0,:length],kwargs['input_ids'][0])
        ids=output[0,length:].tolist();assert 1<=len(ids)<=16
        text=base_processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        measured.update(generated_token_ids=ids,qa_through_generation_seconds=time.perf_counter()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        raw=dict(identity=identity,generation_state=dict(state),raw_generated_text=text,
            stop_terms=[base_processor.tokenizer.decode(base_processor.tokenizer.eos_token_id)],observed=copy.deepcopy(measured))
        if raw_path is not None:durable(raw_path,raw)
        measured['_raw']=raw
        return output
    def prefill(module,args,kwargs):
        if kwargs.get('inputs_embeds') is not None and measured['prefill_calls']==0:
            measured['prefill_calls']=1;measured['actual_prefill_tokens']=kwargs['inputs_embeds'].shape[1]
            assert measured['actual_prefill_tokens']==measured['prefill_tokens']
    def logits(module,args,output):
        assert torch.isfinite(output[:,-1,:]).all();measured['checked_logit_steps']+=1
    hooks=[model.model.register_forward_pre_hook(prefill,with_kwargs=True),model.lm_head.register_forward_hook(logits)]
    backend.processor=Processor();model.generate=generate
    try:
        answer=adapter.ask(frames,row)
        assert measured['generate_calls']==measured['prefill_calls']==1 and measured['checked_logit_steps']>0
        raw=measured.pop('_raw');restored=result_from_raw(raw,row)
        assert answer['raw_output']==restored['raw_output'] and answer['framework_metric']==restored['framework_metric']
        restored['observed']=dict(measured,qa_seconds=time.perf_counter()-start)
        return restored
    finally:
        # 步骤3：退出恢复所有钩子和对象，防止一题设置污染下一题。
        model.generate=old_generate;backend.processor=base_processor;adapter.processor=base_processor
        for hook in hooks:hook.remove()
        assert model.generation_config.to_dict()==before_config
