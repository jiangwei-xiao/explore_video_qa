"""LMMs-Eval原生generate_until桥接：只注入冻结RGB，不替换框架生成代码。"""
import copy
import hashlib
import time
import numpy as np
from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from lmms_eval.models.simple.llava_vid import LlavaVid
from .common import GENERATION,POST_PROMPT,load_task,task_doc


def tensor_hash(tensor):
    """观测接口：BF16先按字节视图哈希，避免转换FP32后掩盖输入差异。"""
    import torch
    value=tensor.detach().cpu().contiguous()
    prefix=(str(value.dtype)+str(list(value.shape))).encode()
    return hashlib.sha256(prefix+value.view(torch.uint8).numpy().tobytes()).hexdigest()


class FrozenInputLlavaVid(LlavaVid):
    """接口：共享已核验LLaVA对象，只覆盖RGB输入获取，继承原生generate_until。"""
    def __init__(self,backend):
        # 步骤1：使用框架基础状态；模型/Tokenizer/预处理器共享，排除重复加载差异。
        lmms.__init__(self)
        self.backend=backend;self._model=backend.model;self._tokenizer=backend.tokenizer
        self._image_processor=backend.processor;self._config=backend.model.config;self._max_length=backend.context
        self._device=backend.torch.device('cuda:0');self.device_map='cuda:0'
        self.pretrained=backend.load_report.get('model_path','frozen-LLaVA-Video-7B-Qwen2')
        self.torch_dtype='bfloat16';self.conv_template='qwen_1_5';self.batch_size_per_gpu=1
        self.use_cache=True;self.truncate_context=False;self.truncation=False;self.video_decode_backend='decord'
        self.max_frames_num=16;self.fps=1;self.force_sample=True;self.add_time_instruction=False
        self.frozen=None;self.load_calls=0

    def load_video(self,video_path,max_frames_num,fps,force_sample=False):
        """步骤2：替代解码入口，不进行再次选帧；源身份由FrameProvider在调用前核验。"""
        assert self.frozen is not None and video_path==self.frozen['path'] and max_frames_num==16
        self.load_calls+=1
        frames=self.frozen['frames']
        return np.stack([np.asarray(im.convert('RGB')) for im in frames]),', '.join(f'{t:.2f}s' for t in self.frozen['times']),self.frozen['duration']

    def ask(self,frames,times,duration,row,video_path):
        """步骤3：原任务函数构造文字→框架Instance→继承的generate_until→原任务计分。"""
        task=load_task();doc=task_doc(row);context=task.videomme_doc_to_text(doc,{'post_prompt':POST_PROMPT})
        from llava.conversation import conv_templates
        conversation=conv_templates[self.conv_template].copy()
        conversation.append_message(conversation.roles[0],'<image>\n'+context)
        conversation.append_message(conversation.roles[1],None)
        prompt=conversation.get_prompt()
        self.frozen=dict(frames=frames,times=times,duration=duration,path=video_path);before=self.load_calls
        instance=Instance(request_type='generate_until',arguments=(context,copy.deepcopy(GENERATION),
            lambda _: [video_path],0,'videomme_reference','test'),idx=0,
            metadata={'task':'videomme_reference','doc_id':0,'repeats':1})
        self.task_dict={'videomme_reference':{'test':[doc]}}
        try:result=self.generate_until([instance])
        finally:self.frozen=None
        assert self.load_calls==before+1 and len(result)==1
        metric=task.videomme_process_results(doc,result)['videomme_perception_score']
        return dict(raw_output=result[0],parsed_answer=metric['pred_answer'] or None,prompt=prompt,
                    framework_metric=metric,context=context,framework_generate='LlavaVid.generate_until')


def observe_generate(backend,frames,max_new_tokens,state,operation):
    """接口：观测实际generate输入/输出与prefill，不改数值计算，不额外调用模型。"""
    import torch
    expected=backend.processor.preprocess(frames,return_tensors='pt')['pixel_values'].to(dtype=torch.bfloat16)
    expected_hash=tensor_hash(expected);n=len(frames)
    captured=dict(expected_pixel_sha256=expected_hash,expected_shape=[n,3,384,384],generate_calls=0,
                  prefill_calls=0,checked_logit_steps=0)
    original=backend.model.generate
    config_before=copy.deepcopy(backend.model.generation_config.to_dict())
    def capture(*args,**kwargs):
        """步骤1：在实际模型调用边界验证RGB预处理结果与输入预算。"""
        captured['generate_calls']+=1;assert captured['generate_calls']==1
        ids=args[0] if args else kwargs.get('inputs',kwargs.get('input_ids'))
        video=kwargs['images'][0]
        assert list(video.shape)==[n,3,384,384] and video.dtype==torch.bfloat16
        captured['pixel_sha256']=tensor_hash(video);assert captured['pixel_sha256']==expected_hash
        gc=kwargs.get('generation_config')
        limit=gc.max_new_tokens if gc else kwargs['max_new_tokens'];assert limit==max_new_tokens
        captured.update(input_token_ids=ids[0].tolist(),text_input_tokens=ids.shape[1],max_new_tokens=limit,
            stopping_criteria=[type(s).__name__ for s in kwargs.get('stopping_criteria',[])],
            model_generation_config=config_before)
        assert int((ids==-200).sum())==1
        assert ids.shape[1]-1+n*210+limit<=min(backend.context,backend.model.config.tokenizer_model_max_length)
        torch.cuda.synchronize();t=time.perf_counter();state['started']=True
        output=original(*args,**kwargs);state['returned']=True;torch.cuda.synchronize()
        captured.update(generation_seconds=time.perf_counter()-t,generated_token_ids=output[0].tolist())
        assert 1<=len(output[0])<=limit
        return output
    def prefill(module,args,kwargs):
        value=kwargs.get('inputs_embeds')
        if value is not None and captured['prefill_calls']==0:
            captured['prefill_calls']+=1;captured['prefill_tokens']=value.shape[1]
            captured['visual_tokens']=value.shape[1]-(captured['text_input_tokens']-1)
            assert captured['visual_tokens']==n*210
    def logits(module,args,output):
        assert torch.isfinite(output[:,-1,:]).all();captured['checked_logit_steps']+=1
    # 步骤2：两个协议都采用相同只读观测钩子，退出时恢复所有属性与句柄。
    hooks=[backend.model.get_model().register_forward_pre_hook(prefill,with_kwargs=True),
           backend.model.lm_head.register_forward_hook(logits)]
    backend.model.generate=capture;torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    try:answer=operation()
    finally:
        backend.model.generate=original
        for h in hooks:h.remove()
    assert captured['generate_calls']==captured['prefill_calls']==1 and captured['checked_logit_steps']>0
    from llava.mm_utils import tokenizer_image_token
    assert tokenizer_image_token(answer['prompt'],backend.tokenizer,-200)==captured['input_token_ids']
    assert backend.model.generation_config.to_dict()==config_before
    captured.update(qa_seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    return answer,captured
