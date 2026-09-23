"""Qwen框架合成适配：原生generate_until，仅对媒体读取提供作用域内依赖注入。"""
import copy
from contextlib import contextmanager
import numpy as np
from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from lmms_eval.models.simple import qwen2_5_vl as reference
from videoqa_full.qwen_backend import MIN_PIXELS,MAX_PIXELS,padded_count
from .common import load_task,task_doc,GENERATION,POST_PROMPT


class VideoInputGuard:
    """接口：检查框架送到processor的帧未改变，并显式固定像素和编码FPS。"""
    def __init__(self,processor,frames):
        self.processor=processor;self.frames=frames;self.observed={}

    def __getattr__(self,name):return getattr(self.processor,name)

    def __call__(self,*args,**kwargs):
        # 步骤1：参考适配器内部即便调用linspace，也必须保持已选1..16帧逐像素不变。
        assert kwargs.get('images') is None and len(kwargs['videos'])==1
        assert np.array_equal(np.asarray(kwargs['videos'][0]),self.frames)
        kwargs.update(fps=2.0,truncation=False,
            images_kwargs={'size':{'shortest_edge':MIN_PIXELS,'longest_edge':MAX_PIXELS}})
        result=self.processor(*args,**kwargs);grid=result['video_grid_thw'].tolist()
        assert grid[0][0]==padded_count(len(self.frames))//2
        assert result['second_per_grid_ts']==[1.0]
        self.observed=dict(video_grid_thw=grid,source_frames=len(self.frames),encoded_frames=2*grid[0][0],
            processed_size=[grid[0][1]*14,grid[0][2]*14],visual_tokens=int(np.prod(grid[0])//4),
            prefill_tokens=result['input_ids'].shape[1],encoding_fps=2.0,prompt=kwargs['text'][0])
        return result


@contextmanager
def frozen_media(frames,path):
    """单进程batch1的媒体依赖注入；只替代文件读取，不替代模型generate，退出必恢复。"""
    old_reader=reference.decord.VideoReader;old_process=reference.process_vision_info
    counts=dict(first_frame_reads=0,video_reads=0)
    class Reader:
        def __init__(self,p):assert p==path
        def __getitem__(self,index):
            assert index==0;counts['first_frame_reads']+=1
            class Frame:
                def asnumpy(self):return frames[0]
            return Frame()
    def process(messages):
        visuals=messages[0][-1]['content']
        assert visuals[0]['type']=='video' and visuals[0]['video']==path
        counts['video_reads']+=1
        return None,[frames]
    reference.decord.VideoReader=Reader;reference.process_vision_info=process
    try:yield counts
    finally:
        reference.decord.VideoReader=old_reader;reference.process_vision_info=old_process


class FrozenInputQwen(reference.Qwen2_5_VL):
    """接口：共享已验收FA2模型，继承固定LMMs-Eval的完整生成和解码流程。"""
    def __init__(self,backend):
        lmms.__init__(self)
        self.backend=backend;self._model=backend.model;self._tokenizer=backend.processor.tokenizer
        self._config=backend.model.config;self._device=backend.torch.device('cuda:0');self.device_map='cuda:0'
        self.processor=backend.processor;self._max_length=backend.model.config.max_position_embeddings
        self.batch_size_per_gpu=1;self.use_cache=True;self.max_num_frames=16
        self.max_pixels=MAX_PIXELS;self.min_pixels=MIN_PIXELS
        self.system_prompt='You are a helpful assistant.';self.interleave_visuals=False;self.reasoning_prompt=None

    def ask(self,frames,row):
        """只接受调用方给出的RGB，原任务函数与框架Instance负责文字和计分。"""
        task=load_task();doc=task_doc(row);context=task.videomme_doc_to_text(doc,{'post_prompt':POST_PROMPT})
        pixels=np.stack([np.asarray(im.convert('RGB')) for im in frames]);path='frozen-input.mp4'
        guard=VideoInputGuard(self.backend.processor,pixels);self.processor=guard
        inst=Instance(request_type='generate_until',arguments=(context,copy.deepcopy(GENERATION),
            lambda _: [path],0,'videomme_reference','test'),idx=0,metadata={'task':'videomme_reference','doc_id':0,'repeats':1})
        self.task_dict={'videomme_reference':{'test':[doc]}}
        try:
            with frozen_media(pixels,path) as count:answers=self.generate_until([inst])
            assert count=={'first_frame_reads':1,'video_reads':1} and len(answers)==1
        finally:self.processor=self.backend.processor
        metric=task.videomme_process_results(doc,answers)['videomme_perception_score']
        return dict(raw_output=answers[0],parsed_answer=metric['pred_answer'],observed=guard.observed,
            framework_metric=metric,framework_generate='Qwen2_5_VL.generate_until',media_reads=count)
