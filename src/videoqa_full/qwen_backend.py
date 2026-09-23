"""Qwen2.5-VL官方视频路径：已选帧输入，不二次抽样，不伪称真实编码FPS。"""
import time
import numpy as np
from videoqa_runtime.common import offline_environment
from videoqa_runtime.protocol import question_text, parse_answer
from .state import QWEN_PATH

MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 768 * 28 * 28


def padded_count(count):
    """唯一源帧预算与模型内部双帧patch补齐分开计算。"""
    if not 1 <= count <= 16:
        raise ValueError('Expected 1..16 unique source frames')
    return count + count % 2


def qa_text(question, options, duration, timestamps):
    """仅去掉LLaVA媒体占位，其余问题、选项、真实时间文字原样沿用。"""
    return question_text(question, options, duration, timestamps).removeprefix('<image>\n')


class QwenVideoBackend:
    """接口：有序RGB/PTS→官方video processor→可核验答案与Token计时。"""
    def __init__(self):
        offline_environment()
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        assert transformers.__version__ == '4.50.0'
        assert torch.cuda.device_count() == 1
        torch.manual_seed(2027)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(QWEN_PATH, local_files_only=True,
            use_fast=False, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
            size={'shortest_edge': MIN_PIXELS, 'longest_edge': MAX_PIXELS})
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_PATH, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map={'': 'cuda:0'})
        self.model.eval().requires_grad_(False)
        assert not any(p.is_meta for p in self.model.parameters())
        assert self.model.config.vision_config.temporal_patch_size == 2
        self.load_report = dict(load_seconds=time.perf_counter()-started, model_path=QWEN_PATH,
                                transformers=transformers.__version__, torch=torch.__version__,
                                dtype=str(self.model.dtype), attention=self.model.config._attn_implementation)

    def prepare(self, frames, timestamps, duration, question, options):
        """接口：显式双帧补齐、等比官方预处理、核验帧数量与时间格；不生成答案。"""
        assert len(frames) == len(timestamps)
        n = len(frames)
        padded = padded_count(n)
        # 步骤1：只在奇数帧末端内部补齐，唯一源帧列表与提示词不增加。
        pictures = [np.asarray(im.convert('RGB')) for im in frames]
        assert len({p.shape for p in pictures}) == 1
        if padded != n:
            pictures.append(pictures[-1])
        video = np.stack(pictures)
        messages = [dict(role='system', content='You are a helpful assistant.'),
                    dict(role='user', content=[dict(type='video'), dict(type='text', text=qa_text(
                        question, options, duration, timestamps))])]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 步骤2：直接传已选帧张量；不调用任何会按FPS读取原片的采样函数。
        inputs = self.processor(text=[prompt], videos=[video], fps=2.0,
                                images_kwargs={'size': {'shortest_edge': MIN_PIXELS, 'longest_edge': MAX_PIXELS}},
                                padding=False, truncation=False, return_tensors='pt')
        assert 'pixel_values_videos' in inputs and 'pixel_values' not in inputs
        grid = inputs['video_grid_thw'].tolist()
        assert len(grid) == 1 and grid[0][0] == padded // 2
        height, width = grid[0][1]*14, grid[0][2]*14
        assert height % 28 == width % 28 == 0 and MIN_PIXELS <= height*width <= MAX_PIXELS
        visual = int(np.prod(grid[0]) // 4)
        assert int((inputs['input_ids'] == self.model.config.video_token_id).sum()) == visual
        assert int((inputs['input_ids'] == self.model.config.image_token_id).sum()) == 0
        assert list(inputs['second_per_grid_ts']) == [1.0]
        assert inputs['input_ids'].shape[1] + 8 <= self.model.config.max_position_embeddings
        # 步骤3：记录编码补齐与真实时间的区别，禁止把默认2FPS写成真实选帧间隔。
        meta = dict(protocol='qwen25vl-video-16-cap-v1', prompt=prompt,
                    source_frame_count=n, encoded_frame_count=padded, internal_padding_frames=padded-n,
                    processed_size=[height,width], video_grid_thw=grid, visual_tokens=visual,
                    prefill_tokens=inputs['input_ids'].shape[1], second_per_grid_ts=[1.0],
                    encoding_fps=2.0, actual_timestamps=list(timestamps),
                    temporal_convention='uniform encoding grid; true source timestamps supplied in text')
        return inputs, meta

    def warmup(self):
        """合成前向覆盖1/11/14/16帧及不同宽高比，不生成真实或合成答案。"""
        from PIL import Image
        torch = self.torch
        reports = []
        for n, size in [(1,(320,240)),(11,(720,1280)),(14,(1280,720)),(16,(1920,1080))]:
            images = [Image.new('RGB',size,(i*13 % 256,128,70)) for i in range(n)]
            inputs, meta = self.prepare(images,list(range(n)),n+1,'Which color is visible?',
                                       ['Red','Blue','Green','Gray'])
            inputs = inputs.to('cuda:0')
            with torch.inference_mode():
                output = self.model(**inputs, use_cache=False)
                assert torch.isfinite(output.logits[:,-1,:]).all()
            torch.cuda.synchronize()
            reports.append({k:v for k,v in meta.items() if k!='prompt'})
            del output, inputs
            for im in images: im.close()
        return reports

    def answer(self, frames, timestamps, duration, question, options, generation_state=None):
        """一次贪心问答；记录实际prefill、视觉Token和有限logits，不改变解析器。"""
        from transformers import GenerationConfig
        torch = self.torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        inputs, meta = self.prepare(frames,timestamps,duration,question,options)
        inputs = inputs.to('cuda:0')
        torch.cuda.synchronize()
        meta['preprocessing_seconds'] = time.perf_counter()-started
        measured = dict(prefill_calls=0, checked_logit_steps=0)
        def prefill(module,args,kwargs):
            value = kwargs.get('inputs_embeds')
            if value is not None and measured['prefill_calls']==0:
                assert value.shape[1] == meta['prefill_tokens']
                measured['prefill_calls'] += 1
        def logits(module,args,output):
            assert torch.isfinite(output[:,-1,:]).all()
            measured['checked_logit_steps'] += 1
        hooks = [self.model.model.register_forward_pre_hook(prefill,with_kwargs=True),
                 self.model.lm_head.register_forward_hook(logits)]
        config = GenerationConfig(max_new_tokens=8, do_sample=False, num_beams=1, use_cache=True,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id)
        # 步骤1：调用账本已由外层持久化；此处只更新实际generate是否进入与返回。
        started = time.perf_counter()
        try:
            if generation_state is not None: generation_state['started']=True
            with torch.inference_mode(): output = self.model.generate(**inputs,generation_config=config)
            if generation_state is not None: generation_state['returned']=True
            torch.cuda.synchronize()
        finally:
            for hook in hooks: hook.remove()
        assert measured['prefill_calls']==1 and measured['checked_logit_steps']>0
        meta['generation_seconds']=time.perf_counter()-started
        # 步骤2：只解析新增Token，不能把问题中的选项字母当作模型输出。
        ids=output[0,inputs['input_ids'].shape[1]:].tolist()
        assert 1 <= len(ids) <= 8
        raw=self.processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
        return dict(**meta,**measured,generated_token_ids=ids,raw_output=raw,parsed_answer=parse_answer(raw),
                    generation_config=config.to_dict(),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
