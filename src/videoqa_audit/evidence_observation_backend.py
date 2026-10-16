"""冻结Qwen的可核查观察接口；独立生成配置，不改正式视频问答后端。"""
import hashlib
import json
import time

import numpy as np

from videoqa_full.qwen_backend import MIN_PIXELS, MAX_PIXELS
from videoqa_full.state import durable

MAX_NEW_TOKENS = 256
INSTRUCTION = '''Inspect these sampled video frames in their provided temporal order.
They are not a complete or continuously observed video.
Report only facts directly visible in the frames and relevant to the question.
Describe visible objects, appearance, text, states, or actions. Copy visible text accurately.
Each observation or relation must cite the supplied frame IDs that show it.
Do not infer unseen transitions, intentions, causes, or events outside these frames.
If first, last, before, after, total count, or another question constraint is not
established by the visible content, list the specific unverified constraint.
A statement written on screen may be reported as visible text; do not silently
convert it into an independently verified historical or whole-video fact.
Do not answer a multiple-choice question, name an option letter, or use outside knowledge.
Return only one JSON object with exactly these keys:
{"observations":[{"frame_ids":[1],"text":"visible fact"}],
 "relations":[{"frame_ids":[1,2],"text":"visibly supported relation"}],
 "unverified_constraints":["specific question constraint"]}
Use at most four observations and one relation. Use empty arrays when unsupported.
Only the supplied frame IDs are valid. No Markdown or explanation outside JSON.'''


def _hash_array(value):
    """接口：CPU字节指纹用来绑定实际处理像素与文本Token。"""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def observation_text(question, frame_ids, times):
    """接口：只接受原问题、可用帧编号和真实时间，不接收选项、答案或评价标签。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Original question is required')
    if len(frame_ids) != len(times) or len(set(frame_ids)) != len(frame_ids):
        raise ValueError('Frame ID/time identity differs')
    if any(type(i) is not int or i not in (1, 2, 3, 4) for i in frame_ids):
        raise ValueError('Invalid public frame IDs')
    if list(times) != sorted(times) or not all(np.isfinite(t) for t in times):
        raise ValueError('Frames must have ordered finite source times')
    mapping = '\n'.join(f'Frame ID {i}: source time {t:.9f} seconds'
                        for i, t in zip(frame_ids, times))
    return INSTRUCTION + '\n\nQuestion:\n' + question + '\n\nAvailable frames:\n' + mapping


def prepare_observation(processor, frames, question, frame_ids, times, model_config):
    """接口：原RGB与观察问题→官方视频输入和完整CPU身份；不运行模型。"""
    if len(frames) not in (1, 4) or len(frames) != len(frame_ids):
        raise ValueError('Expected one or four unique source frames')
    pictures = [np.ascontiguousarray(np.asarray(im.convert('RGB'))) for im in frames]
    if len({x.shape for x in pictures}) != 1:
        raise ValueError('Source geometry changed inside one group')
    source_hashes = [_hash_array(x) for x in pictures]
    n = len(pictures)
    # 步骤1：单帧在内部重复自身形成时间patch，不追加新的源证据或帧编号。
    if n == 1:
        pictures.append(pictures[-1])
    message = [dict(role='system', content='You are a helpful assistant.'),
               dict(role='user', content=[dict(type='video'),
                    dict(type='text', text=observation_text(question, frame_ids, times))])]
    prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], videos=[np.stack(pictures)], fps=2.0,
        images_kwargs={'size': {'shortest_edge': MIN_PIXELS, 'longest_edge': MAX_PIXELS}},
        padding=False, truncation=False, return_tensors='pt')
    grid = inputs['video_grid_thw'].tolist()
    if len(grid) != 1 or grid[0][0] != len(pictures)//2:
        raise ValueError('Unexpected temporal grid or secondary sampling')
    height, width = grid[0][1]*14, grid[0][2]*14
    if height % 28 or width % 28 or not MIN_PIXELS <= height*width <= MAX_PIXELS:
        raise ValueError('Official processed geometry outside frozen protocol')
    visual = int(np.prod(grid[0])//4)
    if int((inputs['input_ids'] == model_config.video_token_id).sum()) != visual:
        raise ValueError('Actual video placeholder count differs')
    if int((inputs['input_ids'] == model_config.image_token_id).sum()) or list(inputs['second_per_grid_ts']) != [1.0]:
        raise ValueError('Media type or encoding FPS changed')
    tokens = inputs['input_ids'][0].tolist()
    if len(tokens)+MAX_NEW_TOKENS > model_config.max_position_embeddings:
        raise ValueError('Observer input would be truncated')
    # 步骤2：文本、像素、源身份、真实时间和编码时间约定分别绑定。
    identity = dict(prompt=prompt, input_token_ids=tokens,
        token_sha256=_hash_array(inputs['input_ids'].numpy()),
        pixel_sha256=_hash_array(inputs['pixel_values_videos'].numpy()),
        source_rgb_sha256=source_hashes, frame_ids=list(frame_ids),
        source_times_seconds=list(times), source_frames=n, encoded_frames=len(pictures),
        internal_padding_frames=len(pictures)-n, video_grid_thw=grid,
        processed_size=[height,width], visual_tokens=visual, prefill_tokens=len(tokens),
        encoding_fps=2.0, max_new_tokens=MAX_NEW_TOKENS)
    return inputs, identity


def parse_observations(raw, allowed_ids, truncated=False):
    """接口：严格检查JSON和帧引用；格式/截断失败保留原文，不触发重试。"""
    try:
        if truncated:
            raise ValueError('generation_limit_without_eos')
        value = json.loads(raw.strip())
        if not isinstance(value, dict) or set(value) != {
                'observations', 'relations', 'unverified_constraints'}:
            raise ValueError('unexpected_json_shape')
        for name, limit in (('observations',4),('relations',1)):
            claims=value[name]
            if not isinstance(claims,list) or len(claims)>limit:
                raise ValueError('invalid_claim_count:'+name)
            for item in claims:
                if not isinstance(item,dict) or set(item)!={'frame_ids','text'}:
                    raise ValueError('invalid_claim_shape')
                ids=item['frame_ids']
                if (not isinstance(ids,list) or not ids or len(ids)!=len(set(ids))
                    or any(type(i) is not int or i not in allowed_ids for i in ids)):
                    raise ValueError('invalid_frame_reference')
                if not isinstance(item['text'],str) or not item['text'].strip():
                    raise ValueError('empty_claim')
        scope=value['unverified_constraints']
        if not isinstance(scope,list) or any(not isinstance(x,str) or not x.strip() for x in scope):
            raise ValueError('invalid_unverified_constraints')
        return dict(valid=True, value=value, error=None)
    except (ValueError,TypeError,json.JSONDecodeError) as error:
        return dict(valid=False, value=None, error=str(error))


class EvidenceObservationBackend:
    """接口：冻结Qwen事实生成器，生成后立刻保存原文，再进行任何格式解析。"""
    def __init__(self):
        from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
        self.backend=QwenFA2VideoBackend()
        self.model=self.backend.model;self.processor=self.backend.processor
        self.torch=self.backend.torch;self.load_report=self.backend.load_report

    def prepare(self, frames, question, frame_ids, times):
        """接口：运行前重算官方CPU输入，与冻结身份逐字节比对后才允许生成。"""
        return prepare_observation(self.processor,frames,question,frame_ids,times,self.model.config)

    def generate_prepared(self, inputs, identity, state, raw_path, call_identity):
        """接口：独立256Token配置→原始返回检查点；不调用历史8/16Token问答函数。"""
        from transformers import GenerationConfig
        torch=self.torch
        frozen_config=self.model.generation_config.to_dict()
        eos=self.model.generation_config.eos_token_id
        eos_ids={eos} if isinstance(eos,int) else set(eos)
        config=GenerationConfig(max_new_tokens=MAX_NEW_TOKENS,do_sample=False,num_beams=1,
            use_cache=True,eos_token_id=eos,pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id)
        measured=dict(vision_calls=0,prefill_calls=0)
        def vision_count(module,args,output):
            measured['vision_calls']+=1
        def prefill_count(module,args,kwargs):
            embedding=kwargs.get('inputs_embeds')
            if embedding is not None and embedding.shape[1]==identity['prefill_tokens']:
                measured['prefill_calls']+=1
        hooks=[self.model.visual.register_forward_hook(vision_count),
               self.model.model.register_forward_pre_hook(prefill_count,with_kwargs=True)]
        inputs=inputs.to('cuda:0');torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        began=time.perf_counter()
        # 步骤1：外层账本已持久化；这里只更新实际generate进入和返回状态。
        try:
            state['started']=True
            with torch.inference_mode():
                output=self.model.generate(**inputs,generation_config=config)
            state['returned']=True
            torch.cuda.synchronize()
        finally:
            for hook in hooks:hook.remove()
        ids=output[0,identity['prefill_tokens']:].tolist()
        raw=self.processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
        truncated=len(ids)==MAX_NEW_TOKENS and bool(ids) and ids[-1] not in eos_ids
        metadata=dict(identity,**measured,generated_token_ids=ids,
            generation_seconds=time.perf_counter()-began,normal_eos=bool(ids) and ids[-1] in eos_ids,
            truncated=truncated,generation_config=config.to_dict(),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        # 步骤2：先保存原Token与原文，任何随后解析/验收失败都不能重发生成。
        checkpoint=dict(identity=call_identity,generation_state=dict(state),
                        raw_output=raw,metadata=metadata)
        durable(raw_path,checkpoint)
        if measured!={'vision_calls':1,'prefill_calls':1}:
            raise ValueError('Unexpected actual model invocation count')
        if not 1<=len(ids)<=MAX_NEW_TOKENS or self.model.generation_config.to_dict()!=frozen_config:
            raise ValueError('Generation config polluted or token budget differs')
        # 步骤3：格式错误也是已完成辅助调用，直接保存判定，不重试或换提示。
        return dict(raw_output=raw,metadata=metadata,
                    parsed=parse_observations(raw,identity['frame_ids'],truncated))
