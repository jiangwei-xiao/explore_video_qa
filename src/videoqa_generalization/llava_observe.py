"""LLaVA参考接口的生成返回检查点；冻结模型数值与任务行为保持不变。"""
import copy
import time
from videoqa_full.state import durable
from videoqa_lmms_bridge.common import load_task, task_doc, POST_PROMPT


def result_from_raw(raw, row):
    """接口：只重建已完整返回的任务计分和观测，不再次进入模型。"""
    if raw['generation_state'] != {'started': True, 'returned': True}:
        raise ValueError('Generation has not been confirmed')
    text = raw['raw_generated_text'].strip()
    metric = load_task().videomme_process_results(task_doc(row), [text])['videomme_perception_score']
    answer = dict(raw_output=text, parsed_answer=metric['pred_answer'] or None,
        framework_metric=metric, context=raw['context'], prompt=raw['prompt'],
        framework_generate='LlavaVid.generate_until')
    return dict(answer=answer, observed=raw['observed'])


def observe(backend, adapter, frames, selection, row, state, path=None, identity=None):
    """接口：一次原生框架generate，返回立即保存原文/Token；后处理失败可恢复。"""
    import torch
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from videoqa_lmms_bridge.llava_adapter import tensor_hash
    context = load_task().videomme_doc_to_text(task_doc(row), {'post_prompt': POST_PROMPT})
    conv = conv_templates['qwen_1_5'].copy()
    conv.append_message(conv.roles[0], '<image>\n'+context); conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt(); n = len(frames)
    expected = backend.processor.preprocess(frames, return_tensors='pt')['pixel_values'].to(dtype=torch.bfloat16)
    expected_hash = tensor_hash(expected)
    del expected
    original = backend.model.generate
    before = copy.deepcopy(backend.model.generation_config.to_dict())
    measured = dict(generate_calls=0, prefill_calls=0, checked_logit_steps=0,
        expected_pixel_sha256=expected_hash, source_frames=n, prompt=prompt, context=context)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()

    def generate(*args, **kwargs):
        """步骤1：真实generate边界校验输入；返回后在框架解码/计分前持久化。"""
        measured['generate_calls'] += 1
        ids = args[0] if args else kwargs.get('inputs', kwargs.get('input_ids'))
        video = kwargs['images'][0]
        if measured['generate_calls'] != 1 or list(video.shape) != [n,3,384,384] or video.dtype != torch.bfloat16:
            raise ValueError('LLaVA visual input changed')
        digest = tensor_hash(video)
        if digest != expected_hash or ids[0].tolist() != tokenizer_image_token(prompt, backend.tokenizer, -200):
            raise ValueError('LLaVA tensor or prompt mismatch')
        if kwargs['max_new_tokens'] != 16 or kwargs.get('do_sample') is not False:
            raise ValueError('LLaVA generation settings changed')
        measured.update(pixel_sha256=digest, input_token_ids=ids[0].tolist(), text_input_tokens=ids.shape[1],
            max_new_tokens=16, model_generation_config=before,
            stopping_criteria=[type(s).__name__ for s in kwargs.get('stopping_criteria', [])])
        if ids.shape[1]-1+n*210+16 > min(backend.context, backend.model.config.tokenizer_model_max_length):
            raise ValueError('LLaVA context would truncate')
        torch.cuda.synchronize(); t = time.perf_counter(); state['started'] = True
        output = original(*args, **kwargs)
        state['returned'] = True; torch.cuda.synchronize()
        measured.update(generated_token_ids=output[0].tolist(), generation_seconds=time.perf_counter()-t,
            qa_seconds=time.perf_counter()-start, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        if not 1 <= len(output[0]) <= 16: raise ValueError('Unexpected generated token count')
        text = backend.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        raw = dict(identity=identity, generation_state=dict(state), raw_generated_text=text,
            prompt=prompt, context=context, observed=copy.deepcopy(measured))
        if path is not None: durable(path, raw)
        measured['_raw'] = raw
        return output

    def prefill(module, args, kwargs):
        """步骤2：只观测真实视频prefill的Token；不执行额外前向。"""
        value = kwargs.get('inputs_embeds')
        if value is not None and measured['prefill_calls'] == 0:
            measured['prefill_calls'] = 1; measured['prefill_tokens'] = value.shape[1]
            measured['visual_tokens'] = value.shape[1]-(measured['text_input_tokens']-1)
            if measured['visual_tokens'] != n*210: raise ValueError('Visual token count changed')

    def logits(module, args, output):
        if not torch.isfinite(output[:,-1,:]).all(): raise ValueError('Nonfinite logits')
        measured['checked_logit_steps'] += 1

    hooks = [backend.model.get_model().register_forward_pre_hook(prefill, with_kwargs=True),
             backend.model.lm_head.register_forward_hook(logits)]
    backend.model.generate = generate
    try:
        s = selection
        answer = adapter.ask(frames, [f['timestamp_seconds'] for f in s['selected_frames']],
            s['video']['duration_seconds'], row, s['video']['path'])
        restored = result_from_raw(measured.pop('_raw'), row)
        if answer != restored['answer']: raise ValueError('Framework and raw reconstruction differ')
        if measured['generate_calls'] != measured['prefill_calls'] or measured['checked_logit_steps'] == 0:
            raise ValueError('Observed generation count invalid')
        return restored
    finally:
        # 步骤3：恢复所有钩子与生成配置边界，避免一题观测污染下一题。
        backend.model.generate = original
        for hook in hooks: hook.remove()
        if backend.model.generation_config.to_dict() != before: raise ValueError('Generation config polluted')
