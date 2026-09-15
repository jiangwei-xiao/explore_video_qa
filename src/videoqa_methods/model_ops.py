import copy
import time
import numpy as np
from videoqa_runtime.baseline_selection import BlipItmScorer, ProtocolError

SCOPE_PROMPT = '''Classify the visual evidence scope needed to answer the video question.

LOCAL: Evidence mainly comes from one object, event, or brief moment.
GLOBAL: Evidence must be combined across multiple events or the whole video,
including summaries, distributed counts, or ordering separate events.
MIXED: A local target also requires broader context or relationships,
or the scope cannot be determined from the question alone.

Use only the question. Do not answer it.
Return exactly one label: LOCAL, GLOBAL, or MIXED.

Question:
{question}'''


def parse_scope(raw):
    """接口：仅接受去空白并转大写后的完整LOCAL/GLOBAL/MIXED；其余返回MIXED及回退标记。"""
    label = raw.strip().upper()
    return (label, False) if label in ('LOCAL', 'GLOBAL', 'MIXED') else ('MIXED', True)


class ScopeClassifier:
    def __init__(self, backend):
        """复用已经加载的LLaVA后端；本接口不加载新权重，也不修改生成配置。"""
        self.backend = backend

    def classify(self, question, state):
        """核心接口：仅输入原问题做一次纯文本范围分类，返回原始输出、解析标签、Token和耗时；state记录是否开始及返回，供恢复保护使用。"""
        import torch
        from transformers import GenerationConfig
        from llava.conversation import conv_templates
        # 步骤1：构造独立的纯文本会话；不沿用视频问答历史，也不拼入选项或题型。
        started = time.perf_counter()
        conv = copy.deepcopy(conv_templates['qwen_1_5'])
        conv.append_message(conv.roles[0], SCOPE_PROMPT.format(question=question))
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        tokenizer, model = self.backend.tokenizer, self.backend.model
        inputs = tokenizer(prompt, return_tensors='pt', truncation=False).to('cuda')
        if inputs['input_ids'].shape[1] + 8 > self.backend.context:
            raise ProtocolError('Scope prompt exceeds context')
        generation = GenerationConfig(max_new_tokens=8, do_sample=False, num_beams=1, use_cache=True,
                                      bos_token_id=model.generation_config.bos_token_id,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        # 步骤2：使用局部生成配置调用一次模型，保持共享模型的默认配置不变。
        state['started'] = True
        with torch.inference_mode():
            output = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], generation_config=generation)
        state['returned'] = True
        torch.cuda.synchronize()
        raw = tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        # 步骤3：解析完整标签，失败回退MIXED；原始输出和耗时都保留，不再次分类。
        label, fallback = parse_scope(raw)
        result = dict(label=label, fallback=fallback, raw_output=raw, prompt=prompt,
                      input_tokens=int(inputs['input_ids'].shape[1]), output_token_ids=output[0].tolist(),
                      seconds=time.perf_counter() - started, visual_inputs=0, query_information='original question only')
        state['result'] = result
        return result


class FeatureScorer(BlipItmScorer):
    def score_batch_features(self, encoded, images):
        """核心接口：同一次BLIP前向同时返回ITM正匹配概率和L2归一化纯视觉CLS，以及预处理/传输和前向计时；最多16帧，不使用图文交互后的文本向量。"""
        import torch
        if not 1 <= len(images) <= 16:
            raise ProtocolError('Invalid BLIP feature batch size')
        torch.cuda.synchronize()
        start = time.perf_counter()
        # 步骤1：官方预处理与CPU→GPU传输独立同步计时，保持FP32和原问题编码。
        pixels = self.processor(images=images, return_tensors='pt')['pixel_values'].to('cuda', torch.float32)
        inputs = {k: v.repeat(len(images), 1).cuda() for k, v in encoded.items() if k in ('input_ids', 'attention_mask')}
        torch.cuda.synchronize()
        preprocess = time.perf_counter() - start
        start = time.perf_counter()
        with torch.inference_mode():
            # 步骤2：一次前向取得正匹配概率及视觉编码器CLS，避免额外视觉前向。
            output = self.model(pixel_values=pixels, **inputs, use_itm_head=True)
            scores = output.itm_score.softmax(-1)[:, 1].cpu().tolist()
            cls = output.last_hidden_state[:, 0, :].float()
            if not torch.isfinite(cls).all() or torch.any(torch.linalg.vector_norm(cls, dim=-1) <= 1e-12):
                raise ProtocolError('Invalid pure image CLS')
            # 步骤3：归一化纯图像CLS并转到CPU；归一化和传输成本计入本阶段。
            features = torch.nn.functional.normalize(cls, p=2, dim=-1).cpu().numpy()
        torch.cuda.synchronize()
        forward = time.perf_counter() - start
        if not np.isfinite(scores).all() or any(not 0 <= s <= 1 for s in scores):
            raise ProtocolError('Invalid ITM probabilities')
        return scores, features, preprocess, forward

    def warmup(self):
        """接口：执行两次合成图像特征前向，用于预热；不读取开发题、不产生答案。"""
        from PIL import Image
        encoded = self.tokenize('Which synthetic color is visible?')
        for _ in range(2):
            self.score_batch_features(encoded, [Image.new('RGB', (1280, 720), 'gray')] * 16)
