"""MLVU任务桥接：继承原生LMMs生成，复用只读观测器但隔离其Video-MME任务绑定。"""
import copy
import importlib.util
from pathlib import Path
import numpy as np
from videoqa_runtime.common import ROOT
from videoqa_mlvu.data import context, task_doc, load_task, POST_PROMPT
from videoqa_lmms_bridge.common import GENERATION


def metric(row, outputs):
    """接口：直接执行固定MLVU原生解析，保持官方任务返回字段不变。"""
    return load_task().mlvu_process_results(task_doc(row), outputs)['mlvu_percetion_score']


class ObserverTaskBridge:
    """私有兼容层：旧观测器的内部函数名不改，实际文字与计分全部调用MLVU原生任务。"""
    def videomme_doc_to_text(self, doc, kwargs):
        """兼容旧观测器函数名；内部只执行MLVU原生任务文字函数。"""
        return load_task().mlvu_doc_to_text(doc, dict(pre_prompt='', post_prompt=POST_PROMPT))

    def videomme_process_results(self, doc, outputs):
        """兼容内部字典键，返回内容仍是原生MLVU逐题指标。"""
        return {'videomme_perception_score': load_task().mlvu_process_results(doc, outputs)['mlvu_percetion_score']}


def observer_module(profile):
    """接口：私有模块实例只重绑定任务依赖，不修改旧模块、原函数或冻结文件字节。"""
    if profile == 'qwen32':
        name = 'videoqa_budget32._mlvu_observe'; path = ROOT / 'src/videoqa_budget32/observe.py'
    else:
        name = 'videoqa_generalization._mlvu_observe'; path = ROOT / 'src/videoqa_generalization/llava_observe.py'
    # 步骤1：复用完整生成边界、像素/Token/停止条件观测逻辑，避免另写一套数值推理。
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.load_task = lambda: ObserverTaskBridge(); module.task_doc = task_doc; module.POST_PROMPT = POST_PROMPT
    return module


def make_adapter(profile, backend):
    """接口：只适配MLVU上下文和冻结媒体读取，真正生成仍由固定框架generate_until完成。"""
    from lmms_eval.api.instance import Instance
    if profile == 'qwen32':
        from videoqa_budget32.qwen_adapter import FrozenInputQwen, VideoInputGuard, frozen_media
        class MlvuQwen(FrozenInputQwen):
            def ask(self, frames, row):
                """步骤1构造官方任务请求，步骤2注入准确源帧，步骤3原生计分。"""
                text = context(row); pixels = np.stack([np.asarray(im.convert('RGB')) for im in frames]); path = 'mlvu-frozen-input.mp4'
                guard = VideoInputGuard(self.backend.processor, pixels); self.processor = guard
                doc = task_doc(row)
                inst = Instance(request_type='generate_until', arguments=(text, copy.deepcopy(GENERATION),
                    lambda _: [path], 0, 'mlvu_dev_reference', 'test'), idx=0,
                    metadata=dict(task='mlvu_dev_reference', doc_id=0, repeats=1))
                self.task_dict = {'mlvu_dev_reference': {'test': [doc]}}
                try:
                    with frozen_media(pixels, path) as count: answers = self.generate_until([inst])
                    if count != dict(first_frame_reads=1, video_reads=1): raise ValueError('Media was resampled')
                finally: self.processor = self.backend.processor
                score = metric(row, answers)
                return dict(raw_output=answers[0], parsed_answer=score['pred_answer'], observed=guard.observed,
                    framework_metric=score, framework_generate='Qwen2_5_VL.generate_until', media_reads=count)
        return MlvuQwen(backend)
    from videoqa_lmms_bridge.llava_adapter import FrozenInputLlavaVid
    class MlvuLlava(FrozenInputLlavaVid):
        def __init__(self, backend):
            super().__init__(backend); self.max_frames_num = 8

        def load_video(self, path, max_frames_num, fps, force_sample=False):
            """接口：实际K8冻结媒体，禁止旧16帧入口二次抽样或加入时间提示。"""
            if max_frames_num != 8 or self.frozen is None or path != self.frozen['path']: raise ValueError('Unexpected media request')
            self.load_calls += 1
            return np.stack([np.asarray(im.convert('RGB')) for im in self.frozen['frames']]), '', self.frozen['duration']

        def ask(self, frames, times, duration, row, video_path):
            """步骤1原生任务文本，步骤2继承框架视频生成，步骤3使用原生MLVU解析。"""
            from llava.conversation import conv_templates
            text = context(row); doc = task_doc(row)
            conv = conv_templates['qwen_1_5'].copy(); conv.append_message(conv.roles[0], '<image>\n'+text)
            conv.append_message(conv.roles[1], None); prompt = conv.get_prompt()
            self.frozen = dict(frames=frames, duration=duration, path=video_path); before = self.load_calls
            instance = Instance(request_type='generate_until', arguments=(text, copy.deepcopy(GENERATION),
                lambda _: [video_path], 0, 'mlvu_dev_reference', 'test'), idx=0,
                metadata=dict(task='mlvu_dev_reference', doc_id=0, repeats=1))
            self.task_dict = {'mlvu_dev_reference': {'test': [doc]}}
            try: answers = self.generate_until([instance])
            finally: self.frozen = None
            if self.load_calls != before+1: raise ValueError('Repeated media read')
            score = metric(row, answers)
            return dict(raw_output=answers[0], parsed_answer=score['pred_answer'] or None, prompt=prompt,
                framework_metric=score, context=text, framework_generate='LlavaVid.generate_until')
    return MlvuLlava(backend)


def observe(profile, observer, backend, adapter, images, selection, row, state, raw_path, identity):
    """接口：新任务入口不改变任何生成计算；原始返回在计分前立即落盘。"""
    if profile == 'qwen32': answer = observer.observe(backend, adapter, images, row, state, raw_path, identity)
    else:
        result = observer.observe(backend, adapter, images, selection, row, state, raw_path, identity)
        answer = dict(result['answer'], observed=result['observed'])
    return dict(answer, framework_task='mlvu_dev_reference', framework_metric_key='mlvu_percetion_score')


def answer_from_raw(profile, raw, row):
    """接口：已返回的生成只恢复停止清理和任务计分，不加载模型或再次generate。"""
    result = observer_module(profile).result_from_raw(raw, row)
    answer = result if profile == 'qwen32' else dict(result['answer'], observed=result['observed'])
    return dict(answer, framework_task='mlvu_dev_reference', framework_metric_key='mlvu_percetion_score')


def cpu_identity(profile, tools, images, row):
    """接口：完整官方提示/像素/Token身份供事前共享与实际推理双重核验，零模型前向。"""
    import torch
    from videoqa_budget32.observe import tensor_hash
    processor, tokenizer = tools; text = context(row)
    # 步骤1：各后端走已有官方预处理路径，MLVU仅改变任务文字。
    if profile == 'qwen32':
        prompt = processor.apply_chat_template([dict(role='system', content='You are a helpful assistant.'),
            dict(role='user', content=[dict(type='video'), dict(type='text', text=text)])], tokenize=False, add_generation_prompt=True)
        texts = [prompt]
        x = processor(text=texts, videos=[np.stack([np.asarray(im) for im in images])], fps=2.,
            images_kwargs={'size': dict(shortest_edge=100352, longest_edge=602112)}, truncation=False, return_tensors='pt')
        return dict(prompt=texts[0], input_token_ids=x['input_ids'][0].tolist(),
                    pixel_sha256=tensor_hash(x['pixel_values_videos']), video_grid_thw=x['video_grid_thw'].tolist())
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    conv = conv_templates['qwen_1_5'].copy(); conv.append_message(conv.roles[0], '<image>\n'+text); conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return dict(prompt=prompt, input_token_ids=tokenizer_image_token(prompt, tokenizer, -200),
                pixel_sha256=tensor_hash(processor.preprocess(images, return_tensors='pt')['pixel_values'].to(torch.bfloat16)))
