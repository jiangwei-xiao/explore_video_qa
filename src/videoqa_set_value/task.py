"""NExT-QA五选项诊断桥接：官方文字、严格解析、原生LMMs生成与原文恢复。"""
import ast
import copy
import importlib.util
import re
import numpy as np

from videoqa_runtime.common import ROOT
from videoqa_lmms_bridge.common import GENERATION,LMMS

POST_PROMPT="\nAnswer with the option's letter from the given choices directly."
_FORMATTER=None


def strict_option(raw):
    """接口：只接受完整A–E、A.或(A)，其他记失败；绝不随机回退。"""
    match=re.fullmatch(r'\s*(?:\(([A-E])\)|([A-E])\.?)\s*',raw)
    return (match.group(1) or match.group(2)) if match else None


def public_doc(row):
    """接口：提供给模型框架的任务对象仅有原题和五选项，不包含标准答案。"""
    if len(row['options'])!=5:raise ValueError('NExT-QA requires five original options')
    return dict(question=row['question'],**{'a'+str(i):v for i,v in enumerate(row['options'])})


def context(row):
    """接口：直接执行固定框架的文字函数；隔离无关NLP下载与随机计分。"""
    global _FORMATTER
    if _FORMATTER is None:
        path=LMMS/'lmms_eval/tasks/nextqa/utils.py';tree=ast.parse(path.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='nextqa_doc_to_text_mc')
        namespace={'OPTIONS':list('ABCDE')}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        _FORMATTER=namespace['nextqa_doc_to_text_mc']
    return _FORMATTER(public_doc(row),dict(pre_prompt='',post_prompt=POST_PROMPT))


def metric(row,raw):
    """接口：生成已返回后才读取标准答案；解析失败计错，不制造随机监督。"""
    pred=strict_option(raw)
    return dict(pred_answer=pred,correct=pred=='ABCDE'[row['answer_index']],parse_failed=pred is None,
        scoring_protocol='nextqa-value-diagnostic-strict-A-E-v1')


class TaskBridge:
    """兼容旧观测器内部任务键；实际上下文与计分均为本轮独立NExT-QA诊断。"""
    def __init__(self,row):self.row=row

    def videomme_doc_to_text(self,doc,kwargs):
        """接口：文字函数只读取公开原题/选项，标准答案不会进入prompt。"""
        return context(self.row)

    def videomme_process_results(self,doc,answers):
        """接口：只在原始返回已落盘后的后处理调用严格诊断计分。"""
        return {'videomme_perception_score':metric(self.row,answers[0])}


def observer_module(row):
    """接口：私有模块重绑定任务依赖，复用生成钩子，不修改旧冻结文件或全局模块。"""
    path=ROOT/'src/videoqa_generalization/llava_observe.py'
    spec=importlib.util.spec_from_file_location('nextqa_value_private_observer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.load_task=lambda:TaskBridge(row);module.task_doc=public_doc;module.POST_PROMPT=POST_PROMPT
    return module


def make_adapter(backend):
    """接口：唯一媒体替换为已冻结8帧，推理真正经过LMMs-Eval原生generate_until。"""
    from videoqa_lmms_bridge.llava_adapter import FrozenInputLlavaVid
    from lmms_eval.api.instance import Instance
    class NextqaLlava(FrozenInputLlavaVid):
        def __init__(self,backend):super().__init__(backend);self.max_frames_num=8

        def load_video(self,path,max_frames_num,fps,force_sample=False):
            """步骤1：只读指定RGB，不重新抽帧，不加入时长/PTS说明。"""
            if max_frames_num!=8 or self.frozen is None or path!=self.frozen['path']:raise ValueError('Unexpected media read')
            self.load_calls+=1
            return np.stack([np.asarray(im.convert('RGB')) for im in self.frozen['frames']]),'',self.frozen['duration']

        def ask(self,frames,times,duration,row,video_path):
            """步骤2：官方任务文字→原生生成→严格解析；原始返回由外层钩子先行保存。"""
            from llava.conversation import conv_templates
            text=context(row);conv=conv_templates['qwen_1_5'].copy()
            conv.append_message(conv.roles[0],'<image>\n'+text);conv.append_message(conv.roles[1],None)
            self.frozen=dict(frames=frames,duration=duration,path=video_path);before=self.load_calls
            instance=Instance(request_type='generate_until',arguments=(text,copy.deepcopy(GENERATION),
                lambda _:[video_path],0,'nextqa_value_diagnostic','test'),idx=0,
                metadata=dict(task='nextqa_value_diagnostic',doc_id=0,repeats=1))
            self.task_dict={'nextqa_value_diagnostic':{'test':[public_doc(row)]}}
            try:answers=self.generate_until([instance])
            finally:self.frozen=None
            if self.load_calls!=before+1 or len(answers)!=1:raise ValueError('Repeated media/generation')
            score=metric(row,answers[0])
            return dict(raw_output=answers[0],parsed_answer=score['pred_answer'],prompt=conv.get_prompt(),
                framework_metric=score,context=text,framework_generate='LlavaVid.generate_until')
    return NextqaLlava(backend)


def cpu_tools():
    """接口：只加载官方Tokenizer/384预处理器，不加载权重或执行模型前向。"""
    from transformers import AutoTokenizer
    from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
    return SigLipImageProcessor(),AutoTokenizer.from_pretrained('/home/models/lmms-lab/LLaVA-Video-7B-Qwen2',local_files_only=True)


def cpu_identity(tools,frames,row):
    """接口：完整提示Token与BF16视觉张量身份；用于生成前共享和实际输入核对。"""
    import torch
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from videoqa_lmms_bridge.llava_adapter import tensor_hash
    processor,tokenizer=tools;conv=conv_templates['qwen_1_5'].copy()
    conv.append_message(conv.roles[0],'<image>\n'+context(row));conv.append_message(conv.roles[1],None)
    prompt=conv.get_prompt();pixels=processor.preprocess(frames,return_tensors='pt')['pixel_values'].to(torch.bfloat16)
    return dict(prompt=prompt,input_token_ids=tokenizer_image_token(prompt,tokenizer,-200),
        pixel_sha256=tensor_hash(pixels),source_frames=len(frames),shape=list(pixels.shape),visual_tokens=len(frames)*210)


def observe(backend,adapter,images,selection,row,state,raw_path,identity):
    """接口：共享冻结数值路径；一次生成原文先落盘，随后才形成训练标签。"""
    result=observer_module(row).observe(backend,adapter,images,selection,row,state,raw_path,identity)
    return dict(result['answer'],observed=result['observed'],framework_task='nextqa_value_diagnostic')


def answer_from_raw(raw,row):
    """接口：完整原始返回只恢复后处理，不加载模型或再次问答。"""
    result=observer_module(row).result_from_raw(raw,row)
    return dict(result['answer'],observed=result['observed'],framework_task='nextqa_value_diagnostic')
