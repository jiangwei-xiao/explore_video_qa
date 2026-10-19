"""LongVideo-Reason四选项独立任务桥：保留乱序字母，实际生成经过LMMs-Eval。"""
import copy
import importlib.util
import re
import numpy as np

from videoqa_runtime.common import ROOT
from videoqa_lmms_bridge.common import GENERATION

POST_PROMPT="\nAnswer with the option's letter from the given choices directly."


def public_doc(row):
    """接口：任务文字只含公开问题/原序选项；不传播来源reasoning、参考答案或字幕。"""
    if len(row['options'])!=4 or set(row['option_letters'])!=set('ABCD') or len(set(row['options']))!=4:
        raise ValueError('Expected four distinct original options with original labels')
    return dict(question=row['question'],options=list(row['options']),option_letters=list(row['option_letters']))


def context(row):
    """接口：逐行保留原展示字母与顺序，答案B始终指B项，而不是列表第二项。"""
    value=public_doc(row)
    return value['question']+'\n'+'\n'.join(f'{label}. {option}' for label,option in zip(value['option_letters'],value['options']))+POST_PROMPT


def strict_option(raw):
    """接口：只接受完整A–D、A.或(A)，不从解释文字、答案标签或其他字母中猜测。"""
    match=re.fullmatch(r'\s*(?:\(([A-D])\)|([A-D])\.?)\s*',raw)
    return (match.group(1) or match.group(2)) if match else None


def metric(row,raw):
    """接口：只有原始生成已经返回后才访问计分标签；参考字母不按展示位置重映射。"""
    pred=strict_option(raw)
    if 'answer_letter' not in row:
        # 步骤1：真实生成路径延迟到原文返回后的计分边界，才载入该题参考字母。
        from videoqa_set_value.longvideo_runtime import scoring_row
        row=scoring_row(row['_scoring_root'],row)
    if row['answer_letter'] not in 'ABCD' or len(row['answer_letter'])!=1:raise ValueError('Invalid scoring-only answer letter')
    return dict(pred_answer=pred,correct=pred==row['answer_letter'],parse_failed=pred is None,
        scoring_protocol='longvideo-reason-value-strict-A-D-v1')


def _bridge():
    """接口：只私有绑定已有观测/CPU工具，冻结NExT-QA模块及任务全局不被修改。"""
    path=ROOT/'src/videoqa_set_value/task.py'
    spec=importlib.util.spec_from_file_location('longvideo_value_private_bridge',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.context=context;module.public_doc=public_doc;module.metric=metric;module.strict_option=strict_option
    return module


def cpu_tools():
    """接口：官方CPU预处理器/Tokenizer，不加载视觉或语言模型权重。"""
    return _bridge().cpu_tools()


def cpu_identity(tools,frames,row):
    """接口：本任务完整提示Token和8帧BF16像素身份，用于事前共享与实际生成验收。"""
    return _bridge().cpu_identity(tools,frames,row)


def make_adapter(backend):
    """接口：仅替换冻结媒体读取；模型推理真正执行固定LMMs-Eval的generate_until。"""
    from videoqa_lmms_bridge.llava_adapter import FrozenInputLlavaVid
    from lmms_eval.api.instance import Instance
    class LongvideoLlava(FrozenInputLlavaVid):
        """本轮独立四选项任务，不冒用Video-MME/NExT-QA任务文字或计分。"""
        def __init__(self,backend):super().__init__(backend);self.max_frames_num=8

        def load_video(self,path,max_frames_num,fps,force_sample=False):
            """步骤1：只读传入8张有序RGB；不重新采样或增加时间说明。"""
            if max_frames_num!=8 or self.frozen is None or path!=self.frozen['path']:raise ValueError('Unexpected media read')
            self.load_calls+=1
            return np.stack([np.asarray(im.convert('RGB')) for im in self.frozen['frames']]),'',self.frozen['duration']

        def ask(self,frames,times,duration,row,video_path):
            """步骤2：固定文字→原生框架生成→原始返回持久化后的独立严格计分。"""
            from llava.conversation import conv_templates
            text=context(row);conv=conv_templates['qwen_1_5'].copy()
            conv.append_message(conv.roles[0],'<image>\n'+text);conv.append_message(conv.roles[1],None)
            self.frozen=dict(frames=frames,duration=duration,path=video_path);before=self.load_calls
            instance=Instance(request_type='generate_until',arguments=(text,copy.deepcopy(GENERATION),
                lambda _:[video_path],0,'longvideo_set_value_diagnostic','test'),idx=0,
                metadata=dict(task='longvideo_set_value_diagnostic',doc_id=0,repeats=1))
            self.task_dict={'longvideo_set_value_diagnostic':{'test':[public_doc(row)]}}
            try:answers=self.generate_until([instance])
            finally:self.frozen=None
            if self.load_calls!=before+1 or len(answers)!=1:raise ValueError('Repeated media/generation')
            score=metric(row,answers[0])
            return dict(raw_output=answers[0],parsed_answer=score['pred_answer'],prompt=conv.get_prompt(),
                framework_metric=score,context=text,framework_generate='LlavaVid.generate_until')
    return LongvideoLlava(backend)


def observe(backend,adapter,images,selection,row,state,raw_path,identity):
    """接口：继承真实generate钩子和返回检查点，仅替换本轮任务公开文字/后处理。"""
    result=_bridge().observe(backend,adapter,images,selection,row,state,raw_path,identity)
    return dict(result,framework_task='longvideo_set_value_diagnostic')


def answer_from_raw(raw,row):
    """接口：完整返回只恢复解析，不重新进入模型；任务隔离与首次生成一致。"""
    return dict(_bridge().answer_from_raw(raw,row),framework_task='longvideo_set_value_diagnostic')
