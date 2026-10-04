"""固定四道当前文字错题，先检查现有32帧，不重答、不按结果替换题目。"""
import argparse
from pathlib import Path

import probe_text_panel as panel
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable

QUESTIONS=['292-3','022-1','331-1','648-3']


def main(run,stage):
    """接口：复用原输入读取与OCR，不传入答案；音频依赖仅在分析元数据中标记。"""
    # 步骤1：零人工目标，只处理四题当前选中帧；模型不接收来源标签。
    panel.TARGETS={q:[] for q in QUESTIONS}
    if stage=='prepare':
        panel.prepare(run)
        path=run/'inputs.json'
        spec=read_json(path)
        # 步骤2：正式识别前补齐独立入口身份和当前面板范围，防止混用历史12题。
        key=str(Path(__file__).resolve())
        if key not in spec['sources']:
            spec['sources'][key]=sha256(key)
            spec['fixed_questions']=QUESTIONS
            spec['attribution_limits']={'331-1':'问题要求音频未提及；当前无音频协议，不能单独归因选帧'}
            durable(path,spec)
        if len(spec['entries'])!=128 or any(r['diagnostic_target'] for r in spec['entries']):
            raise ValueError('四题当前输入不符')
    else:
        # 步骤3：只用冻结图像运行原默认OCR，问答、BLIP和范围分类均不调用。
        spec=panel.check_inputs(run)
        if spec.get('fixed_questions')!=QUESTIONS:
            raise ValueError('未冻结当前四题范围')
        panel.recognize(run)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','recognize'],required=True)
    args=parser.parse_args();main(args.run,args.stage)
