"""接续冻结验证的剩余阶段；不改已完成Qwen协议、代码指纹或准备成本。"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_generalization.inputs import frozen
from videoqa_generalization.run import QPY,LPY,verify,audit


def completed_resume(root):
    """接口：已完成任务只核验原始文件，跳过准备、模型加载及人工报告重写。"""
    folders=['results','calls','raw_generations']
    before={str(p):sha256(p) for stage in ['qwen16','llava16','wfs32'] for folder in folders
            for p in (root/stage/folder).glob('*/*.json')}
    # 步骤1：调用现有完成审计；缺失或不确定记录直接停止，不进入新生成。
    for stage in ['qwen16','llava16','wfs32']:audit(root/stage)
    after={str(p):sha256(p) for stage in ['qwen16','llava16','wfs32'] for folder in folders
           for p in (root/stage/folder).glob('*/*.json')}
    # 步骤2：总共23900文件保持不变，人工归档报告不参与自动重写。
    if before!=after:raise ValueError('Completed resume changed generation files')
    durable(root/'recovery/completed_resume.json',dict(status='passed',unchanged_files=len(before),
        extra_generations=0,preparation_replayed=False,report_rewritten=False,utc=utc()))


def remaining_commands(run_id, gpus):
    """接口：只接续LLaVA16、WFS32和报告；每个子阶段显式恢复，不重做输入或Qwen。"""
    common=['--run-id',run_id,'--gpus',*gpus,'--resume']
    entry=str(ROOT/'scripts/run_exp12_generalization_full.py')
    return [(stage,[python,entry,'--stage',stage,*common]) for stage,python in
            [('llava16',LPY),('wfs32',QPY),('report',QPY)]]


def main():
    """接口：核验已完成2700条来源后后台接续；未知生成仍由原账本拒绝重答。"""
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--gpus',nargs='+',default=list(map(str,range(8))));args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id):raise ValueError('Invalid run ID')
    if not args.gpus or len(set(args.gpus))!=len(args.gpus) or not set(args.gpus)<=set(map(str,range(8))):
        raise ValueError('Invalid GPU scope')
    root=ROOT/'outputs/full_videoqa'/args.run_id
    with exclusive(root/'pipeline.lock'):
        if read_json(root/'pipeline_status.json')['status']=='completed':
            completed_resume(root);return
        # 步骤1：保留暂停原因与原始准备成本，已完成组所有生成文件作为恢复对照。
        spec=verify(root/'qwen16');accepted=read_json(root/'qwen16/completion_audit.json')
        if accepted['status']!='passed' or accepted['new_calls']!=2700 or accepted['extra_generations']!=0:
            raise ValueError('Qwen completion not accepted')
        if args.gpus!=spec['allowed_gpus']:raise ValueError('Recovery GPU scope changed')
        before={str(p):sha256(p) for folder in ['results','calls','raw_generations']
                for p in (root/'qwen16'/folder).glob('*/*.json')}
        if len(before)!=8100:raise ValueError('Qwen source records incomplete')
        frozen(root/'recovery/paused_before_resume.json',read_json(root/'pipeline_status.json'))
        frozen(root/'recovery/qwen_completed_sources.json',before)
        durable(root/'recovery/launcher.json',dict(pid=os.getpid(),sid=os.getsid(0),
            code_sha256=sha256(__file__),reason='Explicit stage resume flag for already populated run',utc=utc()))
        # 步骤2：调用原已冻结入口；代码/模型/输入校验仍执行，不绕过生成安全检查。
        for stage,command in remaining_commands(args.run_id,args.gpus):
            durable(root/'pipeline_status.json',dict(status='running',stage=stage,pid=os.getpid(),sid=os.getsid(0),recovery=True,utc=utc()))
            try:subprocess.run(command,cwd=ROOT,check=True)
            except BaseException as exc:
                durable(root/'pipeline_status.json',dict(status='paused',stage=stage,error=repr(exc),recovery=True,utc=utc()));raise
        # 步骤3：恢复前后Qwen8100份文件必须不变；后两组由原完成审计及报告验收。
        if any(sha256(path)!=digest for path,digest in before.items()):raise ValueError('Completed Qwen answers changed')
        durable(root/'recovery/completion.json',dict(status='passed',qwen_unchanged_files=8100,
            extra_qwen_generations=0,utc=utc()))
        durable(root/'pipeline_status.json',dict(status='completed',new_answer_calls=7900,recovery=True,utc=utc()))


if __name__=='__main__':main()
