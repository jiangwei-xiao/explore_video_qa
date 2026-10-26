"""双配置两组件全量消融：独立后台协调、固定阶段、调用上限与完成恢复。"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def commands(run_id,gpus):
    """接口：准备两配置全部输入后才进入GPU，阶段均显式携带恢复参数。"""
    from videoqa_generalization import component_ablations as study
    entry=str(Path(__file__).resolve())
    def command(stage,profile=None):
        return [study.PYTHONS[profile or 'qwen32'],entry,'--run-id',run_id,'--stage',stage,
                '--gpus',*gpus,'--resume',*(['--profile',profile] if profile else [])]
    return [('offline',command('offline')),
            ('prepare_llava8',command('prepare','llava8')),('prepare_qwen32',command('prepare','qwen32')),
            ('answer_llava8',command('answer','llava8')),('answer_qwen32',command('answer','qwen32')),
            ('report',command('report')),('audit',command('audit'))]


def main():
    """接口：只提供阶段、固定配置、GPU及恢复，不开放算法参数或测试题挑选。"""
    from videoqa_runtime.common import read_json,sha256
    from videoqa_full.state import durable,exclusive,utc
    from videoqa_generalization import component_ablations as study
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=['all','offline','controls','prepare','answer','report','audit'],default='all')
    parser.add_argument('--profile',choices=study.PROFILES);parser.add_argument('--gpus',nargs='+',default=list(map(str,range(8))))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id):raise ValueError('Invalid run ID')
    if not args.gpus or len(set(args.gpus))!=len(args.gpus) or not set(args.gpus)<=set(map(str,range(8))):raise ValueError('Invalid GPU scope')
    if args.stage in ('prepare','answer') and args.profile is None:parser.error('Profile is required')
    root=ROOT/'outputs/full_videoqa'/args.run_id;root.mkdir(parents=True,exist_ok=True)
    if any((root/'qa'/p/'calls').exists() for p in study.PROFILES) and not args.resume:
        raise ValueError('Existing formal calls require explicit resume')
    from videoqa_generalization import component_ablation_runtime as runtime
    from videoqa_generalization import component_ablation_report as report
    if args.stage=='all':
        with exclusive(root/'pipeline.lock'):
            if (root/'completion_audit.json').exists():
                for p in study.PROFILES:runtime.audit(root/'qa'/p)
                return
            durable(root/'launcher.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            for name,command in commands(args.run_id,args.gpus):
                durable(root/'pipeline_status.json',dict(status='running',stage=name,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                try:subprocess.run(command,cwd=ROOT,check=True)
                except BaseException as exc:
                    durable(root/'pipeline_status.json',dict(status='paused',stage=name,error=repr(exc),utc=utc()));raise
            durable(root/'pipeline_status.json',dict(status='completed',utc=utc()))
    elif args.stage in ('offline','controls'):study.offline(root,controls_only=args.stage=='controls')
    elif args.stage=='prepare':runtime.prepare(root,args.profile,args.gpus)
    elif args.stage=='answer':runtime.execute(root/'qa'/args.profile)
    elif args.stage=='report':report.summarize(root);report.render_panel(root)
    else:
        accepts={p:runtime.audit(root/'qa'/p) for p in study.PROFILES}
        total=sum(v['physical_new_calls'] for v in accepts.values());synthetic=sum(v['synthetic_calls'] for v in accepts.values())
        if total>10800 or synthetic>21:raise ValueError('Experiment budget exceeded')
        durable(root/'completion_audit.json',dict(status='passed',method_results=10800,
            new_real_qa=total,synthetic_generations=synthetic,profile_audits=accepts,
            offline_summary_sha256=sha256(root/'offline_summary.json'),summary_sha256=sha256(root/'summary.json'),
            new_blip_calls=0,new_selection_visual_calls=0,automatic_default_replacement=False,utc=utc()))


if __name__=='__main__':
    main()
