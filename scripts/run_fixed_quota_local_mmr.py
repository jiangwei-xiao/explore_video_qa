"""固定区域实际配额H对照：后台完成400题，不自动接续新实验。"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def commands(run_id,gpus):
    """接口：两输入和全局额度先冻结；LL→Q依次执行，所有子阶段显式恢复。"""
    from videoqa_audit import fixed_quota_local_mmr_run as study
    entry=str(Path(__file__).resolve())
    def cmd(stage,profile=None):
        return [study.PYTHONS[profile or 'qwen32'],entry,'--run-id',run_id,'--stage',stage,'--gpus',*gpus,'--resume',*(['--profile',profile] if profile else [])]
    return [(s,cmd(s,p)) for s,p in [('offline',None),('prepare','llava8'),('prepare','qwen32'),
        ('freeze-calls',None),('answer','llava8'),('answer','qwen32'),('report',None),('panel','llava8'),('audit',None)]]


def freeze_calls(root):
    """接口：完整400输入先冻结后发布U≤400，禁止运行中增加额度。"""
    from videoqa_audit import fixed_quota_local_mmr_run as study
    specs={p:study.read(root/'qa'/p/'protocol.json') for p in study.PROFILES}
    maximum=sum(s['max_new_calls'] for s in specs.values())
    if maximum>400:raise ValueError('Global real-generation cap exceeded')
    value=dict(max_new_calls=maximum,max_synthetic_qa=8,profiles={p:dict(
        protocol_sha256=study.sha(root/'qa'/p/'protocol.json'),max_new_calls=s['max_new_calls']) for p,s in specs.items()})
    study.freeze(root/'call_plan.json',value);return value


def audit(root):
    """接口：完成恢复不调用模型或选择器，仅核对来源、额度及原文计分。"""
    from videoqa_audit import fixed_quota_local_mmr_run as study
    from videoqa_audit import fixed_quota_local_mmr_runtime as runtime
    offline=study.verify_offline(root);plan=study.read(root/'call_plan.json')
    profiles={p:runtime.audit(root/'qa'/p) for p in study.PROFILES}
    total=sum(v['new_real_qa'] for v in profiles.values())
    if total!=plan['max_new_calls'] or total>400: raise ValueError('Global call population differs')
    for p in profiles:
        if study.sha(root/'qa'/p/'protocol.json')!=plan['profiles'][p]['protocol_sha256']:raise ValueError('Frozen calls changed')
    summary=study.read(root/'summary.json')
    for p,h in summary['source_result_hashes'].items():
        if study.sha(p)!=h:raise ValueError('Reported source changed')
    value=dict(status='passed',parent_reconstructions=offline['parent_reconstructions'],control_selections=400,
        control_results=400,parent_imports=400,new_real_qa=total,synthetic_qa=8,profiles=profiles,
        call_plan_sha256=study.sha(root/'call_plan.json'),summary_sha256=study.sha(root/'summary.json'),
        semantic_review=study.read(root/'review/panel.json')['semantic_review'],new_blip_calls=0,new_visual_calls=0)
    study.freeze(root/'completion_audit.json',value);return value


def main():
    """接口：仅开放阶段、固定配置、运行ID及GPU，不开放候选参数或挑题。"""
    from videoqa_audit import fixed_quota_local_mmr_run as study
    from videoqa_audit import fixed_quota_local_mmr_runtime as runtime
    from videoqa_audit import fixed_quota_local_mmr_report as report
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',default='all',choices=['all','offline','prepare','freeze-calls','answer','report','panel','audit'])
    parser.add_argument('--profile',choices=study.PROFILES);parser.add_argument('--gpus',nargs='+',default=list(map(str,range(8))))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id):parser.error('Stable run-id required')
    if len(args.gpus)!=len(set(args.gpus)) or not set(args.gpus)<=set(map(str,range(8))):parser.error('Invalid GPU scope')
    if args.stage in ('prepare','answer') and args.profile is None:parser.error('Profile required')
    root=ROOT/'outputs/diagnostics'/args.run_id;root.mkdir(parents=True,exist_ok=True)
    if list((root/'qa').glob('*/calls/*/*.json')) and not args.resume:raise ValueError('Existing calls require explicit resume')
    if args.stage=='all':
        from videoqa_full.state import exclusive,durable
        with exclusive(root/'pipeline.lock'):
            if (root/'completion_audit.json').exists():print(json.dumps(audit(root)));return
            durable(root/'launcher.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=study.utc()))
            # 步骤1子阶段各用自己的冻结环境；步骤2持续心跳，退出码异常就暂停。
            for stage,command in commands(args.run_id,args.gpus):
                with (root/'pipeline.log').open('a',buffering=1) as log:
                    process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        durable(root/'pipeline_status.json',dict(status='running',stage=stage,child_pid=process.pid,
                            pid=os.getpid(),sid=os.getsid(0),utc=study.utc()));time.sleep(5)
                    if process.returncode!=0:
                        durable(root/'pipeline_status.json',dict(status='paused',stage=stage,exit_code=process.returncode,utc=study.utc()))
                        raise RuntimeError('Stage paused: '+stage)
            durable(root/'pipeline_status.json',dict(status='completed',semantic_review='pending',utc=study.utc()))
    elif args.stage=='offline':study.offline(root)
    elif args.stage=='prepare':runtime.prepare(root,args.profile,args.gpus)
    elif args.stage=='freeze-calls':freeze_calls(root)
    elif args.stage=='answer':
        plan=study.read(root/'call_plan.json')
        if study.sha(root/'qa'/args.profile/'protocol.json')!=plan['profiles'][args.profile]['protocol_sha256']:
            raise ValueError('All-profile frozen call plan differs')
        runtime.execute(root/'qa'/args.profile)
    elif args.stage=='report':report.summarize(root)
    elif args.stage=='panel':report.render_panel(root)
    else:print(json.dumps(audit(root)))


if __name__=='__main__':main()
