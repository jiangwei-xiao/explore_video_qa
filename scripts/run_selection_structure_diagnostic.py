"""同粗池双配置诊断：独立后台协调，不自动运行全量QA或新方法。"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def commands(run_id,gpus):
    """接口：全部CPU输入先冻结，然后依次LLaVA8、Qwen32；子阶段显式携带恢复。"""
    from videoqa_audit import selection_structure_run as study
    entry=str(Path(__file__).resolve())
    def command(stage,profile=None):
        return [study.PYTHONS[profile or 'qwen32'],entry,'--run-id',run_id,'--stage',stage,
                '--gpus',*gpus,'--resume',*(['--profile',profile] if profile else [])]
    return [('offline',command('offline')),('prepare_llava8',command('prepare','llava8')),
        ('prepare_qwen32',command('prepare','qwen32')),('freeze_calls',command('freeze-calls')),
        ('answer_llava8',command('answer','llava8')),('answer_qwen32',command('answer','qwen32')),
        ('report',command('report')),('panel',command('panel','llava8')),('audit',command('audit'))]


def freeze_calls(root):
    """接口：两后端的全部导入/共享/新生成分区完成后发布全局上限U≤800。"""
    from videoqa_audit import selection_structure_run as study
    specs={p:study.read(root/'qa'/p/'protocol.json') for p in study.PROFILES}
    maximum=sum(s['max_new_calls'] for s in specs.values())
    if maximum>800: raise ValueError('Global real-generation cap exceeded')
    value=dict(max_new_calls=maximum,max_synthetic_qa=8,profiles={p:dict(
        protocol_sha256=study.sha(root/'qa'/p/'protocol.json'),max_new_calls=s['max_new_calls']) for p,s in specs.items()})
    study.freeze(root/'call_plan.json',value)
    return value


def audit(root):
    """接口：完成恢复核验产物、预算和引用闭合；本分支禁止加载问答模型。"""
    from videoqa_audit import selection_structure_run as study
    from videoqa_audit import selection_structure_runtime as runtime
    offline=study.verify_offline(root); plan=study.read(root/'call_plan.json')
    profiles={p:runtime.audit(root/'qa'/p) for p in study.PROFILES}
    total=sum(v['new_real_qa'] for v in profiles.values());synthetic=sum(v['synthetic_qa'] for v in profiles.values())
    if total!=plan['max_new_calls'] or total>800 or synthetic>8: raise ValueError('Global physical ledger differs')
    for p in profiles:
        if study.sha(root/'qa'/p/'protocol.json')!=plan['profiles'][p]['protocol_sha256']: raise ValueError('Frozen input partition changed')
    summary=study.read(root/'summary.json')
    for p,h in summary['source_result_hashes'].items():
        if study.sha(p)!=h: raise ValueError('Report source changed')
    value=dict(status='passed',parent_replays=offline['parent_replays'],control_selections=offline['control_selections'],
        method_results=1200,new_real_qa=total,synthetic_qa=synthetic,profiles=profiles,
        call_plan_sha256=study.sha(root/'call_plan.json'),summary_sha256=study.sha(root/'summary.json'),
        semantic_review=study.read(root/'review/panel.json')['semantic_review'],new_blip_calls=0,new_visual_calls=0)
    study.freeze(root/'completion_audit.json',value)
    return value


def main():
    """接口：仅提供阶段、固定配置、GPU范围和恢复；不开放规则或题目挑选。"""
    from videoqa_audit import selection_structure_run as study
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id');parser.add_argument('--stage',default='preflight',
        choices=['preflight','all','offline','prepare','freeze-calls','answer','report','panel','audit'])
    parser.add_argument('--profile',choices=study.PROFILES);parser.add_argument('--gpus',nargs='+',default=list(map(str,range(8))))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.stage=='preflight': print(json.dumps(study.preflight(),ensure_ascii=False,indent=2));return
    if not args.run_id or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id): parser.error('A stable run-id is required')
    if len(args.gpus)!=len(set(args.gpus)) or not set(args.gpus)<=set(map(str,range(8))): parser.error('Invalid GPU scope')
    if args.stage in ('prepare','answer') and args.profile is None: parser.error('Profile is required')
    root=ROOT/'outputs/diagnostics'/args.run_id;root.mkdir(parents=True,exist_ok=True)
    if list((root/'qa').glob('*/calls/*/*.json')) and not args.resume: raise ValueError('Existing calls require explicit resume')
    # 步骤1文件/来源未就绪就停止，不自动下载模型、改环境或补建候选。
    ready=study.preflight()
    if ready['status']=='not_ready': raise RuntimeError(json.dumps(ready,ensure_ascii=False))
    from videoqa_audit import selection_structure_runtime as runtime
    from videoqa_audit import selection_structure_report as report
    if args.stage=='all':
        from videoqa_full.state import exclusive,durable
        with exclusive(root/'pipeline.lock'):
            if (root/'completion_audit.json').exists(): print(json.dumps(audit(root)));return
            durable(root/'launcher.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=study.utc()))
            # 步骤2子阶段用自己的冻结环境；协调器持续心跳，不依赖SSH观察连接。
            for name,command in commands(args.run_id,args.gpus):
                with (root/'pipeline.log').open('a',buffering=1) as log:
                    process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        durable(root/'pipeline_status.json',dict(status='running',stage=name,pid=os.getpid(),sid=os.getsid(0),
                            child_pid=process.pid,utc=study.utc()));time.sleep(5)
                    if process.returncode!=0:
                        durable(root/'pipeline_status.json',dict(status='paused',stage=name,exit_code=process.returncode,utc=study.utc()))
                        raise RuntimeError('Stage paused: '+name)
            durable(root/'pipeline_status.json',dict(status='completed',semantic_review='pending',utc=study.utc()))
    elif args.stage=='offline': study.offline(root)
    elif args.stage=='prepare': runtime.prepare(root,args.profile,args.gpus)
    elif args.stage=='freeze-calls': freeze_calls(root)
    elif args.stage=='answer':
        plan=study.read(root/'call_plan.json')
        if study.sha(root/'qa'/args.profile/'protocol.json')!=plan['profiles'][args.profile]['protocol_sha256']:
            raise ValueError('All-profile frozen call plan differs')
        runtime.execute(root/'qa'/args.profile)
    elif args.stage=='report': report.summarize(root)
    elif args.stage=='panel': report.render_panel(root)
    else: print(json.dumps(audit(root)))


if __name__=='__main__': main()
