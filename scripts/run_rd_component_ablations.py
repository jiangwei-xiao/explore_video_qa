"""双配置两组件全量消融：独立后台协调、固定阶段、调用上限与完成恢复。"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def reuse_offline(root,source):
    """接口：仅导入未开始QA、选择关键源码相同的完整离线产物；保留原尝试不覆盖。"""
    from videoqa_runtime.common import read_json,sha256
    from videoqa_full.state import durable,rows,utc
    from videoqa_generalization import component_ablations as study
    root,source=Path(root),Path(source)
    if root==source:raise ValueError('Reuse source must be a separate run')
    old=read_json(source/'offline_protocol.json');summary=read_json(source/'offline_summary.json')
    if list((source/'qa').glob('*/calls/*/*.json')):raise ValueError('Source already consumed real QA; offline-only import forbidden')
    if old['task']!='rd_component_ablations' or old['methods']!=study.CONTROLS or old['question_ids']!=[r['question_id'] for r in rows()]:
        raise ValueError('Offline source experiment identity differs')
    if summary['status']!='passed' or summary['parent_replays']!=5400 or summary['control_selections']!=10800:
        raise ValueError('Incomplete offline source cannot be reused')
    current={p:sha256(ROOT/p) for p in study.CORE_FILES}
    engineering={'scripts/run_rd_component_ablations.py','src/videoqa_generalization/component_ablation_runtime.py',
                 'src/videoqa_generalization/component_ablation_report.py'}
    for p,h in current.items():
        if p not in engineering and old['code_sha256'][p]!=h:
            raise ValueError('Selection-critical implementation changed: '+p)
    spec=dict(old,code_sha256=current)
    study.freeze(root/'offline_protocol.json',spec)
    if (root/'offline_summary.json').exists():return
    began=time.perf_counter()
    # 步骤1完整校验源导出并复制到独立目录，不修改候选、前缀、配额或任何旧记录。
    for relative,expected in summary['exports'].items():
        path=source/relative
        if not path.resolve().is_relative_to(source.resolve()) or sha256(path)!=expected:
            raise ValueError('Verified offline export changed')
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha256(target)!=expected:raise ValueError('Partial reused export differs')
        else:shutil.copyfile(path,target)
    for profile in study.PROFILES:
        for path in (source/'offline'/profile).glob('*.json'):
            target=root/'offline'/profile/path.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    # 步骤2新运行绑定当前工程实现，原计算成本保留为来源，不伪装成新选择耗时。
    for relative in current:
        target=root/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
    value=dict(summary,protocol_sha256=sha256(root/'offline_protocol.json'),
               original_offline_wall_seconds=summary['seconds'],seconds=time.perf_counter()-began,
               reused_from=str(source),source_summary_sha256=sha256(source/'offline_summary.json'))
    durable(root/'offline_summary.json',value)
    durable(root/'offline_reuse_acceptance.json',dict(status='passed',source=str(source),
        source_qa_calls=0,parent_replays=5400,control_selections=10800,selection_code_exact=True,
        new_model_calls=0,reuse_seconds=time.perf_counter()-began,utc=utc()))


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
    parser.add_argument('--reuse-offline-run')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id):raise ValueError('Invalid run ID')
    if not args.gpus or len(set(args.gpus))!=len(args.gpus) or not set(args.gpus)<=set(map(str,range(8))):raise ValueError('Invalid GPU scope')
    if args.stage in ('prepare','answer') and args.profile is None:parser.error('Profile is required')
    root=ROOT/'outputs/full_videoqa'/args.run_id;root.mkdir(parents=True,exist_ok=True)
    if args.reuse_offline_run:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.reuse_offline_run):raise ValueError('Invalid offline source run')
        reuse_offline(root,ROOT/'outputs/full_videoqa'/args.reuse_offline_run)
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
