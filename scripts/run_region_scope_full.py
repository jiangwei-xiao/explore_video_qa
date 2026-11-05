"""exp27双配置全量入口：冻结输入后后台顺序问答，完成恢复只做审计。"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def commands(run_id,gpus):
    """接口：两配置均完成输入冻结后才问答；每个子阶段显式传递恢复标志。"""
    from videoqa_generalization import region_scope_full as study
    entry=str(Path(__file__).resolve())
    def command(stage,profile=None):
        return [study.PYTHONS[profile or 'qwen32'],entry,'--run-id',run_id,'--stage',stage,'--gpus',*gpus,'--resume',*(['--profile',profile] if profile else [])]
    return [('offline',command('offline')),('prepare_qwen32',command('prepare','qwen32')),
        ('prepare_llava8',command('prepare','llava8')),('input_acceptance',command('inputs')),
        ('answer_qwen32',command('answer','qwen32')),('answer_llava8',command('answer','llava8')),
        ('report',command('report')),('audit',command('audit'))]


def complete(root,publish=False):
    """接口：完整性与源指纹审计；完成标记发布最后，恢复不能新增选择或问答。"""
    from videoqa_runtime.common import read_json,sha256
    from videoqa_generalization import region_scope_full as study
    from videoqa_generalization import region_scope_full_runtime as runtime
    root=Path(root);path=root/'completion_audit.json'
    # 步骤1两配置独立验证物理调用与所有引用，再核对完成产物绑定。
    accept={p:runtime.audit(root/'qa'/p) for p in study.PROFILES}
    if sum(v['physical_new_calls'] for v in accept.values())!=1096 or sum(v['synthetic_calls'] for v in accept.values())>16:raise ValueError('Global call budget differs')
    if path.exists():
        old=read_json(path)
        for rel,h in old['artifact_hashes'].items():
            if sha256(root/rel)!=h:raise ValueError('Completed artifact changed')
        return old
    if not publish:raise ValueError('Completion has not been published')
    if read_json(root/'summary.json')['status']!='completed':raise ValueError('Summary incomplete')
    files=[root/'offline_protocol.json',root/'offline_summary.json',root/'input_acceptance.json',root/'summary.json',root/'report.md']
    for profile in study.PROFILES:
        run=root/'qa'/profile
        files.extend([run/'protocol.json',run/'prepare_binding.json',run/'completion_audit.json'])
        for folder in ('inputs','results','calls','raw_generations','synthetic'):
            files.extend((run/folder).rglob('*.json'))
    # 步骤2绑定5400份离线选择/轨迹及实际问答，防止合法缓存错槽或后续改写。
    files.extend(root/rel for rel in read_json(root/'offline_summary.json')['exports'])
    value=dict(status='passed',method=study.METHOD,method_results=5400,new_real_qa=1096,
        synthetic_generations=sum(v['synthetic_calls'] for v in accept.values()),profile_audits=accept,
        artifact_hashes={str(p.relative_to(root)):sha256(p) for p in files},new_blip_calls=0,new_selection_visual_calls=0,
        automatic_default_replacement=False)
    study.freeze(path,value);return value


def main():
    """接口：仅阶段/GPU/恢复，不接受算法参数、样本筛选或额外问答预算。"""
    from videoqa_runtime.common import read_json,sha256
    from videoqa_full.state import durable,exclusive,utc
    from videoqa_generalization import region_scope_full as study
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=('all','offline','prepare','inputs','answer','report','audit'),default='all')
    parser.add_argument('--profile',choices=study.PROFILES);parser.add_argument('--gpus',nargs='+',default=list(map(str,range(8))))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',args.run_id):parser.error('Invalid run ID')
    if args.stage in ('prepare','answer') and not args.profile:parser.error('Profile required')
    root=ROOT/'outputs/full_videoqa'/args.run_id
    if (root/'offline_protocol.json').exists() and not args.resume:raise ValueError('Existing run requires --resume')
    from videoqa_generalization import region_scope_full_runtime as runtime
    from videoqa_generalization import region_scope_full_report as report
    if args.stage=='all':
        with exclusive(root/'pipeline.lock'):
            if (root/'completion_audit.json').exists():complete(root);print('already_completed',flush=True);return
            durable(root/'launcher.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            for name,command in commands(args.run_id,args.gpus):
                durable(root/'pipeline_status.json',dict(status='running',stage=name,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                try:subprocess.run(command,cwd=ROOT,check=True)
                except BaseException as exc:
                    durable(root/'pipeline_status.json',dict(status='paused',stage=name,error=repr(exc),utc=utc()));raise
            durable(root/'pipeline_status.json',dict(status='completed',utc=utc()))
    elif args.stage=='offline':study.offline(root)
    elif args.stage=='prepare':study.prepare(root,args.profile,args.gpus)
    elif args.stage=='inputs':
        specs={p:study.verify(root/'qa'/p) for p in study.PROFILES}
        if sum(v['max_new_calls'] for v in specs.values())!=1096:raise ValueError('Expected total calls differ')
        study.freeze(root/'input_acceptance.json',dict(status='passed',new_real_qa=1096,total_results=5400,
            protocol_hashes={p:sha256(root/'qa'/p/'protocol.json') for p in study.PROFILES}))
    elif args.stage=='answer':
        accepted=read_json(root/'input_acceptance.json')
        for p,h in accepted['protocol_hashes'].items():
            if sha256(root/'qa'/p/'protocol.json')!=h:raise ValueError('Global frozen input changed')
        runtime.execute(root/'qa'/args.profile)
    elif args.stage=='report':report.summarize(root);report.render_panel(root)
    else:complete(root,publish=True)


if __name__=='__main__':main()
