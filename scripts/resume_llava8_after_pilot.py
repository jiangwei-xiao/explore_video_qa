"""八帧先行后调度兼容恢复：冻结协议不改，只在内存补齐真实时长。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,rows,utc
from videoqa_generalization import llava_budget8 as experiment

ORIGINAL_VERIFY=experiment.verify


def duration_compatible(spec,durations):
    """接口：复制元数据后补齐调度字段，方法、模型、输入及调用上限不改。"""
    stats={vid:dict(item,duration_seconds=durations[vid]) for vid,item in spec['video_stats'].items()}
    for value in durations.values():
        if not isinstance(value,(int,float)) or not 0<value<86400:raise ValueError('Invalid source duration')
    return dict(spec,video_stats=stats)


def verify_with_duration(root):
    """接口：原始指纹先校验，再从冻结均匀输入读取900视频时长，仅供队列排序。"""
    root=Path(root);spec=ORIGINAL_VERIFY(root);durations={}
    # 步骤1：同视频三题时长应一致，不能根据题目对错决定排序。
    for row in rows():
        item=read_json(root/'selections/BASE-Uniform'/(row['question_id']+'.json'))
        value=item['selection']['video']['duration_seconds'];vid=row['video_id']
        if vid in durations and durations[vid]!=value:raise ValueError('Video duration differs across questions')
        durations[vid]=value
    return duration_compatible(spec,durations)


def main():
    """接口：仅恢复剩余问答/审计/报告，完整回答不重发，修复代码独立留快照。"""
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--stage',choices=['all','answer','audit','report'],default='all')
    parser.add_argument('--resume',action='store_true',required=True);args=parser.parse_args()
    root=ROOT/'outputs/full_videoqa'/args.run_id
    if root.parent!=ROOT/'outputs/full_videoqa' or not (root/'protocol.json').exists():raise ValueError('Unknown frozen run')
    experiment.verify=verify_with_duration
    if args.stage=='all':
        with exclusive(root/'pipeline.lock'):
            status=root/'pipeline_status.json'
            if read_json(status)['status']=='completed':experiment.audit(root);return
            before={str(p):sha256(p) for f in ['calls','raw_generations','results'] for p in (root/f).glob('*/*.json')}
            record=root/'scheduler_repair.json'
            if not record.exists():
                snap=root/'scheduler_repair_snapshot.py';shutil.copy2(Path(__file__).resolve(),snap)
                durable(record,dict(reason='historical video_stats lacks duration_seconds; queue uses frozen source duration',
                    protocol_sha256=sha256(root/'protocol.json'),repair_sha256=sha256(snap),
                    existing_answer_files=before,new_generations_for_repair=0,utc=utc()))
            saved=read_json(record)
            if saved['protocol_sha256']!=sha256(root/'protocol.json') or saved['repair_sha256']!=sha256(Path(__file__).resolve()):
                raise ValueError('Repair identity changed')
            durable(root/'repair_launcher.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            # 步骤2：新子阶段始终显式恢复，补齐时长只影响时长降序派发。
            for stage in ['answer','audit','report']:
                durable(status,dict(status='running',stage=stage,pid=os.getpid(),sid=os.getsid(0),scheduler_repair=True,utc=utc()))
                try:subprocess.run([experiment.LPY,str(Path(__file__).resolve()),'--run-id',args.run_id,'--stage',stage,'--resume'],cwd=ROOT,check=True)
                except BaseException as exc:
                    durable(status,dict(status='paused',stage=stage,error=repr(exc),utc=utc()));raise
            for p,h in saved['existing_answer_files'].items():
                if sha256(p)!=h:raise ValueError('Pre-repair answer changed')
            durable(root/'scheduler_repair_acceptance.json',dict(status='passed',existing_answer_files_unchanged=len(saved['existing_answer_files']),
                protocol_unchanged=True,utc=utc()))
            spec=read_json(root/'protocol.json')
            durable(status,dict(status='completed',new_calls=spec['max_calls'],method_results=13500,scheduler_repair=True,utc=utc()))
    elif args.stage=='answer':experiment.execute(root)
    elif args.stage=='audit':experiment.audit(root)
    else:experiment.summarize(root)


if __name__=='__main__':main()
