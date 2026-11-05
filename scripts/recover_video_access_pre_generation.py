#!/usr/bin/env python3
"""接口：只允许零模型调用的预生成阶段改版；核验复用预览后启动新指纹运行。"""
import argparse
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')


def main():
    """接口：等待原CPU任务正常结束，不重启生成、不复用旧分数/特征/答案。"""
    from videoqa_runtime.common import read_json,sha256
    from videoqa_full.state import durable,utc
    from videoqa_access.trial import freeze
    import importlib.util
    spec=importlib.util.spec_from_file_location('access_recovery_pid',ROOT/'scripts/audit_video_access_pilot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--target',required=True);p.add_argument('--gpus',default='4,5,6,7');a=p.parse_args()
    source=Path(a.source).resolve();target=Path(a.target).resolve()
    if source==target or source.parent!=target.parent or source.parent!=ROOT/'outputs/analysis':raise ValueError('Unsafe recovery paths')
    target.mkdir(parents=True,exist_ok=True)
    if (target/'launch.json').exists():raise RuntimeError('Target has already been launched; do not repeat')
    launch=read_json(source/'launch.json');pid=launch['pid'];proc=Path('/proc')/str(pid)/'stat'
    ticks=proc.read_text().split(') ',1)[1].split()[19] if proc.exists() else None
    # 步骤1仅等待原进程；旧协调器将因已声明的源指纹变化在GPU前暂停。
    while ticks is not None and module.live(pid,ticks):
        durable(target/'recovery_status.json',dict(stage='waiting_original_cpu_completion',pid=os.getpid(),target_pid=pid,utc=utc()));time.sleep(10)
    calls=list(source.glob('qa/**/calls/*/*.json'))+list(source.glob('scorer_cache/**/*.json'))
    if calls:raise RuntimeError('Source contains model call ledgers; automatic pre-generation recovery forbidden')
    if not (source/'failure.json').exists():raise ValueError('Expected pre-generation fingerprint stop is not recorded')
    old=read_json(source/'protocol.json')
    required={'src/videoqa_access/trial.py':('build_preview','coarse_frames','prepare_manifest','subset','freeze'),
        'src/videoqa_access/transport.py':('preview_bytes','rgb_hash','pack_frames','CountingReader')}
    for name,functions in required.items():
        previous=source/'code_snapshot'/name
        if sha256(previous)!=old['code'][name]:raise ValueError('Historical preview source snapshot changed')
        def nodes(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        left,right=nodes(previous),nodes(ROOT/name)
        if any(left.get(f)!=right.get(f) for f in functions):raise ValueError('Preview-producing function changed; cannot import')
    manifest=read_json(source/'manifest.json');required={r['video_id'] for r in manifest['rows'] if r['dataset']=='videomme'}
    if not all((source/'indices'/(vid+'.json')).exists() for vid in required):raise ValueError('Original preview phase incomplete')
    # 步骤2只复制不可变原始预览和时钟元数据；所有新评分/选择/问答必须重新生成。
    exports={}
    for name in ('manifest.json','labels.json','engineering_acceptance.json','link_calibration.json','pre_model_runtime_acceptance.json','label_isolation_and_auditor_smoke.json'):
        src=source/name;dst=target/name;shutil.copy2(src,dst);exports[name]=sha256(src)
        if sha256(dst)!=exports[name]:raise ValueError('Imported receipt bytes differ')
    for directory in ('indices','previews','transport_payloads'):
        for src in (source/directory).rglob('*'):
            if not src.is_file():continue
            rel=src.relative_to(source);dst=target/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            h=sha256(src)
            if sha256(dst)!=h:raise ValueError('Imported source asset differs')
            exports[str(rel)]=h
    freeze(target/'pre_generation_recovery.json',dict(source_run=str(source),source_protocol_sha256=sha256(source/'protocol.json'),
        source_failure=read_json(source/'failure.json'),zero_source_model_call_ledgers=True,imported_exports=exports,
        rationale='pre-generation fixes: fail-fast scheduling, environment checks, symmetric on-demand preview/original API; preview construction AST unchanged and algorithms unchanged',utc=utc()))
    # 步骤3新运行使用新代码快照，旧目录和CPU建设成本原样保留；启动一次协调器/审计器。
    log=(target/'coordinator.log').open('ab',buffering=0)
    command=['nohup','setsid','/App/conda/envs/explore_video_qa/bin/python','-u',str(ROOT/'scripts/run_video_access_pilot.py'),'coordinate','--root',str(target),'--gpus',a.gpus]
    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);time.sleep(.2)
    durable(target/'launch.json',dict(pid=child.pid,sid=os.getsid(child.pid),recovered_from=str(source),launched_at=utc()))
    log=(target/'independent_audit.log').open('ab',buffering=0)
    audit=subprocess.Popen(['nohup','setsid','/App/conda/envs/explore_video_qa_lmms/bin/python','-u',str(ROOT/'scripts/audit_video_access_pilot.py'),'--root',str(target)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);time.sleep(.2)
    durable(target/'independent_audit_launch.json',dict(pid=audit.pid,sid=os.getsid(audit.pid),target_pid=child.pid))
    durable(target/'recovery_status.json',dict(stage='launched_once',pid=child.pid,audit_pid=audit.pid,utc=utc()))
    print(dict(coordinator_pid=child.pid,audit_pid=audit.pid),flush=True)


if __name__=='__main__':main()
