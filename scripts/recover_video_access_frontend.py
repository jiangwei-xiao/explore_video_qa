#!/usr/bin/env python3
"""接口：仅在零QA且全部BLIP批次已确认完成时，核验复用前端缓存并修复执行环境。"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')


def main():
    """接口：缺依赖不是重新评分理由；缓存须绑定原模型、变换代码和完全相同的关键依赖。"""
    import numpy as np
    from videoqa_runtime.common import read_json,sha256
    from videoqa_full.state import durable,utc
    from videoqa_access.runtime import PY,QPY
    from videoqa_full.prepare import environment
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--target',required=True);a=p.parse_args()
    source=Path(a.source).resolve();target=Path(a.target).resolve()
    if source==target or source.parent!=ROOT/'outputs/analysis' or source.parent!=target.parent:raise ValueError('Unsafe paths')
    if target.exists():raise ValueError('Fresh recovery directory required')
    if list(source.glob('qa/**/calls/*/*.json')) or list((source/'prepared').glob('*/*.json')):raise ValueError('Expected zero-QA before first completed selection')
    if not (source/'failure.json').exists():raise ValueError('Source failure not recorded')
    old=read_json(source/'protocol.json');accept_path=ROOT/'outputs/analysis/video_access_frontend_environment_acceptance_20261105/acceptance.json';accept=read_json(accept_path)
    if accept['status']!='passed' or max(accept[k] for k in ('itm_score_max_error','itm_visual_max_error','pure_visual_max_error'))!=0:
        raise ValueError('Synthetic exact numerical compatibility not established')
    for python in (PY,QPY):
        if environment(python)!=old['environments'][python]:raise ValueError('Existing environment changed')
    for name in ('src/videoqa_access/trial.py','src/videoqa_access/transport.py','src/videoqa_methods/model_ops.py','src/videoqa_methods/density_pipeline.py'):
        if sha256(ROOT/name)!=old['code'][name]:raise ValueError('BLIP input or model path code changed')
    for model,stat in old['models'].items():
        if sha256(model)!=stat['sha256']:raise ValueError('Model bytes changed')
    cps=list(source.glob('scorer_cache/**/blip_calls/*.json'));cached={}
    # 步骤1逐批核验完成状态、原始数组及归一化，不把未知前向当成可恢复缓存。
    for cp in cps:
        rec=read_json(cp);arr=cp.with_suffix('.npz')
        if rec['status']!='completed' or sha256(arr)!=rec['array_sha256']:raise ValueError('Uncertain or changed BLIP batch')
        with np.load(arr,allow_pickle=False) as data:
            f=data['features']
            if f.dtype!=np.float32 or f.shape!=(rec['frames'],768) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5,rtol=0):raise ValueError('Invalid frozen features')
            if rec['kind']=='itm' and (len(data['scores'])!=rec['frames'] or not np.isfinite(data['scores']).all() or np.any(data['scores']<0) or np.any(data['scores']>1)):raise ValueError('Invalid frozen ITM')
        cached[str(cp.relative_to(source))]=sha256(cp);cached[str(arr.relative_to(source))]=sha256(arr)
    if len(cps)!=125:raise ValueError('Frozen complete batch count differs; inspect before importing')
    target.mkdir(parents=True);exports={}
    # 步骤2保留同一GPU/逻辑槽位，完整记录所有继承资产的来源与费用；不复制失败标记或旧代码协议。
    for directory in ('indices','previews','transport_payloads','scorer_cache'):
        for src in (source/directory).rglob('*'):
            if not src.is_file():continue
            rel=src.relative_to(source);dst=target/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            h=sha256(src)
            if sha256(dst)!=h:raise ValueError('Copied frontend asset differs')
            exports[str(rel)]=h
    for name in ('manifest.json','labels.json','engineering_acceptance.json','link_calibration.json','pre_model_runtime_acceptance.json','label_isolation_and_auditor_smoke.json','parser_label_isolation.json','preview_asset_summary.json'):
        shutil.copy2(source/name,target/name);exports[name]=sha256(source/name)
    assignment=source/'assignments/videomme_select.json';dst=target/'assignments/videomme_select.json';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(assignment,dst)
    durable(target/'frontend_recovery.json',dict(source_run=str(source),source_protocol_sha256=sha256(source/'protocol.json'),
        source_failure=read_json(source/'failure.json'),environment_acceptance_file=str(accept_path),environment_acceptance_sha256=sha256(accept_path),
        imported_complete_blip_batches=len(cps),imported_cache_sha256=cached,imported_assets_sha256=exports,source_real_qa=0,
        selection_environment=QPY,key_dependency_or_weight_changes=False,physical_selection_gpus=read_json(assignment)['gpus'],utc=utc()))
    # 步骤3启动一次新指纹协调器及独立审计；旧结果和旧失败原样保存。
    log=(target/'coordinator.log').open('ab',buffering=0)
    child=subprocess.Popen(['nohup','setsid',QPY,'-u',str(ROOT/'scripts/run_video_access_pilot.py'),'coordinate','--root',str(target),'--gpus','4,5,6,7'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);time.sleep(.2)
    durable(target/'launch.json',dict(pid=child.pid,sid=os.getsid(child.pid),recovered_from=str(source),launched_at=utc()))
    log=(target/'independent_audit.log').open('ab',buffering=0)
    audit=subprocess.Popen(['nohup','setsid',QPY,'-u',str(ROOT/'scripts/audit_video_access_pilot.py'),'--root',str(target)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);time.sleep(.2)
    durable(target/'independent_audit_launch.json',dict(pid=audit.pid,sid=os.getsid(audit.pid),target_pid=child.pid))
    print(dict(coordinator_pid=child.pid,audit_pid=audit.pid,imported_blip_batches=len(cps)),flush=True)


if __name__=='__main__':main()
