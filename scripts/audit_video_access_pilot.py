#!/usr/bin/env python3
"""接口：独立进程等待固定协调PID结束，再复算原始回答、批次输入和零新增恢复。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')


def live(pid,start_ticks):
    """接口：用/proc进程启动tick识别同一任务，避免把PID重用当成继续运行。"""
    path=Path('/proc')/str(pid)/'stat'
    if not path.exists():return False
    try:fields=path.read_text().split(') ',1)[1].split()
    except FileNotFoundError:return False
    return fields[0]!='Z' and fields[19]==start_ticks


def main():
    """接口：独立复算不读取模型GPU，不重复问答；预算和源身份错误必须失败。"""
    from videoqa_runtime.common import read_json,sha256
    from videoqa_full.state import durable,utc
    from videoqa_access.runtime import verify,PY,QPY,data_rows
    from videoqa_access.report import completion_audit
    from videoqa_access.trial import METHODS
    from videoqa_full.prepare import environment
    from videoqa_lmms_bridge.common import framework_identity
    from videoqa_audit.frame_group_diagnostic import restore_answer,canonical_option
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root).resolve()
    launch=read_json(root/'launch.json');pid=launch['pid'];proc=Path('/proc')/str(pid)/'stat'
    ticks=proc.read_text().split(') ',1)[1].split()[19] if proc.exists() else None
    # 步骤1只等待同一已验证进程；观察失败或连接关闭不触发任何重启。
    while ticks is not None and live(pid,ticks):
        durable(root/'independent_audit_status.json',dict(stage='waiting_coordinator',pid=os.getpid(),target_pid=pid,target_start_ticks=ticks,utc=utc()));time.sleep(10)
    if not (root/'completion.json').exists():raise RuntimeError('Coordinator ended without completion; preserve current state')
    frozen=verify(root);env=read_json(root/'pre_model_runtime_acceptance.json')
    representation=read_json(root/'representation_acceptance.json')
    if representation['status']!='passed' or representation['protocol_sha256']!=sha256(root/'protocol.json'):
        raise ValueError('Two-quality source access acceptance missing')
    if environment(PY)!=env['prepare_environment'] or environment(QPY)!=env['qa_environment'] or framework_identity()!=frozen['framework']:
        raise ValueError('Environment/framework changed during trial')
    audited=completion_audit(root);datasets=audited['datasets'];physical=synthetic=0
    native_scores=[];labels=read_json(root/'labels.json')
    # 步骤2清点所有实际账本而不仅是结果引用；拒绝多余调用和未确认生成。
    for dataset in datasets:
        plan=read_json(root/'qa'/dataset/'plan.json');expected={tuple(x) for x in plan['calls']};actual=set()
        for callpath in (root/'qa'/dataset/'calls').glob('*/*.json'):
            call=read_json(callpath)
            if call['status']!='completed' or call['generation_state']!={'started':True,'returned':True}:raise ValueError('Uncertain generation exists')
            actual.add((call['question_id'],call['method']));physical+=1
        if actual!=expected:raise ValueError('Actual physical calls differ from frozen plan')
        for path in (root/'qa'/dataset/'synthetic/calls').glob('*/*.json'):
            if read_json(path)['status']!='completed':raise ValueError('Uncertain synthetic generation')
            synthetic+=1
        for row in data_rows(root,dataset):
            q=row['question_id'];prepared=read_json(root/'prepared'/dataset/(q+'.json'))
            # 步骤3所有合法细帧必须先出现在该方法原片请求，旧特征不能越权补齐候选。
            rd=prepared['items']['RD-1.2-exp12'];pool=read_json(root/'pools'/dataset/(q+'.json'))
            acquired={tuple(x) for req in rd['requests'] for x in req['requested_sources']}
            if any((f['source_pts'],f['source_frame_index']) not in acquired for f in pool['candidates'][pool['initial_count']:]):
                raise ValueError('Unacquired fine candidate used')
            if rd['extra_visual']['itm_frames']!=0:raise ValueError('RD unexpectedly scored fine candidates')
            if prepared['items']['CTRL-WFS-preview']['requests']:raise ValueError('Preview-only control accessed original')
            for method in METHODS:
                item=prepared['items'][method]
                fetched_originals={tuple(x) for req in item['requests'] if req['representation']=='original' for x in req['requested_sources']}
                selected={(f['source_pts'],f['source_frame_index']) for f in item['selection']['selected_frames']}
                if method!='CTRL-WFS-preview' and fetched_originals!=selected:raise ValueError('Unselected full-resolution source accessed')
                result=read_json(root/'qa'/dataset/'results'/method/(q+'.json'))
                raw=read_json(root/result['raw_generation_file']);label=labels[dataset+'/'+q]
                # 模型输入对象使用占位计分字母；实际标签只在完整返回后交官方计分函数复算。
                actual=restore_answer(dict(dataset=dataset,profile='llava8'),raw,dict(row,answer_index=label))
                native_correct=actual['parsed_answer']=='ABCD'[label]
                normalized=canonical_option(dict(dataset=dataset,profile='llava8'),actual)
                if native_correct!=result['native_correct'] or (normalized=='ABCD'[label])!=result['correct']:
                    raise ValueError('Official post-generation scoring differs')
                native_scores.append(dict(dataset=dataset,question_id=q,method=method,
                    native_correct=native_correct,normalized_correct=result['correct'],
                    official_scored_answer=actual,scoring_after_raw_return=True))
    if physical!=audited['physical_real_calls'] or synthetic!=audited['synthetic_calls'] or physical>780 or synthetic>8:
        raise ValueError('Generation budget does not close')
    # 步骤4实际执行完成后的普通协调CLI；前后所有已完成核心文件的字节/mtime一致。
    paths=set(audited['exports'])|{'completion.json','protocol.json','manifest.json','labels.json'}
    before={rel:(sha256(root/rel),(root/rel).stat().st_mtime_ns) for rel in paths}
    result=subprocess.run([QPY,str(ROOT/'scripts/run_video_access_pilot.py'),'coordinate','--root',str(root)],cwd=ROOT,stdin=subprocess.DEVNULL,capture_output=True,text=True)
    if result.returncode:raise RuntimeError('Completed CLI recovery failed: '+result.stderr[-2000:])
    after={rel:(sha256(root/rel),(root/rel).stat().st_mtime_ns) for rel in paths}
    if before!=after:raise ValueError('Completed recovery modified core output')
    durable(root/'official_postgeneration_scoring.json',dict(records=native_scores,
        note='Actual labels applied after raw return. Internal placeholder metrics are not reported as benchmark correctness.'))
    record=dict(status='passed',datasets=datasets,physical_real_calls=physical,synthetic_calls=synthetic,
        acquired_fine_sources_checked=True,environment_and_framework_unchanged=True,
        completed_cli_recovery=dict(returncode=0,new_generations=0,unchanged_core_files=len(paths)),
        audit_code_sha256=sha256(Path(__file__)),utc=utc())
    durable(root/'independent_completion_audit.json',record);print(record,flush=True)


if __name__=='__main__':main()
