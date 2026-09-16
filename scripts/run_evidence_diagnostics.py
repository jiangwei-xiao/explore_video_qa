"""最多20次人工证据修复诊断；每题原输入与修复输入同卡同进程，失败不自动重试。"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT,read_json,write_json,sha256,offline_environment
from prepare_evidence_diagnostics import validate_case


def check_manifest(manifest):
    """运行前验证预算、类别配额、历史身份和全部精确输入，不进入模型。"""
    cases=manifest['cases'];ids=[c['question_id'] for c in cases]
    if manifest.get('status')!='frozen' or manifest.get('visual_review_confirmed') is not True:raise ValueError('清单尚未完成输入验收')
    if len(ids)!=len(set(ids)) or len(cases)>10 or 2*len(cases)>20:raise ValueError('超出问答预算或重复题')
    if manifest['maximum_started_calls']!=20:raise ValueError('不得扩大20次上限')
    if manifest.get('planned_calls')!=2*len(cases) or any(c['stratum'] not in ('all_wrong','discordant','all_correct') for c in cases):raise ValueError('计划调用数或类别非法')
    for category,limit in [('all_wrong',6),('discordant',2),('all_correct',2)]:
        if sum(c['stratum']==category for c in cases)>limit:raise ValueError('类别配额超限')
    for case in cases:
        original=read_json(case['source_result']);pool=original.get('pool',original.get('selection'))
        if sha256(case['source_result'])!=case['source_result_sha256']:raise ValueError('历史结果已变')
        if case['original_frames']!=pool['selected_frames']:raise ValueError('原输入发生变化')
        validate_case(case,pool['candidates'])
        if case['question']!=original['question'] or case['options']!=original['options']:raise ValueError('题目或选项变更')


def call_path(run,qid,variant):
    """固定调用定位，不允许通过改重试编号绕过已开始记录。"""
    return Path(run)/'calls'/f'{qid}.{variant}.json'


def check_recovery(run,manifest_sha):
    """已完成记录可跳过；任何未完成/失败调用均要求人工核查，防止重复生成。"""
    records=[read_json(p) for p in (Path(run)/'calls').glob('*.json')]
    if len(records)>20:raise ValueError('已超过预算')
    for r in records:
        if r['manifest_sha256']!=manifest_sha or r['status']!='completed':
            raise RuntimeError('有失败或不确定调用，禁止自动恢复生成')
    return records


def worker(gpu,cases,run,manifest_sha,stop):
    """一个工作进程使用一张空闲GPU；只加载LLaVA，不加载BLIP或执行范围分类。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    offline_environment();run=Path(run)
    with (run/'logs'/f'gpu_{gpu}.log').open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import torch
            from videoqa_runtime.llava_backend import LlavaBackend
            from videoqa_runtime.video import decode_selected
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            # 步骤1：常驻模型及一次合成前向预热，单列成本，不生成开发题答案。
            start=time.perf_counter();backend=LlavaBackend();loaded=time.perf_counter()-start
            start=time.perf_counter();backend.warmup()
            write_json(run/'workers'/f'{gpu}.json',dict(gpu=gpu,pid=os.getpid(),load_seconds=loaded,warmup_seconds=time.perf_counter()-start,load_report=backend.load_report))
            for case in cases:
                qid=case['question_id'];order=['original','repaired'] if case['order_index']%2==0 else ['repaired','original']
                if (run/'pairs'/f'{qid}.json').exists():continue
                # 步骤2：同题两种输入在同卡执行，奇偶交替顺序，原视频只按冻结PTS读帧。
                if sha256(case['video']['path'])!=case['video_sha256']:raise ValueError('视频身份变化')
                for variant in order:
                    if stop.is_set():return
                    destination=call_path(run,qid,variant)
                    if destination.exists():continue
                    with (run/'budget.lock').open('a') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX)
                        if len(list((run/'calls').glob('*.json')))>=20:raise RuntimeError('20次硬上限已达')
                        # 步骤3：调用前原子持久化意图；异常或断线保守占用额度，不自动重试。
                        record=dict(question_id=qid,variant=variant,manifest_sha256=manifest_sha,status='reserved_may_generate',gpu=gpu,started_utc=time.time())
                        if destination.exists():raise RuntimeError('调用已占用，禁止重复')
                        write_json(destination,record)
                    state={};begin=time.perf_counter()
                    try:
                        frames=case['original_frames'] if variant=='original' else case['repaired_frames']
                        images=decode_selected(case['video'],frames)
                        answer=backend.answer(images,[f['timestamp_seconds'] for f in frames],case['video']['duration_seconds'],case['question'],case['options'],generation_state=state)
                        if answer['visual_tokens']!=3360 or answer['pixel_shape']!=[16,3,384,384]:raise ValueError('视觉输入不符合协议')
                        if variant=='original' and answer['prompt']!=case['historical_prompt']:raise ValueError('原输入提示词与历史不一致')
                        record.update(status='completed',answer=answer,selected_frames=frames,generation_state=state,
                                      reference=case['reference'],correct=answer['parsed_answer']==case['reference'],seconds=time.perf_counter()-begin)
                        write_json(destination,record)
                        print(qid,variant,answer['raw_output'],flush=True)
                        # 步骤4：复测不一致只暂停该题归因；原输入先执行时不继续新的修复调用。
                        if variant=='original' and answer['parsed_answer']!=case['historical_prediction']:
                            write_json(run/'pairs'/f'{qid}.json',dict(status='unstable_original_attribution_paused',question_id=qid))
                            break
                    except BaseException as error:
                        record.update(status='failed_or_uncertain',generation_state=state,error=f'{type(error).__name__}: {error}',seconds=time.perf_counter()-begin)
                        write_json(destination,record);raise
                else:
                    write_json(run/'pairs'/f'{qid}.json',dict(status='completed',question_id=qid))
        except BaseException as error:
            stop.set();write_json(run/'workers'/f'{gpu}.error.json',dict(error=f'{type(error).__name__}: {error}'));raise


def run_diagnostics(run,gpus):
    """协调器持有排他锁，校验GPU和冻结清单，至多启动两个长期进程。"""
    run=Path(run).resolve();manifest=read_json(run/'manifest.json');check_manifest(manifest);fingerprint=sha256(run/'manifest.json')
    freeze=read_json(run/'freeze.json')
    if freeze['manifest_sha256']!=fingerprint:raise ValueError('冻结清单被更改')
    if not 1<=len(gpus)<=2 or len(set(gpus))!=len(gpus):raise ValueError('使用一至两张不同GPU')
    with (run/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_recovery(run,fingerprint)
        # 完成态只读返回，不重写原批次墙钟，也不因当前GPU忙而重跑已完成调用。
        if all((run/'pairs'/f'{c["question_id"]}.json').exists() for c in manifest['cases']):
            print('All diagnostic pairs already closed; no calls or timing records changed.',flush=True)
            return
        for name,expected in freeze.get('code_sha256',{}).items():
            if sha256(ROOT/name)!=expected:raise ValueError('诊断执行代码已改变')
        # 步骤1：检查空闲卡，不抢占现有进程；没有空闲卡时不加载模型。
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
        states={p[0].strip():(int(p[1]),int(p[2])) for p in (line.split(',') for line in lines)}
        if any(g not in states or states[g][0]>512 or states[g][1]>5 for g in gpus):raise RuntimeError('指定GPU不空闲')
        for name in ('calls','logs','workers','pairs'): (run/name).mkdir(exist_ok=True)
        write_json(run/'pid.json',dict(pid=os.getpid(),sid=os.getsid(0),gpus=gpus,manifest_sha256=fingerprint))
        pending=[c for c in manifest['cases'] if not (run/'pairs'/f'{c["question_id"]}.json').exists()]
        ctx=mp.get_context('spawn');stop=ctx.Event();processes=[];start=time.perf_counter()
        # 步骤2：任务以题为单位分配，两种输入不会拆到不同GPU。
        for i,g in enumerate(gpus):
            jobs=pending[i::len(gpus)]
            if jobs:
                process=ctx.Process(target=worker,args=(g,jobs,str(run),fingerprint,stop));process.start();processes.append(process)
        while any(p.is_alive() for p in processes):
            for p in processes:p.join(timeout=1)
            calls=[read_json(p) for p in (run/'calls').glob('*.json')]
            write_json(run/'status.json',dict(status='running',completed=sum(c['status']=='completed' for c in calls),budget_used=len(calls),maximum=20))
        # 步骤3：总结只基于实际调用，任何不完整对照都不能冒充已完成。
        rows=[]
        for case in manifest['cases']:
            q=case['question_id'];a=call_path(run,q,'original');b=call_path(run,q,'repaired')
            if not a.exists() or not b.exists():rows.append(dict(question_id=q,status='incomplete'));continue
            old,new=read_json(a),read_json(b)
            if old['status']!='completed' or new['status']!='completed':rows.append(dict(question_id=q,status='failed'));continue
            stable=old['answer']['parsed_answer']==case['historical_prediction']
            rows.append(dict(question_id=q,method=case['method'],category=case['stratum'],status='paired' if stable else 'unstable',
                original=old['answer']['parsed_answer'],repaired=new['answer']['parsed_answer'],old_correct=old['correct'],new_correct=new['correct'],
                improvement=int(new['correct'])-int(old['correct']),changed=old['answer']['parsed_answer']!=new['answer']['parsed_answer']))
        used=len(list((run/'calls').glob('*.json')))
        status='paused' if any(p.exitcode for p in processes) else 'finished'
        write_json(run/'summary.json',dict(status=status,pairs=rows,budget_used=used,maximum_started_calls=20,wall_seconds=time.perf_counter()-start,
                   interpretation='目的性选题、人工定向修复；不是算法准确率或总体性能上限',blip_calls=0,scope_calls=0))
        write_json(run/'status.json',dict(status=status,budget_used=used,maximum=20))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--gpus',default='0,1')
    args=p.parse_args();run_diagnostics(args.run,args.gpus.split(','))
