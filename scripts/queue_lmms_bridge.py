#!/usr/bin/env python3
"""等待离线索引与空闲GPU，再顺序执行合成/20次桥接；不包含全量实验。"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import durable,utc,exclusive
from videoqa_runtime.baseline_run import available_gpus


def main():
    """接口：只等待已验证的离线进程；缺失/失败停下，绝不自动重启或抢卡。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    import re
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id)
    run=ROOT/'outputs/analysis'/a.run_id
    with exclusive(run/'queue.lock'):
        snapshot={str(f.relative_to(ROOT)):sha256(f) for f in (ROOT/'src/videoqa_lmms_bridge').glob('*.py')}
        for name in ('scripts/run_lmms_bridge.py','scripts/check_lmms_qwen_synthetic.py','scripts/queue_lmms_bridge.py'):
            snapshot[name]=sha256(ROOT/name)
        def verify():
            for name,h in snapshot.items():assert sha256(ROOT/name)==h,'Code changed while queued: '+name
        def status(value,**extra):durable(run/'queue_status.json',dict(status=value,pid=os.getpid(),session_id=os.getsid(0),utc=utc(),**extra))
        try:
            # 步骤1：只观察当前离线任务，不凭状态文件假定它还活着。
            while True:
                verify();s=read_json(run/'offline_status.json')
                if s['status']=='completed':break
                assert s['status']=='running',s
                os.kill(s['pid'],0)
                cmd=Path(f"/proc/{s['pid']}/cmdline").read_bytes()
                assert b'analyze_lmms_uniform.py' in cmd and a.run_id.encode() in cmd
                status('waiting_offline',offline_pid=s['pid'],offline_completed=s['completed'])
                time.sleep(30)
            stages=[('llava_synthetic_acceptance.json','/App/conda/envs/explore_video_qa_lmms/bin/python',
                ['scripts/run_lmms_bridge.py','--run-id',a.run_id,'--phase','synthetic']),
                ('qwen_framework_synthetic.json','/App/conda/envs/explore_video_qa_lmms_qwen/bin/python',
                 ['scripts/check_lmms_qwen_synthetic.py','--output',str(run/'qwen_framework_synthetic.json')])]
            for filename,python,command in stages:
                if (run/filename).exists():
                    assert read_json(run/filename)['status']=='passed','Failed synthetic acceptance needs review'
                    continue
                # 步骤2：资源等待不占生成额度；只有真正空闲才启动一次子进程。
                while True:
                    verify()
                    try:gpus,_=available_gpus([str(i) for i in range(8)]);break
                    except RuntimeError:status('waiting_gpu',stage=filename);time.sleep(30)
                status('running_synthetic',stage=filename,gpu=gpus[0]);subprocess.run([python,*command,'--gpu',gpus[0]],cwd=ROOT,check=True)
            verify();status('preparing_bridge')
            subprocess.run(['/App/conda/envs/explore_video_qa_lmms/bin/python','scripts/run_lmms_bridge.py',
                '--run-id',a.run_id,'--phase','prepare'],cwd=ROOT,check=True)
            while True:
                verify()
                try:gpus,_=available_gpus([str(i) for i in range(8)]);break
                except RuntimeError:status('waiting_gpu',stage='bridge');time.sleep(30)
            # 步骤3：合成通过后仅运行批准的20次桥接，失败退出，不自动追加或重跑。
            status('running_bridge',gpu=gpus[0])
            subprocess.run(['/App/conda/envs/explore_video_qa_lmms/bin/python','scripts/run_lmms_bridge.py',
                '--run-id',a.run_id,'--phase','bridge','--gpu',gpus[0]],cwd=ROOT,check=True)
            status('completed')
        except BaseException as error:status('paused',error=repr(error));raise


if __name__=='__main__':main()
