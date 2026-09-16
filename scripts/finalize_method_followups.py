"""后台分析收尾：等待正式阶段完成，生成统计与变化图板；绝不启动模型调用。"""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT, read_json, write_json


def finish(run):
    """接口：独立排他分析锁，观察完成文件；运行暂停或进程退出时停止，不自动重启实验。"""
    run=Path(run)
    with (run/'analysis.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # 步骤1：只观察持久化状态，连接关闭不影响本后台分析进程。
        while True:
            state=read_json(run/'status.json')
            completed=[p for p in run.glob('execution_*.json') if read_json(p)['status']=='model_phases_completed']
            if state['status']=='model_phases_completed' and completed: break
            if state['status']=='paused': raise RuntimeError('Experiment paused; do not finalize incomplete results')
            try: os.kill(state['pid'],0)
            except ProcessLookupError: raise RuntimeError('Coordinator exited before completion; inspect call journals')
            time.sleep(10)
        # 步骤2：统计和实际帧可视化分开调用，不加载模型权重或执行问答。
        write_json(run/'analysis_status.json',dict(status='summarizing',pid=os.getpid()))
        subprocess.run([sys.executable,str(ROOT/'scripts/summarize_method_followups.py'),str(run)],check=True,cwd=ROOT)
        write_json(run/'analysis_status.json',dict(status='rendering',pid=os.getpid()))
        subprocess.run([sys.executable,str(ROOT/'scripts/render_followup_cases.py'),str(run)],check=True,cwd=ROOT)
        write_json(run/'analysis_status.json',dict(status='ready_for_review',pid=os.getpid()))


if __name__=='__main__': finish(sys.argv[1])
