"""独立后台收尾协调：只读全量审计/复核页，不参与模型调度或重复生成。"""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse
from pathlib import Path
import subprocess
import sys
import time
import re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc

PYTHON='/App/conda/envs/explore_video_qa_retrieval/bin/python'


def main():
    """接口：只观察已确认的主PID，检查点到齐后执行只读核验；终止不触发重启。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id):raise ValueError('Invalid run ID')
    root=ROOT/'outputs/full_videoqa'/a.run_id;finished=set()
    launcher=root/'operational_recovery/launcher.json'
    pid=read_json(launcher if launcher.exists() else root/'launcher.json')['pid']
    commands=[('coarse',root/'stage_completion/coarse.json',[PYTHON,str(ROOT/'scripts/audit_scorer_comparison.py'),'--run-id',a.run_id,'--stage','coarse']),
        ('qwen32',root/'stage_completion/refine.json',[PYTHON,str(ROOT/'scripts/audit_scorer_comparison.py'),'--run-id',a.run_id,'--stage','qwen32']),
        ('llava8',root/'stage_completion/refine.json',[PYTHON,str(ROOT/'scripts/audit_scorer_comparison.py'),'--run-id',a.run_id,'--stage','llava8']),
        ('review',root/'comparison.json',[PYTHON,str(ROOT/'scripts/render_scorer_comparison_cases.py'),'--run-id',a.run_id])]
    with exclusive(root/'independent_audit/watcher.lock'):
        try:
            # 步骤1：完成标记与主进程都作为证据；主任务暂停/退出时不擅自恢复生成。
            while len(finished)<len(commands):
                status=read_json(root/'pipeline_status.json')
                if status['status']=='paused':raise RuntimeError('Main pipeline paused; preserve state')
                for label,gate,command in commands:
                    if label in finished or not gate.exists():continue
                    if label=='review' and not all(x in finished for x in ('coarse','qwen32','llava8')):continue
                    durable(root/'independent_audit/watcher_status.json',dict(status='auditing',stage=label,pid=os.getpid(),sid=os.getsid(0),main_pid=pid,utc=utc()))
                    subprocess.run(command,cwd=ROOT,check=True);finished.add(label)
                if len(finished)==len(commands):break
                if status['status']!='completed':os.kill(pid,0)
                durable(root/'independent_audit/watcher_status.json',dict(status='waiting',finished=sorted(finished),pid=os.getpid(),sid=os.getsid(0),main_pid=pid,utc=utc()))
                time.sleep(30)
            # 步骤2：原始源及官方权重收尾再次验字节，不能只依赖准备阶段的意图或旧绿灯。
            for path,h in read_json(root/'preflight.json')['sources'].items():
                if sha256(path)!=h:raise ValueError('Final source fingerprint changed: '+path)
            durable(root/'independent_audit/watcher_status.json',dict(status='completed',finished=sorted(finished),
                main_pid=pid,new_model_calls=0,actual_frame_semantic_review='pending',utc=utc()))
        except BaseException as exc:
            durable(root/'independent_audit/watcher_status.json',dict(status='paused',finished=sorted(finished),error=repr(exc),utc=utc()));raise


if __name__=='__main__':main()
