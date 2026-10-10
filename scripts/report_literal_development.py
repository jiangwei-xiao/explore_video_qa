"""exp32后台收尾：只等待原QA协调器、完整审计和统计，不重新启动模型任务。"""
import argparse
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json
from videoqa_full.state import durable, utc
from videoqa_audit.literal_development_report import finish


def main():
    """接口：绑定存活QA PID，完成后只做恢复/计分；异常落盘，不触发新生成。"""
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True); parser.add_argument('--qa-pid', type=int, required=True)
    args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum() or args.qa_pid <= 1:
        parser.error('Invalid run or original QA process')
    root = ROOT / 'outputs/methods' / args.run_id; run = root / 'qa/qwen32'
    try:
        # 步骤1：原协调器未完成时按具体PID核验存活，不把状态文件当作存活证明。
        while not (run / 'summary.json').exists():
            if (root / 'qa_failure.json').exists() or (root / 'failure.json').exists():
                raise ValueError('Original preparation or QA paused; do not restart')
            os.kill(args.qa_pid, 0)
            durable(root / 'analysis_wait_status.json', dict(status='waiting_original_QA', original_qa_pid=args.qa_pid,
                pid=os.getpid(), sid=os.getsid(0), utc=utc()))
            time.sleep(15)
        # 步骤2：完成态实际恢复与统计，不加载新模型、不突破正式调用额度。
        finish(root)
    except BaseException as exc:
        durable(root / 'analysis_failure.json', dict(status='paused', error=repr(exc), utc=utc())); raise


if __name__ == '__main__':
    main()
