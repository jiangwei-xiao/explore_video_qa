"""exp32开发200正式入口：等待独立选择准备、输入冻结、一次生成与安全恢复。"""
import argparse
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json
from videoqa_full.state import durable, utc
from videoqa_audit.literal_development_runtime import execute


def main():
    """接口：后台只观察现有准备进程，不因等待超时重启OCR；失败则暂停正式生成。"""
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True); args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum():
        parser.error('Invalid run ID')
    root = ROOT / 'outputs/methods' / args.run_id
    try:
        # 步骤1：固定选择完毕才运行；等待记录不是对准备进程存活的伪验证。
        while not (root / 'summary.json').exists():
            if (root / 'failure.json').exists():
                raise ValueError('Selection preparation paused: ' + str(read_json(root / 'failure.json')))
            durable(root / 'qa_wait_status.json', dict(status='waiting_prepared_inputs', pid=os.getpid(), sid=os.getsid(0), utc=utc()))
            time.sleep(15)
        # 步骤2：正式入口自行冻结导入划分和调用预算，不需要会话继续发指令。
        execute(root)
    except BaseException as exc:
        durable(root / 'qa_failure.json', dict(status='paused', error=repr(exc), utc=utc())); raise


if __name__ == '__main__':
    main()
