"""RD成果证据的只读汇总入口；不提供模型、解码、选择或生成开关。"""
import argparse
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    """接口：运行编号→独立统计目录；恢复只核对已有来源与导出，不修改原实验。"""
    # 步骤1约束目录名；步骤2调用只依赖冻结结果与PDF的汇总器。
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum():
        parser.error('Invalid evidence run ID')
    from videoqa_audit.rd_evidence import run
    result = run(ROOT / 'outputs/analysis' / args.run_id)
    print(dict(status=result['status'], result_rows=len(result['rows']),
               references=len(result['references']), paired_rows=len(result['paired']),
               recommendation=result['recommendation']['method'], decision=result['stage_decision'],
               new_model_calls=result['new_model_calls']))


if __name__ == '__main__':
    main()
