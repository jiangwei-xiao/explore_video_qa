"""exp22已完成问答的纯后处理恢复；必须使用正式Qwen环境，不调用生成入口。"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,utc
from videoqa_budget32 import development
import report_external_support as report

NAME='rd12_exp22__qwen32__dev200__20261004_r1'
METHOD='RD-1.2-exp22'
QA=ROOT/'outputs/diagnostics'/NAME
COORD=ROOT/'outputs/methods'/NAME


def main():
    """接口：先确认200题生成完整，再仅计分/比较；未知生成不在此处处理。"""
    # 步骤1：完成审计和指纹必须通过，记录600份文件基线。
    if read_json(QA/'completion_audit.json')['status']!='passed':raise ValueError('QA尚未验收')
    spec=development.verify(QA)
    if spec['methods']!=[METHOD] or spec['max_calls']!=200:raise ValueError('运行身份不符')
    files=[p for d in ['calls','raw_generations','results'] for p in (QA/d/METHOD).glob('*.json')]
    if len(files)!=600:raise ValueError('调用、原始返回和答案数量不符')
    before={str(p):sha256(p) for p in files}
    # 步骤2：原汇总器以已完成QA为前置；不调用协调器或任何生成函数。
    durable(COORD/'status.json',dict(status='completed',qa_complete=True,report_status='recovering',utc=utc()))
    report.NAME=NAME;report.METHOD=METHOD;report.RUN=QA;report.COORD=COORD
    try:
        report.main(False)
    except BaseException as exc:
        durable(COORD/'status.json',dict(status='paused',qa_complete=True,error=repr(exc),utc=utc()))
        raise
    # 步骤3：后处理不得更改已有生成；显式记录0次新模型调用。
    after={str(p):sha256(p) for p in files}
    if before!=after:raise ValueError('后处理改写了生成相关文件')
    durable(QA/'report_recovery.json',dict(status='passed',code_sha256=sha256(__file__),
        unchanged_records=600,new_model_calls=0,record_hashes=before,utc=utc()))
    durable(COORD/'status.json',dict(status='completed',qa_complete=True,report_status='passed',utc=utc()))


if __name__=='__main__':main()
