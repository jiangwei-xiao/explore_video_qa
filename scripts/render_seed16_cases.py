"""M16/K32固定得失面板，按配置命名避免同一方法ID覆盖图板。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json
from videoqa_full.state import durable
from render_rd_failure_panel import render
from report_seed16_configuration import RUN,PARENT,METHOD


def main():
    """接口：只用独立报告冻结的六题清单，精确核验RGB，不新增问答。"""
    assert read_json(RUN/'configuration_report_status.json')['status']=='passed'
    report=read_json(RUN/'configuration_report.json');directory=RUN/'case_review';jobs=[]
    # 步骤1：同方法不同配置使用独立标签；不将M16旧诊断混为本轮输入。
    for item in report['case_manifest']:
        q=item['question_id'];jobs.extend([(directory,q,'exp12_M16_K32',RUN/'results'/METHOD/(q+'.json')),
            (directory,q,'exp12_M32_K32',PARENT/'imported/results'/METHOD/(q+'.json'))])
    # 步骤2：图板生成与语义审核分开，成功解码不代表证据充分。
    with ThreadPoolExecutor(max_workers=4) as executor:records=list(executor.map(render,jobs))
    durable(directory/'manifest.json',dict(cases=report['case_manifest'],records=records,new_model_calls=0,semantic_review='pending'))
    print(dict(questions=len(report['case_manifest']),boards=len(records)),flush=True)


if __name__=='__main__':main()
