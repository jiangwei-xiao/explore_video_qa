"""补查消融五个固定得失案例，精确RGB图板，不新增模型调用。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json
from videoqa_full.state import durable
from render_rd_failure_panel import render
from report_coarse_only_ablation import RUN,PARENT,METHOD


def main():
    """接口：严格使用已冻结的分时长首个得失清单，不能按画面效果更换案例。"""
    report=read_json(RUN/'ablation_report.json');directory=RUN/'case_review';jobs=[]
    # 步骤1：只接已通过独立核验的200题，不依据部分结果生成有利面板。
    assert read_json(RUN/'independent_report_status.json')['status']=='passed'
    for item in report['case_manifest']:
        q=item['question_id']
        jobs.extend([(directory,q,METHOD,RUN/'results'/METHOD/(q+'.json')),
            (directory,q,'RD-1.2-exp12',PARENT/'imported/results/RD-1.2-exp12'/(q+'.json'))])
    # 步骤2：复用图板函数逐帧验证历史RGB；生成图板不等于完成语义审核。
    with ThreadPoolExecutor(max_workers=4) as executor:records=list(executor.map(render,jobs))
    durable(directory/'manifest.json',dict(cases=report['case_manifest'],records=records,new_model_calls=0,
        semantic_review='pending',scope='Fixed first gain/loss per duration; no long gain case'))
    print(dict(questions=len(report['case_manifest']),boards=len(records),new_model_calls=0),flush=True)


if __name__=='__main__':main()
