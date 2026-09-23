#!/usr/bin/env python3
"""后台完成验收：三组输入/原始生成/计分一致，完成恢复不重复调用。"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import rows,durable,utc
from videoqa_full.prepare import verify_frozen_rd
from videoqa_lmms_qwen.run import METHODS,PYTHON,verify,validate
from videoqa_lmms_bridge.common import load_task
from videoqa_lmms_bridge.frames import FrameProvider


def audit(run):
    """接口：完整8100结果后才检查恢复；不调用任何真实或合成模型。"""
    assert read_json(run/'execution.json')['status']=='completed'
    spec=verify(run);data=rows();summary=read_json(run/'summary.json');seen={}
    # 步骤1：原任务聚合和独立计数一致，来源文件和代码快照均未变。
    for m in METHODS:
        group={r['question_id']:validate(run,r,m,spec) for r in data};seen[m]=group
        c=sum(r['correct'] for r in group.values())
        score=load_task().videomme_aggregate_results([r['answer']['framework_metric'] for r in group.values()])
        assert abs(score-100*c/2700)<1e-10 and c==summary['accuracy']['qwen__'+m]['all2700']['all']['correct']
    for q in seen[METHODS[0]]:assert len({seen[m][q]['gpu'] for m in METHODS})==1
    for n,h in spec['source_results'].items():assert sha256(n)==h
    for n,h in spec['code_sha256'].items():assert sha256(run/'code_snapshot'/n)==h
    release=verify_frozen_rd();frozen=read_json(ROOT/release['frozen_manifest_path'])['sha256']
    for n,h in frozen.items():assert sha256(ROOT/n)==h,n
    # 步骤2：明确范围的RGB重新读取，不宣称全部源视频再次全量解码。
    sample=list(dict.fromkeys(spec['pilot_question_ids']+['395-2','844-1','872-2']))
    for m in METHODS:
        for q in sample:
            r=seen[m][q];images,hashes=FrameProvider(r['selection']).decode()
            try:assert hashes==r['rgb_sha256']
            finally:
                for im in images:im.close()
    # 步骤3：完成恢复只能重做统计，三类原始文件与模型工作进程记录必须不变。
    def hashes():
        paths=[p for folder in ('results','calls','raw_generations') for p in (run/folder).glob('*/*.json')]
        assert len(paths)==24300
        paths+=list((run/'workers').glob('*.json'));return {str(p.relative_to(run)):sha256(p) for p in paths}
    before=hashes()
    with (run/'resume_acceptance.log').open('w') as log:
        subprocess.run([PYTHON,str(ROOT/'scripts/run_lmms_qwen_full.py'),'--run-id',run.name,'--resume'],
            cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)
    assert hashes()==before
    result=dict(status='passed',new_calls=8100,resume_new_calls=0,unchanged_result_call_raw_files=24300,
        source_results_checked=8100,rd_frozen_files_checked=len(frozen),rgb_redecoded_questions_per_method=sample,
        failures=[str(p.relative_to(run)) for p in (run/'failures').glob('*.json')],
        protocol_sha256=sha256(run/'protocol.json'),utc=utc())
    durable(run/'completion_audit.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--wait-pid',type=int);a=p.parse_args()
    assert Path(a.run_id).name==a.run_id and a.run_id not in ('.','..')
    run=ROOT/'outputs/full_videoqa'/a.run_id
    if a.wait_pid:
        while True:
            try:os.kill(a.wait_pid,0)
            except ProcessLookupError:break
            time.sleep(15)
    audit(run)
