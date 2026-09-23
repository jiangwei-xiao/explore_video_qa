#!/usr/bin/env python3
"""两方法完成审计：CPU复算/有限RGB核查/无新增生成的完成恢复。"""
import argparse
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import rows,durable,utc
from videoqa_full.prepare import verify_frozen_rd
from videoqa_lmms_baseline.methods import verify,validate,METHODS,UNIFORM,PYTHON
from videoqa_lmms_bridge.frames import FrameProvider
from videoqa_lmms_bridge.common import load_task
from videoqa_lmms_baseline.run import validate as validate_uniform


def audit(run):
    """接口：5400条完成后才允许进入恢复测试；不加载模型，不追加失败调用。"""
    assert read_json(run/'execution.json')['status']=='completed'
    spec=verify(run);data=rows();summary=read_json(run/'summary.json')
    # 步骤1：全量验证账本、原任务聚合与独立计数，以及均匀对照未被修改。
    base=read_json(UNIFORM/'protocol.json');base_fp=sha256(UNIFORM/'protocol.json')
    groups={'BASE-Uniform':{r['question_id']:validate_uniform(UNIFORM,r,base,base_fp) for r in data}}
    for m in METHODS:groups[m]={r['question_id']:validate(run,r,m,spec) for r in data}
    for m,g in groups.items():
        assert len(g)==2700
        native=load_task().videomme_aggregate_results([r['result']['answer']['framework_metric'] for r in g.values()])
        correct=sum(r['correct'] for r in g.values())
        assert abs(native-100*correct/2700)<1e-10
        assert summary['accuracy'][m]['all2700']['all']['correct']==correct
    for name,h in spec['source_results'].items():assert sha256(name)==h
    for name,h in spec['code_sha256'].items():assert sha256(run/'code_snapshot'/name)==h
    release=verify_frozen_rd();frozen=read_json(ROOT/release['frozen_manifest_path'])['sha256']
    for name,h in frozen.items():assert sha256(ROOT/name)==h,name
    # 步骤2：先行题加中长代表题独立RGB重读；范围明确，不称全量重解码。
    sample=list(dict.fromkeys(spec['pilot_question_ids']+['395-2','844-1','872-2']))
    for m in METHODS:
        for q in sample:
            r=groups[m][q];images,hashes=FrameProvider(r['selection']).decode()
            try:assert hashes==r['rgb_sha256'],(m,q)
            finally:
                for image in images:image.close()
    # 步骤3：恢复完成阶段只能重新汇总，答案/账本/工作进程记录必须不变。
    def immutable_files():
        paths=list((run/'results').glob('*/*.json'))+list((run/'calls').glob('*/*.json'))+list((run/'workers').glob('*.json'))
        return {str(p.relative_to(run)):sha256(p) for p in paths}
    before=immutable_files();assert len(list((run/'calls').glob('*/*.json')))==5400
    with (run/'resume_acceptance.log').open('w') as log:
        subprocess.run([PYTHON,str(ROOT/'scripts/run_lmms_methods_full.py'),'--run-id',run.name,'--resume'],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert immutable_files()==before
    assert not list((run/'failures').glob('*.json'))
    result=dict(status='passed',questions_per_method=2700,new_calls=5400,resume_new_calls=0,
        result_call_files_unchanged=10800,rd_frozen_files_checked=len(frozen),
        source_result_hashes_checked=len(spec['source_results']),rgb_redecoded_questions_per_method=sample,
        protocol_sha256=sha256(run/'protocol.json'),audit_code_sha256=sha256(__file__),utc=utc())
    durable(run/'completion_audit.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    assert Path(a.run_id).name==a.run_id and a.run_id not in ('.','..')
    audit(ROOT/'outputs/full_videoqa'/a.run_id)
