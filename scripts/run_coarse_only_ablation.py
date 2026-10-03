"""补查消融独立后台协调器，复用已验收200题执行层，不继承旧140题放行门槛。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_budget32 import development
from export_analysis_development import export_one
from videoqa_audit.query_input_probe import BASE
RUN_ID='rd12_exp20__qwen32__dev200__20261003_r1'
METHOD='RD-1.2-exp20'
RUN=ROOT/'outputs/methods'/RUN_ID
SOURCE=ROOT/'outputs/analysis/coarse_only_ablation200_20261003_r1'
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
PYTHON='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'


def main():
    """接口：脱离会话完成精确RGB冻结、先行及200题生成，异常停止、不自动追加调用。"""
    with exclusive(RUN/'run.lock'):
        source=read_json(SOURCE/'protocol.json');summary=read_json(SOURCE/'summary.json')
        assert summary['status']=='completed' and summary['common32_count']==196
        for p,h in source['sources'].items():assert sha256(p)==h,p
        assert METHOD in {r['id'] for r in read_json(ROOT/'configs/method_versions.json')['entries']}
        ids=[r['question_id'] for r in read_json(DATA)['rows']];assert ids==source['question_ids'] and len(ids)==200
        sources={str(p):sha256(p) for p in [Path(__file__),ROOT/'scripts/export_analysis_development.py',
            SOURCE/'protocol.json',SOURCE/'summary.json',DATA,*[SOURCE/'questions'/(q+'.json') for q in ids]]}
        spec=dict(method=METHOD,source_run=str(SOURCE),question_ids=ids,sources=sources,max_qa_calls=200,
            allowed_gpus=list(map(str,range(8))),comparison='exp12 vs remove refinement candidates; no accuracy gate',
            short_pools=summary['short_pools'],common32_count=196)
        pp=RUN/'protocol.json'
        if pp.exists():assert read_json(pp)==spec
        else:durable(pp,spec)
        durable(RUN/'launcher_identity.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤1：复用精确源帧导出，只冻结新输入，不导入旧答案。
        with ThreadPoolExecutor(max_workers=4) as executor:
            for n,_ in enumerate(executor.map(export_one,[(str(RUN),q) for q in ids]),1):
                durable(RUN/'status.json',dict(status='exporting',completed=n,total=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        directory=RUN/'qa_inputs'
        manifest=dict(status='completed',question_count=200,frame_budget=32,question_ids=ids,dataset_manifest=str(DATA),
            manifest_sha256=sha256(DATA),parent_run=str(BASE),parent_protocol_sha256=sha256(BASE/'protocol.json'),
            method=METHOD,configuration=source['hypothesis']+'; same frozen Qwen32 reference protocol',diagnostic=False,
            exports={q:sha256(directory/(q+'.json')) for q in ids})
        path=directory/'selection_manifest.json'
        if path.exists():assert read_json(path)==manifest
        else:durable(path,manifest)
        # 步骤2：原200题执行器负责调用上限、前五题、物理GPU绑定及无重复恢复。
        qa=ROOT/'outputs/diagnostics'/RUN_ID;development.prepare(qa,directory,spec['allowed_gpus'])
        durable(RUN/'status.json',dict(status='qa',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        subprocess.run([PYTHON,str(ROOT/'scripts/run_frozen_development.py'),'--run-id',RUN_ID,'--stage','qa','--resume'],check=True)
        # 步骤3：执行器完成后有600份生成相关记录恢复审计，结果比较不自动开启新实验。
        assert read_json(qa/'completion_audit.json')['status']=='passed'
        durable(RUN/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        durable(RUN/'status.json',dict(status='paused',error=repr(exc),pid=os.getpid(),utc=utc()));raise
