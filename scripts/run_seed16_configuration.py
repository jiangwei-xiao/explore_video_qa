"""exp12 M16/K32独立后台评测，方法ID不变，配置与运行目录隔离。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import subprocess
import shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_budget32 import development
from export_analysis_development import export_one
from videoqa_audit.query_input_probe import BASE
METHOD='RD-1.2-exp12'
RUN_ID='rd12_exp12__m16k32__qwen32__dev200__20261003_r1'
RUN=ROOT/'outputs/methods'/RUN_ID
SOURCE=ROOT/'outputs/analysis/seed16_exp12_200_20261003_r1'
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
PYTHON='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'


def main():
    """接口：冻结200份RGB后独立接续验收与问答；不与同方法M32/K32答案混写。"""
    with exclusive(RUN/'run.lock'):
        source=read_json(SOURCE/'protocol.json');summary=read_json(SOURCE/'summary.json')
        assert summary['status']=='completed' and summary['m32_control_exact']==200 and summary['actual_frame_counts']==[32]
        for p,h in source['sources'].items():assert sha256(p)==h,p
        assert METHOD in {r['id'] for r in read_json(ROOT/'configs/method_versions.json')['entries']}
        ids=[r['question_id'] for r in read_json(DATA)['rows']];assert ids==source['question_ids'] and len(ids)==200
        sources={str(p):sha256(p) for p in [Path(__file__),ROOT/'scripts/export_analysis_development.py',
            SOURCE/'protocol.json',SOURCE/'summary.json',DATA,*[SOURCE/'questions'/(q+'.json') for q in ids]]}
        # 保存本轮新增选择实现与配置；模型、视频和特征只引用哈希，避免复制大资产。
        preflight=ROOT/'outputs/analysis/seed16_pool200_20261003_r1/preflight.json'
        assert source['sources'][str(preflight)]==sha256(preflight)
        snapshot_sources={**read_json(preflight)['sources'],**source['sources'],**sources}
        for path,digest in snapshot_sources.items():
            p=Path(path);relative=p.relative_to(ROOT)
            if relative.parts[0] not in {'src','scripts','configs'}:continue
            assert sha256(p)==digest
            dest=RUN/'code_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha256(dest)==digest
            else:shutil.copy2(p,dest)
        spec=dict(method=METHOD,configuration_key='M16_K32',source_run=str(SOURCE),question_ids=ids,sources=sources,
            max_qa_calls=200,allowed_gpus=list(map(str,range(8))),
            comparison='same exp12 algorithm; only M32 to M16 with K32 fixed; original M32 results stay in original run')
        pp=RUN/'protocol.json'
        if pp.exists():assert read_json(pp)==spec
        else:durable(pp,spec)
        durable(RUN/'launcher_identity.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤1：源帧读取复用旧兼容ID，内部新池ID只在分析轨迹中使用。
        with ThreadPoolExecutor(max_workers=4) as executor:
            for n,_ in enumerate(executor.map(export_one,[(str(RUN),q) for q in ids]),1):
                durable(RUN/'status.json',dict(status='exporting',completed=n,total=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        directory=RUN/'qa_inputs'
        manifest=dict(status='completed',question_count=200,frame_budget=32,question_ids=ids,dataset_manifest=str(DATA),
            manifest_sha256=sha256(DATA),parent_run=str(BASE),parent_protocol_sha256=sha256(BASE/'protocol.json'),
            method=METHOD,configuration='M16/K32; '+source['hypothesis']+'; frozen Qwen32 reference protocol',diagnostic=False,
            exports={q:sha256(directory/(q+'.json')) for q in ids})
        path=directory/'selection_manifest.json'
        if path.exists():assert read_json(path)==manifest
        else:durable(path,manifest)
        # 步骤2：正式入口独立协议哈希隔离调用，前五题按正确运行放行，不按准确率。
        qa=ROOT/'outputs/diagnostics'/RUN_ID;prepared=development.prepare(qa,directory,spec['allowed_gpus'])
        assert prepared['configuration'].startswith('M16/K32; ')
        durable(RUN/'status.json',dict(status='qa',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        subprocess.run([PYTHON,str(ROOT/'scripts/run_frozen_development.py'),'--run-id',RUN_ID,'--stage','qa','--resume'],check=True)
        # 步骤3：完成恢复必须不新增生成；不自动追加参数对照或全量。
        assert read_json(qa/'completion_audit.json')['status']=='passed'
        durable(RUN/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        durable(RUN/'status.json',dict(status='paused',error=repr(exc),pid=os.getpid(),utc=utc()));raise
