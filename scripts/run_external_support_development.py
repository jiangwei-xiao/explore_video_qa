"""exp21正式接口重放、输入冻结及后台200题执行，不修改公共推理层。"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import shutil
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_budget32.frames import FrameProvider
from videoqa_budget32 import development
from videoqa_audit.query_input_probe import BASE,FULL
from videoqa_methods.external_support import select_external_support
METHOD='RD-1.2-exp21'
NAME='rd12_exp21__qwen32__dev200__20261003_r1'
RUN=ROOT/'outputs/methods'/NAME
SOURCE=ROOT/'outputs/analysis/external_support200_20261003_r1'
PARENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
PYTHON='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'


def export_rgb(record):
    """接口：只从已验证的冻结选择读取原RGB，不导入历史回答。"""
    q=record['question_id'];dest=RUN/'qa_inputs'/(q+'.json')
    sources={str(p):sha256(p) for p in [RUN/'protocol.json',RUN/'replay_acceptance.json',SOURCE/'selections.json']}
    if dest.exists():
        value=read_json(dest);assert value['sources']==sources and value['selection']==record['selection'];return
    images,hashes=FrameProvider(record['selection']).decode()
    for image in images:image.close()
    durable(dest,dict(question_id=q,method=METHOD,selection=record['selection'],selected_indices=record['selected_indices'],
        rgb_sha256=hashes,sources=sources))


def main():
    """接口：先逐题重放正式接口，再独立运行最多200次QA；不按分数调整或重试。"""
    with exclusive(RUN/'run.lock'):
        source=read_json(SOURCE/'selections.json');summary=read_json(SOURCE/'summary.json')
        assert summary['questions']==200 and summary['new_model_calls']==0
        for p,h in source['sources'].items():assert sha256(p)==h,p
        wrapper=ROOT/'scripts/expand_external_support_development.py'
        assert read_json(SOURCE/'input_manifest.json')['expansion_runner_sha256']==sha256(wrapper)
        ids=[r['question_id'] for r in read_json(DATA)['rows']];records=source['records']
        assert len(ids)==len(set(ids))==200 and [r['question_id'] for r in records]==ids
        assert METHOD in {r['id'] for r in read_json(ROOT/'configs/method_versions.json')['entries']}
        sources=dict(source['sources'])
        for p in [Path(__file__),ROOT/'src/videoqa_methods/external_support.py',wrapper,DATA,SOURCE/'selections.json',SOURCE/'summary.json']:
            sources[str(p)]=sha256(p)
        spec=dict(method=METHOD,question_ids=ids,sources=sources,max_qa_calls=200,allowed_gpus=list(map(str,range(8))),
            configuration='M32/K32; fixed exp12 quotas; at most4 external support references with4-second suppression',
            stop='One frozen version; no additional QA retries or automatic full evaluation')
        pp=RUN/'protocol.json'
        if pp.exists():assert read_json(pp)==spec
        else:durable(pp,spec)
        durable(RUN/'launcher_identity.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤1：源码与配置复制快照，大资产保留来源哈希；旧文件不改写。
        for path,h in sources.items():
            p=Path(path);rel=p.relative_to(ROOT)
            if rel.parts[0] not in {'scripts','src','configs'}:continue
            dest=RUN/'code_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha256(dest)==h
            else:shutil.copy2(p,dest)
        acceptance=RUN/'replay_acceptance.json'
        expected=dict(status='passed',question_count=200,protocol_sha256=sha256(pp),offline_sha256=sha256(SOURCE/'selections.json'))
        if acceptance.exists():assert read_json(acceptance)==expected
        else:
            # 步骤2：正式纯接口逐题对齐离线选择、全部局部前缀及提议，尚不加载问答模型。
            for n,r in enumerate(records,1):
                q=r['question_id'];pool=read_json(BASE/'pools'/(q+'.json'));f=np.load(BASE/'features'/(q+'.npy'),allow_pickle=False)
                selected,trace=select_external_support(pool,f,read_json(FULL/'traces'/(q+'.json')),read_json(PARENT/'traces'/(q+'.json')))
                assert selected==r['selected_indices'] and trace['old_indices']==r['old_indices'],q
                assert trace['proposals']==r['proposals'] and trace['suppressed']==r['suppressed'] and trace['regions']==r['regions'],q
                durable(RUN/'status.json',dict(status='replaying',completed=n,total=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            durable(acceptance,expected)
        with ThreadPoolExecutor(max_workers=4) as executor:
            for n,_ in enumerate(executor.map(export_rgb,records),1):
                durable(RUN/'status.json',dict(status='exporting',completed=n,total=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        directory=RUN/'qa_inputs';manifest=dict(status='completed',question_count=200,frame_budget=32,question_ids=ids,
            dataset_manifest=str(DATA),manifest_sha256=sha256(DATA),parent_run=str(BASE),parent_protocol_sha256=sha256(BASE/'protocol.json'),
            method=METHOD,configuration=spec['configuration'],diagnostic=False,exports={q:sha256(directory/(q+'.json')) for q in ids})
        mp=directory/'selection_manifest.json'
        if mp.exists():assert read_json(mp)==manifest
        else:durable(mp,manifest)
        # 步骤3：复用前五题验收、物理GPU检查、调用账本和完成恢复，异常不静默重启。
        qa=ROOT/'outputs/diagnostics'/NAME;development.prepare(qa,directory,spec['allowed_gpus'])
        durable(RUN/'status.json',dict(status='qa',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        subprocess.run([PYTHON,str(ROOT/'scripts/run_frozen_development.py'),'--run-id',NAME,'--stage','qa','--resume'],check=True)
        assert read_json(qa/'completion_audit.json')['status']=='passed'
        durable(RUN/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        durable(RUN/'status.json',dict(status='paused',error=repr(exc),pid=os.getpid(),utc=utc()));raise
