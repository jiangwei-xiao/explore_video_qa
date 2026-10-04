"""WFS发布选帧的固定200题参考评测；复用Qwen执行层，不改RD或参考选择。"""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_budget32.frames import FrameProvider
from videoqa_budget32 import development
from videoqa_audit.query_input_probe import BASE
from run_optional_seed64_development import immutable
import run_optional_seed64_development as reporting

NAME='wfs_release__qwen32__dev200__20261004_r1'
METHOD='REF-WFS-SB'
RUN=ROOT/'outputs/methods'/NAME
QA=ROOT/'outputs/diagnostics'/NAME
SOURCE=ROOT/'outputs/analysis/wfs_release_intake_20261004_r1'
RGB=ROOT/'outputs/analysis/wfs_release_rgb11_20261004_r1'
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
PYTHON='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'


def export(job):
    """接口：导出一题的官方源帧与实际RGB；原视频哈希变化即停止，不导入预测。"""
    q,digest=job;path=SOURCE/'mapped_selections'/(q+'.json')
    if sha256(path)!=digest:raise ValueError('Released mapping changed')
    item=read_json(path);dest=RUN/'qa_inputs'/(q+'.json')
    sources={str(path):digest,str(RUN/'protocol.json'):sha256(RUN/'protocol.json')}
    # 步骤1：已有完整导出只验证来源与内容，不重复解码。
    if dest.exists():
        previous=read_json(dest)
        if previous['sources']!=sources or previous['selection']!=item['selection']:
            raise ValueError('Export resume mismatch')
        return
    if sha256(item['selection']['video']['path'])!=item['source_video_sha256']:
        raise ValueError('Local source video changed')
    # 步骤2：正式取帧接口只读取指定源帧，绝不进行额外采样或选帧。
    images,hashes=FrameProvider(item['selection']).decode()
    for image in images:image.close()
    immutable(dest,dict(question_id=q,method=METHOD,selection=item['selection'],rgb_sha256=hashes,
        sources=sources,released_source_sha256=item['released_source_sha256'],
        source_video_sha256=item['source_video_sha256'],selection_source='official published frame list'))


def prepare():
    """接口：冻结200题发布输入、代码和协议，完成后才允许最多200次问答。"""
    audit=read_json(SOURCE/'audit.json');rgb=read_json(RGB/'summary.json')
    if audit['status']!='passed' or audit['questions']!=2700 or rgb['status']!='passed' or rgb['questions']!=11:
        raise ValueError('Input acceptance incomplete')
    ids=[r['question_id'] for r in read_json(DATA)['rows']]
    if len(ids)!=len(set(ids)) or len(ids)!=200:raise ValueError('Frozen population changed')
    if METHOD not in {r['id'] for r in read_json(ROOT/'configs/method_versions.json')['entries']}:
        raise ValueError('External reference is not registered')
    # 步骤1：来源哈希固定；可维护登记状态不作为算法文件强制绑定。
    paths=[Path(__file__),Path(reporting.__file__),ROOT/'scripts/report_external_support.py',
           SOURCE/'audit.json',RGB/'summary.json',DATA]
    spec=dict(method=METHOD,question_ids=ids,max_qa_calls=200,allowed_gpus=list(map(str,range(8))),
        sources={str(p):sha256(p) for p in paths},released_sha256=sha256(SOURCE/'reproduce_videomme_f32.json'),
        configuration='K32 published WFS-SB a424fc4 source frames; local frozen Qwen reference protocol',
        boundary='External reference, not RD improvement or complete reproduction; no automatic full run')
    immutable(RUN/'protocol.json',spec)
    for path,digest in audit['sources'].items():
        if sha256(path)!=digest:raise ValueError('Intake audit source changed')
    for path,digest in rgb['records'].items():
        if sha256(path)!=digest:raise ValueError('Independent RGB acceptance changed')
    # 步骤2：全部200份RGB在生成前冻结，进度落盘，CPU并发不超过4。
    jobs=[(q,audit['exports'][str(SOURCE/'mapped_selections'/(q+'.json'))]) for q in ids]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for n,_ in enumerate(executor.map(export,jobs),1):
            durable(RUN/'status.json',dict(status='exporting',completed=n,total=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
    directory=RUN/'qa_inputs'
    immutable(directory/'selection_manifest.json',dict(status='completed',question_count=200,frame_budget=32,
        question_ids=ids,dataset_manifest=str(DATA),manifest_sha256=sha256(DATA),parent_run=str(BASE),
        parent_protocol_sha256=sha256(BASE/'protocol.json'),method=METHOD,configuration=spec['configuration'],
        diagnostic=True,exports={q:sha256(directory/(q+'.json')) for q in ids}))
    for path,digest in spec['sources'].items():
        p=Path(path);rel=p.relative_to(ROOT)
        if rel.parts[0] not in {'scripts','src'}:continue
        dest=RUN/'code_snapshot'/rel
        if dest.exists():
            if sha256(dest)!=digest:raise ValueError('Snapshot changed')
        else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    development.prepare(QA,directory,spec['allowed_gpus'])


def main(stage):
    """接口：准备、后台问答与只读恢复分开；不会自动追加全量或失败生成。"""
    with exclusive(RUN/'run.lock'):
        # 步骤1：复用通用配对报告，但明确替换为外部方法身份，不变更其评分实现。
        if stage=='report':
            reporting.NAME,reporting.METHOD,reporting.RUN,reporting.QA=NAME,METHOD,RUN,QA
            reporting.report_only();return
        complete=QA/'completion_audit.json'
        if not (complete.exists() and read_json(complete)['status']=='passed'):
            prepare()
            if stage=='prepare':return
            # 步骤2：先行五题及其余任务由已验收账本执行，完成后释放锁再汇总。
            durable(RUN/'launcher_identity.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            durable(RUN/'status.json',dict(status='qa',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            subprocess.run([PYTHON,str(ROOT/'scripts/run_frozen_development.py'),'--run-id',NAME,'--stage','qa','--resume'],check=True)
    # 步骤3：已完成项只启动报告子进程，不再次准备输入或加载生成模型。
    subprocess.run([PYTHON,str(Path(__file__)),'--stage','report'],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','run','report'],default='run')
    try:main(parser.parse_args().stage)
    except BaseException as exc:
        durable(RUN/'status.json',dict(status='paused',error=repr(exc),utc=utc()));raise
