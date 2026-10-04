"""exp22固定等权输入融合的独立200题后台协调器。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.query_input_probe import BASE
from videoqa_budget32.frames import FrameProvider
from videoqa_budget32 import development
from videoqa_methods.temporal_mixture import select_equal_temporal_mixture

METHOD='RD-1.2-exp22'
NAME='rd12_exp22__qwen32__dev200__20261004_r1'
RUN=ROOT/'outputs/methods'/NAME
SOURCE=ROOT/'outputs/analysis/equal_temporal_mixture200_20261004_r2'
PARENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'
PYTHON='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'


def export(q):
    """接口：逐源帧核对历史RGB，只导入输入身份，不导入任何答案。"""
    offline=SOURCE/'selections'/(q+'.json');record=read_json(offline)
    up=BASE/'results/BASE-Uniform'/(q+'.json')
    rp=PARENT/'imported/results/RD-1.2-exp12'/(q+'.json')
    if not rp.exists():rp=PARENT/'results/RD-1.2-exp12'/(q+'.json')
    sources={str(p):sha256(p) for p in [RUN/'protocol.json',offline,up,rp]}
    dest=RUN/'qa_inputs'/(q+'.json')
    if dest.exists():
        old=read_json(dest)
        if old['sources']!=sources or old['selection']!=record['selection']:raise ValueError('输入恢复不一致')
        return
    # 步骤1：构建源身份到RGB的历史映射，重合帧的两路身份也必须一致。
    identity={}
    for path in [up,rp]:
        prior=read_json(path)
        for frame,digest in zip(prior['selection']['selected_frames'],prior['rgb_sha256']):
            key=(frame['source_pts'],frame['source_frame_index'])
            if key in identity and identity[key]!=digest:raise ValueError('两路共享帧RGB不符')
            identity[key]=digest
    images,hashes=FrameProvider(record['selection']).decode()
    for im in images:im.close()
    # 步骤2：实际解码必须复现来源RGB，视频接口不得二次抽帧。
    expected=[identity[(f['source_pts'],f['source_frame_index'])] for f in record['selection']['selected_frames']]
    if hashes!=expected:raise ValueError('新输入与冻结来源像素不一致')
    durable(dest,dict(question_id=q,method=METHOD,selection=record['selection'],
        selected_indices=record['trace']['selected_indices'],rgb_sha256=hashes,sources=sources))


def main():
    """接口：离线重放、输入冻结、先行5题、最多200题；异常停止，不自动扩展。"""
    with exclusive(RUN/'run.lock'):
        summary=read_json(SOURCE/'summary.json')
        if summary['questions']!=200 or summary['new_model_calls']!=0:raise ValueError('准备分母错误')
        for path,digest in summary['sources'].items():
            if sha256(path)!=digest:raise ValueError('离线来源变化')
        ids=[r['question_id'] for r in read_json(DATA)['rows']]
        if METHOD not in {r['id'] for r in read_json(ROOT/'configs/method_versions.json')['entries']}:
            raise ValueError('方法未登记')
        sources=dict(summary['sources'])
        for p in [Path(__file__),SOURCE/'summary.json',ROOT/'scripts/report_external_support.py']:
            sources[str(p)]=sha256(p)
        spec=dict(method=METHOD,question_ids=ids,sources=sources,max_qa_calls=200,
            allowed_gpus=list(map(str,range(8))),
            configuration='K32 equal 1/2 Uniform source-frame measure + 1/2 frozen exp12 measure; exact unique temporal quantiles',
            stop='single fixed ratio; no generation retry, no automatic full evaluation')
        pp=RUN/'protocol.json'
        if pp.exists() and read_json(pp)!=spec:raise ValueError('正式协议变化')
        if not pp.exists():durable(pp,spec)
        # 步骤1：重放全部200题的纯选择与轨迹，源码快照不覆盖历史版本。
        for q in ids:
            old=read_json(SOURCE/'selections'/(q+'.json'))
            u=read_json(BASE/'selections/BASE-Uniform'/(q+'.json'))['selection']
            r=read_json(PARENT/'selections/RD-1.2-exp12'/(q+'.json'))['selection']
            selection,trace=select_equal_temporal_mixture(u,r)
            if selection!=old['selection'] or trace!=old['trace']:raise ValueError('正式接口重放不一致')
        for path,digest in sources.items():
            p=Path(path);rel=p.relative_to(ROOT)
            if rel.parts[0] not in {'src','scripts','configs'}:continue
            dest=RUN/'code_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                if sha256(dest)!=digest:raise ValueError('源码快照变化')
            else:shutil.copy2(p,dest)
        durable(RUN/'launcher_identity.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        with ThreadPoolExecutor(max_workers=4) as executor:
            for n,_ in enumerate(executor.map(export,ids),1):
                durable(RUN/'status.json',dict(status='exporting',completed=n,total=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤2：复用已验收Qwen参考协议，不使用旧8Token或额外时间说明入口。
        directory=RUN/'qa_inputs'
        manifest=dict(status='completed',question_count=200,frame_budget=32,question_ids=ids,
            dataset_manifest=str(DATA),manifest_sha256=sha256(DATA),parent_run=str(BASE),
            parent_protocol_sha256=sha256(BASE/'protocol.json'),method=METHOD,
            configuration=spec['configuration'],diagnostic=False,
            exports={q:sha256(directory/(q+'.json')) for q in ids})
        mp=directory/'selection_manifest.json'
        if mp.exists() and read_json(mp)!=manifest:raise ValueError('输入清单变化')
        if not mp.exists():durable(mp,manifest)
        qa=ROOT/'outputs/diagnostics'/NAME
        development.prepare(qa,directory,spec['allowed_gpus'])
        # 步骤3：后台执行器管理空闲GPU、先行验收、原始返回检查点及安全恢复。
        durable(RUN/'status.json',dict(status='qa',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        subprocess.run([PYTHON,str(ROOT/'scripts/run_frozen_development.py'),'--run-id',NAME,'--stage','qa','--resume'],check=True)
        if read_json(qa/'completion_audit.json')['status']!='passed':raise ValueError('完成审计未通过')
        durable(RUN/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤4：复用已验收的exp12配对报告，只配置运行身份，不重复模型生成。
        import report_external_support as report
        report.NAME=NAME;report.METHOD=METHOD;report.RUN=qa;report.COORD=RUN
        report.main(False)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        durable(RUN/'status.json',dict(status='paused',error=repr(exc),pid=os.getpid(),utc=utc()))
        raise
