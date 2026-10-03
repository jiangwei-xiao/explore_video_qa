"""exp12全量准备资格核验：验证旧冻结资产、200题精确重放；不调用任何模型。"""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import rows, durable, exclusive, utc
from videoqa_audit.query_input_probe import BASE, FULL
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage

RUN=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
DEV=ROOT/'outputs/diagnostics/rd12_exp12__qwen32__dev200__20261002_r1'
COORD=ROOT/'outputs/methods/rd12_exp12__qwen32__dev200__20261002_r1'
SOURCE=ROOT/'outputs/analysis/relevance_coverage_dev200_20261002_r1'
DATA=ROOT/'data/manifests/video_mme_development_200_20260920.json'


def asset_check(q):
    """接口：每题池/特征按原检查点校验字节，exp07全量前缀保持原样。"""
    pool,features,checkpoint=BASE/'pools'/(q+'.json'),BASE/'features'/(q+'.npy'),BASE/'selection_checkpoints'/(q+'.json')
    reference=FULL/'traces'/(q+'.json')
    check=read_json(checkpoint)
    result={str(p):sha256(p) for p in [pool,features,checkpoint,reference]}
    assert result[str(pool)]==check['files'][str(pool.relative_to(BASE))]
    assert result[str(features)]==check['files'][str(features.relative_to(BASE))]
    return q,result


def content_check(job):
    """接口：既有视频或模型文件只读重哈希；输入改变就停止，不重建或替换资产。"""
    path,expected=job
    actual=sha256(path)
    assert actual==expected,path
    return path,actual


def main():
    """接口：独立后台预检，完整匹配后才有资格进入新的全量准备，不在此启动问答。"""
    with exclusive(RUN/'preflight.lock'):
        data=rows(); ids=[r['question_id'] for r in data]
        dev_ids=[r['question_id'] for r in read_json(DATA)['rows']]
        assert len(ids)==2700 and len(dev_ids)==200 and set(dev_ids)<=set(ids)
        for directory,suffix in [(BASE/'pools','.json'),(BASE/'features','.npy'),(FULL/'traces','.json')]:
            assert {p.stem for p in directory.glob('*'+suffix)}==set(ids)
        receipt=read_json(COORD/'freeze_receipt.json')
        assert sha256(COORD/'frozen_artifacts.json')==receipt['sha256']
        frozen=read_json(COORD/'frozen_artifacts.json')['files']
        # 步骤1：先核验旧开发证据与原算法字节；不重新选择开发答案。
        for path,digest in frozen.items():
            assert sha256(ROOT/path)==digest,path
        old_spec=read_json(SOURCE/'protocol.json')
        code={p:h for p,h in old_spec['sources'].items() if p.endswith('.py')}
        for path,digest in code.items():assert sha256(path)==digest,path
        assets={}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures=[executor.submit(asset_check,q) for q in ids]
            for future in as_completed(futures):
                q,hashes=future.result();assets[q]=hashes
                durable(RUN/'preflight_status.json',dict(status='checking_assets',completed=len(assets),total=2700,
                    pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤2：用全量exp07前缀重放旧200题，离散帧与顺序必须完全相同。
        replay=[]
        for q in dev_ids:
            pool=read_json(BASE/'pools'/(q+'.json'))
            selected,trace=select_relevance_coverage(pool,np.load(BASE/'features'/(q+'.npy'),allow_pickle=False),read_json(FULL/'traces'/(q+'.json')))
            old=read_json(COORD/'qa_inputs'/(q+'.json'))
            assert selected==old['selected_indices'],q
            assert [pool['candidates'][i] for i in selected]==old['selection']['selected_frames'],q
            replay.append(q)
        dev_spec=read_json(DEV/'protocol.json');base_spec=read_json(BASE/'protocol.json')
        # 步骤3：视频与Qwen权重只读重哈希，不将旧校验冒充本次内容核验。
        media={base_spec['video_stats'][v]['path']:h for v,h in base_spec['video_hashes'].items()}
        models={p:value['sha256'] for p,value in dev_spec['model_files'].items()}
        targets={**media,**models};checked={}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures=[executor.submit(content_check,job) for job in targets.items()]
            for future in as_completed(futures):
                path,digest=future.result();checked[path]=digest
                durable(RUN/'preflight_status.json',dict(status='checking_video_model_bytes',completed=len(checked),total=len(targets),
                    pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤4：仅出具预检资格记录；200导入＋2500新回答须另有正式账本与执行入口。
        record=dict(status='passed',method='RD-1.2-exp12',full_questions=2700,videos=len(media),
            development_replayed=len(replay),development_ids=dev_ids,planned_new_questions=2500,
            assets=assets,verified_video_model_bytes=checked,method_code=code,
            old_development_freeze_sha256=receipt['sha256'],dataset_sha256=sha256(DATA),
            preflight_code_sha256=sha256(__file__),free_bytes=os.statvfs(RUN).f_bavail*os.statvfs(RUN).f_frsize,
            new_model_calls=0,scope='Preparation eligibility only; no new selections for remaining2500 or QA yet')
        durable(RUN/'preflight.json',record)
        durable(RUN/'preflight_status.json',dict(status='passed',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        print({k:v for k,v in record.items() if k not in ['assets','verified_video_model_bytes','method_code','development_ids']},flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        durable(RUN/'preflight_failure.json',dict(status='paused',error=repr(exc),utc=utc()))
        raise
