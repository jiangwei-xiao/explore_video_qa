"""M16/K32资产可复用预检；不是旧RD诊断复用为exp12成绩，不运行问答。"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
from pathlib import Path
import sys
import argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_budget32.selection import config
from videoqa_audit.query_input_probe import BASE
from videoqa_methods.retarget_seed_pool import retarget_pool


def main(run):
    """接口：200题逐目标复验，M32控制须精确重建原候选和特征，缺资产立即停止。"""
    with exclusive(run/'run.lock'):
        data=ROOT/'data/manifests/video_mme_development_200_20260920.json';ids=[r['question_id'] for r in read_json(data)['rows']]
        sources={str(p):sha256(p) for p in [Path(__file__),data,ROOT/'configs/retrieval_density_v1.json',
            ROOT/'src/videoqa_methods/retarget_seed_pool.py',ROOT/'src/videoqa_methods/density_selection.py']}
        records=[]
        # 步骤1：先重建M32作为控制，证明复用映射没有更换采样协议或错位特征。
        for q in ids:
            pp=BASE/'pools'/(q+'.json');fp=BASE/'features'/(q+'.npy');sources[str(pp)]=sha256(pp);sources[str(fp)]=sha256(fp)
            old=read_json(pp);f=np.load(fp,allow_pickle=False)
            control,cf=retarget_pool(old,f,config(32,32))
            assert control['plan']==old['plan'] and control['candidates']==old['candidates'],q
            assert control['original_candidate_indices']==list(range(len(f))) and np.array_equal(cf,f),q
            new,nf=retarget_pool(old,f,config(32,16));assert np.array_equal(nf,f[new['original_candidate_indices']])
            # 步骤2：保存所有新目标与裁剪记录；原池不改写，不生成虚假的新计算成本。
            path=run/'pools'/(q+'.json')
            if path.exists():assert read_json(path)==new
            else:durable(path,new)
            records.append(dict(question_id=q,old_regions=len(old['plan']['regions']),new_regions=len(new['plan']['regions']),
                old_fine=len(old['candidates'])-old['initial_count'],new_fine=len(new['candidates'])-new['initial_count'],
                new_pool_count=len(new['candidates']),pool_sha256=sha256(path)))
            durable(run/'status.json',dict(status='running',completed=len(records),total=200,pid=os.getpid(),utc=utc()))
        assert len(records)==200
        result=dict(status='passed',question_count=200,sources=sources,records=records,new_model_calls=0,
            old_regions=sum(r['old_regions'] for r in records),new_regions=sum(r['new_regions'] for r in records),
            old_fine=sum(r['old_fine'] for r in records),new_fine=sum(r['new_fine'] for r in records),
            short_pools={r['question_id']:r['new_pool_count'] for r in records if r['new_pool_count']<32},
            interpretation='Pool preflight only; no exp12 M16 selections or answers yet')
        out=run/'preflight.json'
        if out.exists():assert read_json(out)==result
        else:durable(out,result)
        durable(run/'status.json',dict(status='completed',completed=200,utc=utc()))
        print({k:v for k,v in result.items() if k not in ['sources','records']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    try:main(Path(p.parse_args().run))
    except BaseException as exc:
        durable(Path(p.parse_args().run)/'status.json',dict(status='paused',error=repr(exc),utc=utc()));raise
