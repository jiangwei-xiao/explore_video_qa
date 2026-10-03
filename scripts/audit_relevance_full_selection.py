"""exp12全量独立CPU审计：不复用选择器的曲线或DP实现，不新增模型调用。"""
import argparse
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_budget32.relevance_full import BASE,REFERENCE,METHOD,verify


def independent_optimum(curves,weights,budget):
    """接口：反向标量DP仅计算最优值，不复用原前向DP或依赖原配额。"""
    if len(curves)!=len(weights) or not curves:raise ValueError('Invalid curves')
    table=np.full((len(curves)+1,budget+1),-np.inf,dtype=np.float64)
    table[-1,0]=0.
    # 步骤1：每个区域至少一帧，枚举剩余预算；独立方向减少同一实现错误被重复验证。
    for r in range(len(curves)-1,-1,-1):
        for remaining in range(1,budget+1):
            values=[float(weights[r])*float(curves[r][take])+table[r+1,remaining-take]
                    for take in range(1,min(len(curves[r])-1,remaining)+1)]
            if values:table[r,remaining]=max(values)
    if not np.isfinite(table[0,budget]):raise ValueError('Infeasible allocation')
    return float(table[0,budget])


def wait_pipeline(run):
    """接口：只等待已确认的后台协调器，失败/消失则退出，不重启任何模型任务。"""
    identity=read_json(run/'launcher_identity.json');pid=identity['pid']
    while True:
        state=read_json(run/'pipeline_status.json')
        if state['status']=='completed':return
        if state['status']=='paused':raise RuntimeError('Full pipeline paused')
        proc=Path('/proc')/str(pid)/'cmdline'
        if not proc.exists():raise RuntimeError('Pipeline disappeared without terminal record')
        command=proc.read_bytes().decode().split('\0')
        assert '--run-id' in command and command[command.index('--run-id')+1]==run.name
        durable(run/'independent_audit_status.json',dict(status='waiting_pipeline',pipeline_pid=pid,
            pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        time.sleep(15)


def audit(run):
    """接口：全2700题核验源资格、固定前缀、FP64覆盖与独立最优值，绝不改写正式结果。"""
    with exclusive(run/'independent_audit.lock'):
        spec=verify(run);output=run/'independent_selection_audit.json'
        if output.exists():
            old=read_json(output)
            assert old['protocol_sha256']==sha256(run/'protocol.json') and old['auditor_sha256']==sha256(__file__)
            return old
        records=[];max_curve_error=max_dp_gap=0.
        # 步骤1：矩阵方式一次得到各前缀累计最大相似度，与生产代码的逐帧更新独立。
        for q in spec['question_ids']:
            item=read_json(run/'selections'/METHOD/(q+'.json'));trace=read_json(run/item['trace_file'])
            parent=read_json(REFERENCE/'traces'/(q+'.json'));pool=read_json(BASE/'pools'/(q+'.json'))
            features=np.load(BASE/'features'/(q+'.npy'),allow_pickle=False).astype(np.float64)
            scores=np.asarray(pool['initial_scores'],dtype=np.float64)
            selected=[];curves=[];weights=[];objective=0.;error=0.
            assert len(trace['regions'])==len(parent['regions'])
            for r,old in zip(trace['regions'],parent['regions']):
                assert r['region_id']==old['region_id'] and r['prefix']==old['prefix']
                assert r['reference_coarse_indices']==old['reference_coarse_indices'] and r['density_weight']==old['density_weight']
                refs=r['reference_coarse_indices'];prefix=r['prefix'];raw=scores[refs]
                expected=np.full(len(refs),1/len(refs)) if raw.sum()==0 else raw/raw.sum()
                assert np.allclose(expected,r['reference_weights'],atol=1e-12,rtol=0)
                similarity=np.clip(features[refs]@features[prefix].T,0,1)
                computed=np.r_[0.,expected@np.maximum.accumulate(similarity,axis=1)]
                delta=float(np.max(np.abs(computed-np.asarray(r['coverage_curve']))));error=max(error,delta)
                assert delta<=1e-6,(q,r['region_id'],delta)
                count=trace['region_budgets'][str(r['region_id'])]
                assert 1<=count<=len(prefix)
                selected.extend(prefix[:count]);curves.append(r['coverage_curve']);weights.append(r['density_weight'])
                objective+=r['density_weight']*r['coverage_curve'][count]
            # 步骤2：帧与配额闭合；独立最优值允许求和方向的浮点舍入，不重选数学同分项。
            selected.sort(key=lambda i:(pool['candidates'][i]['source_pts'],i))
            assert selected==item['selected_indices'] and len(selected)==len(set(selected))
            assert [pool['candidates'][i] for i in selected]==item['selection']['selected_frames']
            assert set(parent['protected_indices'])<=set(selected)
            assert len(selected)==sum(parent['region_budgets'].values())==min(32,len(pool['candidates']))
            optimum=independent_optimum(curves,weights,len(selected));gap=abs(objective-optimum)
            assert gap<=1e-10 and abs(objective-trace['objective'])<=1e-10,(q,gap)
            max_curve_error=max(max_curve_error,error);max_dp_gap=max(max_dp_gap,gap)
            records.append(dict(question_id=q,curve_error=error,dp_gap=gap,source_frames=len(selected)))
            if len(records)%25==0:durable(run/'independent_audit_status.json',dict(status='running',completed=len(records),total=2700,
                pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤3：只有实际检查完2700题才写passed，不把正式恢复审计代替数学审计。
        assert len(records)==2700
        result=dict(status='passed',questions=2700,records=records,max_fp64_curve_error=max_curve_error,
                    max_independent_dp_gap=max_dp_gap,protocol_sha256=sha256(run/'protocol.json'),
                    auditor_sha256=sha256(__file__),new_model_calls=0)
        durable(output,result);durable(run/'independent_audit_status.json',dict(status='passed',completed=2700,utc=utc()))
        print({k:v for k,v in result.items() if k!='records'},flush=True)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--wait',action='store_true');args=p.parse_args()
    run=Path(args.run)
    try:
        if args.wait:wait_pipeline(run)
        audit(run)
    except BaseException as exc:
        durable(run/'independent_audit_status.json',dict(status='paused',error=repr(exc),utc=utc()))
        raise
