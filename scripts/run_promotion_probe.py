#!/usr/bin/env python3
"""单GPU后台执行64次辅助分类；持久调用状态，失败/不确定生成不自动重试。"""
import os,sys,time,fcntl,traceback
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_runtime.baseline_run import available_gpus
from videoqa_methods.followups import durable,utc
from videoqa_methods.region_context import read_exact_frames
from videoqa_audit.promotion_probe import inspect_frame
OUT=ROOT/'outputs/analysis/promotion_probe64_20260922_r1'

def main(gpu):
    """步骤1核验协议和事前标签；步骤2加载模型；步骤3逐次持久分类并生成独立统计。"""
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        spec=read_json(OUT/'protocol.json');labels=read_json(OUT/'labels.json')
        assert set(labels)=={r['sample_id'] for r in spec['rows']} and len(labels)==64
        for name,digest in spec['code_hashes'].items():assert sha256(ROOT/name)==digest
        fingerprint=dict(protocol=sha256(OUT/'protocol.json'),labels=sha256(OUT/'labels.json'),runner=sha256(Path(__file__)))
        if (OUT/'execution_protocol.json').exists():assert read_json(OUT/'execution_protocol.json')['fingerprint']==fingerprint
        else:durable(OUT/'execution_protocol.json',dict(fingerprint=fingerprint,created_at_utc=utc(),max_started_calls=64,original_QA_calls=0))
        calls=OUT/'calls/promotion';calls.mkdir(parents=True,exist_ok=True)
        existing={p.stem:read_json(p) for p in calls.glob('*.json')}
        assert len(existing)<=64 and set(existing)<=set(labels)
        assert all(x['status']=='completed' and x['fingerprint']==fingerprint for x in existing.values()),'Uncertain call: no retry'
        if len(existing)==64:print('All64 completed; no model loaded or call repeated');return
        available_gpus([gpu]);os.environ['CUDA_VISIBLE_DEVICES']=gpu;offline_environment()
        import torch
        from videoqa_runtime.llava_backend import LlavaBackend
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        started=time.perf_counter();model_start=started
        durable(OUT/'status.json',dict(status='loading',completed=len(existing),pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        backend=LlavaBackend();load_seconds=time.perf_counter()-model_start;warm=time.perf_counter();backend.warmup()
        durable(OUT/'model.json',dict(load_report=backend.load_report,load_seconds=load_seconds,warmup_seconds=time.perf_counter()-warm,synthetic_forwards=1,gpu=gpu))
        try:
            for row in spec['rows']:
                sample=row['sample_id'];path=calls/(sample+'.json')
                if path.exists():continue
                source=ROOT/'outputs/methods/density_e1_200_20260922_r1/results/e1'/f'{row["question_id"]}.json'
                assert sha256(source)==row['source_result_sha256']
                image=read_exact_frames(row['video'],[row['candidate']])[0]
                record=dict(sample_id=sample,task='auxiliary_promotion_classification',status='started_may_generate',fingerprint=fingerprint,
                    started_at_utc=utc(),generation_state={},source_pts=row['candidate']['source_pts'],cohort=row['cohort'])
                assert len(list(calls.glob('*.json')))<64
                durable(path,record);t=time.perf_counter()
                try:
                    result=inspect_frame(backend,image,row['question'],record['generation_state'])
                    record['result']=result  # 已返回输出先保留，后续协议校验失败也不能丢失生成证据。
                    assert result['inference']['visual_tokens']==210
                    record.update(status='completed',result=result,wall_seconds=time.perf_counter()-t,completed_at_utc=utc());durable(path,record)
                except BaseException as error:
                    record.update(status='failed_or_uncertain',error=repr(error));durable(path,record);raise
                finally:image.close()
                durable(OUT/'status.json',dict(status='classifying',completed=len(list(calls.glob('*.json'))),utc=utc()))
            durable(OUT/'execution.json',dict(status='completed',calls=64,original_QA_calls=0,wall_seconds=time.perf_counter()-started,utc=utc()))
            durable(OUT/'status.json',dict(status='completed',completed=64,utc=utc()))
        except BaseException as error:
            durable(OUT/'status.json',dict(status='paused',error=repr(error),traceback=traceback.format_exc(),utc=utc()));raise

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');a=p.parse_args();main(a.gpu)
