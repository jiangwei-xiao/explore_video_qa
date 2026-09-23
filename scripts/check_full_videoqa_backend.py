#!/usr/bin/env python3
"""纯合成后端验收：持久保存成功和失败，不占真实问答调用预算。"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    """接口：指定一个空闲GPU，只做合成前向；异常保存后原样抛出。"""
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['llava','qwen25vl'],required=True)
    p.add_argument('--gpu',default='0');p.add_argument('--output',required=True);a=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    from videoqa_full.state import durable,utc
    from videoqa_runtime.baseline_run import available_gpus
    available,_=available_gpus([a.gpu]);assert a.gpu in available
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    record=dict(model=a.model,gpu=a.gpu,real_answer_calls=0,started_at=utc())
    t=time.perf_counter()
    try:
        if a.model=='qwen25vl':
            from videoqa_full.qwen_backend import QwenVideoBackend
            backend=QwenVideoBackend()
        else:
            from videoqa_runtime.llava_backend import LlavaBackend
            backend=LlavaBackend()
        record['load_report']=backend.load_report
        record['checks']=backend.warmup();record['status']='passed'
    except BaseException as exc:
        record.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        record.update(elapsed_seconds=time.perf_counter()-t,finished_at=utc())
        durable(a.output,record)


if __name__=='__main__':main()
