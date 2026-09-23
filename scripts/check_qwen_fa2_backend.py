#!/usr/bin/env python3
"""FA2独立合成验收：不启动全量，不生成开发/测试题答案。"""
import os
import sys
import time
import traceback
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    """接口：固定1/11/14/16帧合成前向与一次合成generate，结果和失败均落盘。"""
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--output',required=True);a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    from videoqa_runtime.baseline_run import available_gpus
    from videoqa_full.state import durable,utc
    available,_=available_gpus([a.gpu]);assert a.gpu in available
    import torch
    from PIL import Image
    from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    record=dict(status='started',real_answer_calls=0,synthetic_forwards=0,synthetic_generations=0,started_at=utc())
    t=time.perf_counter()
    try:
        # 步骤0：小张量核验FA2可用且与同定义SDPA数值接近，不宣称整模型逐位一致。
        from flash_attn import flash_attn_func
        torch.manual_seed(2027)
        q,k,v=[torch.randn(1,32,4,80,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        with torch.inference_mode():
            actual=flash_attn_func(q,k,v,dropout_p=0.0,causal=False)
            reference=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),dropout_p=0.0).transpose(1,2)
        difference=float((actual.float()-reference.float()).abs().max())
        record['synthetic_kernel_comparison']=dict(max_abs_difference=difference,atol=0.02,rtol=0.05,
            allclose=bool(torch.allclose(actual,reference,atol=0.02,rtol=0.05)))
        assert record['synthetic_kernel_comparison']['allclose']
        del q,k,v,actual,reference
        # 步骤1：复用同一套短帧/奇数/宽高比验收，精度和像素预算不降低。
        backend=QwenFA2VideoBackend();record['load_report']=backend.load_report
        record['checks']=backend.warmup();record['synthetic_forwards']=4
        # 步骤2：合成帧仅检验generate/Token钩子，不纳入真实问答预算或正确率。
        frames=[Image.new('RGB',(1280,720),(i*12,96,160)) for i in range(16)]
        state={};record['synthetic_generations']=1
        record['synthetic_answer']=backend.answer(frames,list(range(16)),17,
            'Which color is visible?', ['Red','Blue','Green','Gray'],generation_state=state)
        assert state=={'started':True,'returned':True}
        record['generation_state']=state;record['status']='passed'
        for im in frames:im.close()
    except BaseException as error:
        record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        record.update(seconds=time.perf_counter()-t,finished_at=utc())
        durable(a.output,record)


if __name__=='__main__':main()
