#!/usr/bin/env python3
"""Qwen只做新框架合成检查，绝不接受真实题号或全量运行参数。"""
import argparse
import os
import sys
import traceback
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--output',required=True);a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    from videoqa_runtime.common import offline_environment
    from videoqa_runtime.baseline_run import available_gpus
    from videoqa_full.state import durable,utc
    offline_environment();assert a.gpu in available_gpus([a.gpu])[0]
    import torch
    from PIL import Image
    from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
    from videoqa_lmms_bridge.qwen_adapter import FrozenInputQwen
    from videoqa_lmms_bridge.bridge import synthetic_row
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    record=dict(status='running',real_calls=0,synthetic_generations=0,checks=[],started_at=utc())
    try:
        backend=QwenFA2VideoBackend();adapter=FrozenInputQwen(backend);record['load_report']=backend.load_report
        original=backend.model.generate
        for n in (1,11,14,16):
            frames=[Image.new('RGB',(1280,720),(50,i*10,180)) for i in range(n)]
            measured=dict(prefill_calls=0)
            def generate(*args,**kwargs):
                assert kwargs['max_new_tokens']==16 and not kwargs['do_sample']
                measured['generate_calls']=measured.get('generate_calls',0)+1
                output=original(*args,**kwargs)
                measured['generated_tokens']=output.shape[1]-kwargs['input_ids'].shape[1]
                assert 1<=measured['generated_tokens']<=16
                return output
            def prefill(module,args,kwargs):
                value=kwargs.get('inputs_embeds')
                if value is not None and measured['prefill_calls']==0:
                    measured['prefill_calls']=1;measured['actual_prefill_tokens']=value.shape[1]
            hook=backend.model.model.register_forward_pre_hook(prefill,with_kwargs=True)
            def finite_logits(module,args,output):
                assert torch.isfinite(output[:,-1,:]).all()
                measured['checked_logit_steps']=measured.get('checked_logit_steps',0)+1
            logit_hook=backend.model.lm_head.register_forward_hook(finite_logits)
            backend.model.generate=generate;record['synthetic_generations']+=1
            try:
                with torch.inference_mode():result=adapter.ask(frames,synthetic_row(n))
            finally:backend.model.generate=original;hook.remove();logit_hook.remove()
            assert measured['prefill_calls']==measured['generate_calls']==1
            assert measured['checked_logit_steps']>0
            assert measured['actual_prefill_tokens']==result['observed']['prefill_tokens']
            record['checks'].append(dict(n=n,result=result,generation=measured))
            for im in frames:im.close()
        record['status']='passed'
    except BaseException as exc:
        record.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        record['finished_at']=utc();durable(a.output,record)
