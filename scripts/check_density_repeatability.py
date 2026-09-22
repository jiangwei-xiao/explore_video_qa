#!/usr/bin/env python3
"""两道退化题固定新旧输入复测；12次上限，不重选帧、不覆盖正式成绩。"""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import sys
import time

os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_runtime.baseline_run import available_gpus
from videoqa_methods.followups import CallLedger, durable, utc
from videoqa_methods.region_context import read_exact_frames

SOURCES={'old':ROOT/'outputs/methods/retrieval_density_v1_videomme50_20260920_r1',
         'soft':ROOT/'outputs/methods/density_soft_videomme50_20260920_r1'}
QUESTIONS=('260-2','691-3')


def call_plan():
    """接口：固定两题×两套输入×三次，逐题新旧交替；无按结果追加分支。"""
    return [dict(question_id=q,variant=v,repeat=r,call_id=f'{q}__{v}__{r}')
            for q in QUESTIONS for r in (1,2,3) for v in ('old','soft')]


def tensor_identity(tensor):
    """记录实际送入generate的张量字节，不用重建预处理结果替代。"""
    import torch
    data=tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=hashlib.sha256(data).hexdigest())


def summarize(run,plan,fingerprint):
    """只读复核12次账本，逐输入比较输出；不重算或替换原实验成绩。"""
    assert {p.stem for p in (run/'calls/answer').glob('*.json')}=={p['call_id'] for p in plan}
    groups={}
    for job in plan:
        call=read_json(run/'calls/answer'/f'{job["call_id"]}.json')
        assert call['status']=='completed' and call['protocol_sha256']==fingerprint
        assert call['generation_state']=={'started':True,'returned':True}
        answer=call['result']; inp=read_json(run/'inputs'/f'{job["call_id"]}.json')
        assert answer['visual_tokens']==3360 and answer['pixel_shape']==[16,3,384,384]
        key=job['question_id']+'__'+job['variant']
        original=read_json(SOURCES[job['variant']]/'results'/f'{job["question_id"]}.json')
        group=groups.setdefault(key,dict(historical_answer=original['answer']['parsed_answer'],
            historical_gpu=original['physical_gpu'],reference=original['reference_answer'],outputs=[],tokens=[],input_hashes=[]))
        group['outputs'].append(answer['parsed_answer']);group['tokens'].append(answer['generated_token_ids'])
        group['input_hashes'].append(inp['actual_generate_inputs'])
    for group in groups.values():
        assert group['input_hashes'][1:]==group['input_hashes'][:-1]
        group['stable_tokens']=all(x==group['tokens'][0] for x in group['tokens'])
        group['all_match_history']=all(x==group['historical_answer'] for x in group['outputs'])
    result=dict(status='completed',answer_calls=12,blip_calls=0,scope_calls=0,query_calls=0,
                groups=groups,limitation='Same-process fixed-input repeatability only; not cross-device proof or single-frame causality.')
    durable(run/'summary.json',result)
    return result


def execute(run_id,gpu,resume=False):
    """接口：后台独立诊断；固定原推理配置，异常暂停，恢复禁止重答已完成调用。"""
    if not run_id.replace('_','').isalnum():raise ValueError('Invalid run id')
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    run=ROOT/'outputs/diagnostics'/run_id;run.mkdir(parents=True,exist_ok=resume)
    with (run/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ('calls/answer','inputs','code_snapshot'):(run/name).mkdir(parents=True,exist_ok=True)
        # 步骤1：冻结来源与正式代码身份，禁止诊断顺便更改数值配置。
        spec=read_json(SOURCES['soft']/'protocol.json')
        sources={str(p.relative_to(ROOT)):sha256(p) for base in SOURCES.values() for q in QUESTIONS for p in [base/'results'/f'{q}.json']}
        for name,digest in {**spec['code_sha256'],**spec['history_sha256']}.items():
            if sha256(ROOT/name)!=digest:raise RuntimeError('Frozen source changed: '+name)
        protocol=dict(plan=call_plan(),gpu=gpu,maximum_answer_calls=12,source_hashes=sources,
                      source_protocol_hash=sha256(SOURCES['soft']/'protocol.json'),script_sha256=sha256(Path(__file__)))
        if resume:
            assert read_json(run/'protocol.json')==protocol
        else:
            durable(run/'protocol.json',protocol)
            import shutil
            shutil.copy2(__file__,run/'code_snapshot'/Path(__file__).name)
        fingerprint=sha256(run/'protocol.json');plan=call_plan()
        calls=list((run/'calls/answer').glob('*.json'))
        assert len(calls)<=12 and all(p.stem in {j['call_id'] for j in plan} for p in calls)
        for p in calls:
            assert read_json(p)['status']=='completed','Uncertain generation; do not repeat'
        if len(calls)==12:
            summarize(run,plan,fingerprint);print('Verified12 completed calls; no model load or generation');return
        if calls:raise RuntimeError('Partial execution requires review; do not claim one-process continuity')
        available_gpus([gpu]);started=time.perf_counter()
        durable(run/'status.json',dict(status='loading',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        try:
            import torch
            from videoqa_runtime.llava_backend import LlavaBackend
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            backend=LlavaBackend();backend.warmup()
            durable(run/'environment.json',dict(load_report=backend.load_report,gpu=gpu,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),synthetic_forward_calls=1))
            # 步骤2：按原PTS解码四套固定输入；不重新打分、分类或选择。
            prepared={}
            for q in QUESTIONS:
                for variant,base in SOURCES.items():
                    record=read_json(base/'results'/f'{q}.json');s=record['selection']
                    images=read_exact_frames(s['video'],s['selected_frames'])
                    prepared[q,variant]=(record,images)
            original_generate=backend.model.generate;active={};reference_hashes={}
            def observed_generate(*args,**kwargs):
                """只观察真实输入后调用原generate，计算实现和生成配置保持不变。"""
                identity=dict(input_ids=tensor_identity(args[0]),video=tensor_identity(kwargs['images'][0]),
                              attention_mask=tensor_identity(kwargs['attention_mask']))
                key=active['key']
                if key in reference_hashes:assert reference_hashes[key]==identity,'Repeated input bytes changed'
                reference_hashes[key]=identity
                active['input']['actual_generate_inputs']=identity
                durable(run/'inputs'/f'{active["id"]}.json',active['input'])
                return original_generate(*args,**kwargs)
            backend.model.generate=observed_generate
            # 步骤3：调用前持久化，完整保留所有重复结果，不因对错重试。
            for job in plan:
                key=(job['question_id'],job['variant']);record,images=prepared[key];s=record['selection']
                active.clear();active.update(key=key,id=job['call_id'],input={})
                def before_generate(info):
                    assert info['prompt']==record['answer']['prompt'],'Historical prompt changed'
                    assert info['generation_config']['do_sample'] is False
                    assert info['generation_config']['max_new_tokens']==8
                    active['input'].update(info)
                ledger=CallLedger(run,job['call_id'],fingerprint,gpu)
                ledger.invoke('answer',lambda state:backend.answer(images,[r['timestamp_seconds'] for r in s['selected_frames']],
                    s['video']['duration_seconds'],record['question'],record['options'],before_generate=before_generate,generation_state=state))
                durable(run/'status.json',dict(status='running',completed=len(list((run/'calls/answer').glob('*.json'))),pid=os.getpid(),utc=utc()))
            for _,images in prepared.values():
                for image in images:image.close()
            result=summarize(run,plan,fingerprint)
            durable(run/'status.json',dict(status='completed',completed=12,wall_seconds=time.perf_counter()-started,utc=utc()))
            print({k:v['outputs'] for k,v in result['groups'].items()},flush=True)
        except BaseException as error:
            durable(run/'status.json',dict(status='paused',error=repr(error),utc=utc()))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--gpu',default='0');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();execute(args.run_id,args.gpu,args.resume)
