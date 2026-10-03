"""独立四次prefill诊断：原生视频QA置信度能否反映既有人工修复，不生成答案。"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,utc
from videoqa_budget32.frames import FrameProvider
from videoqa_budget32.observe import tensor_hash
SOURCE=ROOT/'outputs/diagnostics/evidence_repairs_qwen32_20261003_r1'
RUN=ROOT/'outputs/diagnostics/native_confidence_pairs2_20261003_r2'


def restore_media_placeholder(observed):
    """接口：历史处理器可能原位展开text列表，仅折叠已记录数量的视频占位；后续Token必须逐字节重放。"""
    token='<|video_pad|>';prompt=observed['prompt'];count=prompt.count(token)
    if count==1:return prompt
    if count!=observed['visual_tokens'] or token*count not in prompt:
        raise ValueError('Unexpected historical media placeholder layout')
    return prompt.replace(token*count,token,1)


def score(backend,images,observed,path,identity):
    """接口：重建历史像素和Token；账本先于单次前向落盘，不重试不确定状态。"""
    if path.exists():
        value=read_json(path)
        if value.get('status')!='completed' or value['identity']!=identity:raise ValueError('Unsafe resume')
        return value
    torch=backend.torch;model=backend.model;processor=backend.processor
    from videoqa_full.qwen_backend import MIN_PIXELS,MAX_PIXELS
    # 步骤1：完整原生视频输入，不使用多图、FIRST/SECOND提示词或标准答案。
    inputs=processor(text=[restore_media_placeholder(observed)],images=None,videos=[np.stack([np.asarray(x.convert('RGB')) for x in images])],
        fps=2.0,images_kwargs={'size':dict(shortest_edge=MIN_PIXELS,longest_edge=MAX_PIXELS)},
        padding=False,truncation=False,return_tensors='pt')
    assert inputs['input_ids'][0].tolist()==observed['input_token_ids']
    assert tensor_hash(inputs['pixel_values_videos'])==observed['processor_pixel_sha256']
    assert inputs['video_grid_thw'].tolist()==observed['video_grid_thw']
    label_ids=[processor.tokenizer.encode(label,add_special_tokens=False) for label in 'ABCD']
    assert all(len(ids)==1 for ids in label_ids)
    ids=[i[0] for i in label_ids];inputs=inputs.to('cuda:0');n=inputs['input_ids'].shape[1]
    prepared=model.prepare_inputs_for_generation(**inputs,cache_position=torch.arange(n,device='cuda:0'),use_cache=True)
    prepared['return_dict']=True
    # 步骤2：计一次真实prefill，读取末位分布；不调用generate，不向模型暴露答案或修复标签。
    durable(path,dict(status='started_forward',identity=identity,pid=os.getpid(),utc=utc()))
    torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode(): output=model(**prepared)
    torch.cuda.synchronize();logits=output.logits[0,-1,:].float()
    assert torch.isfinite(logits).all()
    chosen=int(logits.argmax());expected=observed['generated_token_ids'][0]
    full_probs=logits.softmax(-1);letter_logits=logits[ids];p=letter_logits.softmax(-1)
    ordered=torch.sort(p,descending=True).values
    result=dict(status='completed',identity=identity,token_ids=ids,
        letter_logits=letter_logits.cpu().tolist(),conditional_letter_probabilities=p.cpu().tolist(),
        letter_mass=float(full_probs[ids].sum()),entropy=float(-(p*p.log()).sum()),
        top_probability=float(ordered[0]),margin=float(ordered[0]-ordered[1]),
        top_token=chosen,previous_first_token=expected,matches_historical_first_token=chosen==expected,
        pixel_sha256=observed['processor_pixel_sha256'],prefill_tokens=n,
        seconds=time.perf_counter()-start,forward_calls=1,generation_calls=0)
    durable(path,result)
    if chosen!=expected:raise ValueError('Historical first-token mismatch; no retry')
    return result


def main():
    """接口：独立新阶段四次前向加一次合成控制，不追加旧16次诊断、不形成算法成绩。"""
    with exclusive(RUN/'run.lock'):
        from videoqa_budget32.run import verify
        verify(SOURCE)
        warm=sorted((SOURCE/'synthetic').glob('*.json'))
        assert len(warm)==1
        files=[Path(__file__),SOURCE/'protocol.json',SOURCE/'visibility_review.json',SOURCE/'completion_audit.json',warm[0]]
        schedule=[('409-3','original'),('409-3','repair'),('556-1','repair'),('556-1','original')]
        files += [SOURCE/folder/condition/(q+'.json') for q,condition in schedule for folder in ['results','raw_generations']]
        protocol=dict(source_hashes={str(p):sha256(p) for p in files},schedule=schedule,
            max_real_forwards=4,max_synthetic_forwards=1,new_generations=0,
            hypothesis='Within each fixed same-question pair, lower entropy/higher confidence might favor informative evidence; not calibrated sufficiency',
            stop='No prompt changes, extra pairs, uncertain retries or automatic selector adoption',
            scope='New bounded native-video confidence diagnostic; prior 16-call phase remains closed')
        pp=RUN/'protocol.json'
        # 步骤1：规范JSON再比较，确保恢复不会因tuple/list差异误判。
        import json
        protocol=json.loads(json.dumps(protocol))
        if pp.exists():assert read_json(pp)==protocol
        else:durable(pp,protocol)
        fp=sha256(pp);existing=list((RUN/'calls').glob('*/*.json'));assert len(existing)<=4
        for p in existing:
            r=read_json(p);assert r['status']=='completed' and r['matches_historical_first_token']
            assert r['identity']['protocol_sha256']==fp
        pending=[(q,c) for q,c in schedule if not (RUN/'calls'/c/(q+'.json')).exists()]
        if pending:
            from videoqa_runtime.baseline_run import available_gpus
            free,_=available_gpus(['1']);assert '1' in free
            offline_environment()
            from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
            import torch
            from PIL import Image
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            durable(RUN/'status.json',dict(status='loading',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            backend=QwenFA2VideoBackend()
            # 步骤2：先复现旧合成首Token及像素，再允许四次真实输入前向。
            images=[Image.new('RGB',(896,672),(40,i*7,150)) for i in range(32)]
            score(backend,images,read_json(warm[0])['result']['observed'],RUN/'synthetic.json',dict(protocol_sha256=fp,kind='synthetic'))
            for image in images:image.close()
            for q,c in pending:
                raw=read_json(SOURCE/'raw_generations'/c/(q+'.json'));r=read_json(SOURCE/'results'/c/(q+'.json'))
                images,hashes=FrameProvider(r['selection']).decode();assert hashes==r['rgb_sha256']
                durable(RUN/'status.json',dict(status='running',question_id=q,condition=c,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                score(backend,images,raw['observed'],RUN/'calls'/c/(q+'.json'),dict(protocol_sha256=fp,question_id=q,condition=c))
                for image in images:image.close()
        # 步骤3：只比较事前规定的指标，不读取GT选择有利概率或修改诊断结论。
        pairs=[]
        for q in ['409-3','556-1']:
            a=read_json(RUN/'calls/original'/(q+'.json'));b=read_json(RUN/'calls/repair'/(q+'.json'))
            pairs.append(dict(question_id=q,entropy_original=a['entropy'],entropy_repair=b['entropy'],
                confidence_original=a['top_probability'],confidence_repair=b['top_probability'],
                repair_has_lower_entropy=b['entropy']<a['entropy'],repair_has_higher_confidence=b['top_probability']>a['top_probability']))
        durable(RUN/'summary.json',dict(pairs=pairs,real_forward_calls=4,new_generations=0,
            interpretation='Posthoc two manually repaired cases, not independent selector validation'))
        durable(RUN/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        print(pairs,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        durable(RUN/'status.json',dict(status='paused',error=repr(exc),pid=os.getpid(),utc=utc()));raise
