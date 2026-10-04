"""四组原生QA分布的空白视频对照；复用真实输入logits，不生成新答案。"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from videoqa_runtime.common import read_json, sha256, offline_environment
from videoqa_full.state import durable, exclusive, utc
from videoqa_budget32.frames import FrameProvider
from videoqa_budget32.observe import tensor_hash
from videoqa_audit.visual_prior_probe import distribution_shift
import probe_native_answer_confidence as native
import probe_native_confidence_controls as controls

RUN = ROOT/'outputs/diagnostics/blank_video_prior4_20261004_r1'
OLD = ROOT/'outputs/diagnostics/native_confidence_pairs2_20261003_r2'
CTRL = ROOT/'outputs/diagnostics/native_confidence_controls2_20261003_r1'


def blank_score(backend, images, observed, path, fingerprint):
    """接口：一次32帧纯白视频prefill，仅保存字母分布；未知调用不能重跑。"""
    if path.exists():
        r = read_json(path)
        if r['status'] != 'completed' or r['fingerprint'] != fingerprint:
            raise ValueError('Unsafe blank forward resume')
        return r
    from videoqa_full.qwen_backend import MIN_PIXELS, MAX_PIXELS
    torch, model, processor = backend.torch, backend.model, backend.processor
    # 步骤1：同题提示与尺寸不变，唯一干预是全部像素换成纯白，无真实PTS说明。
    x = processor(text=[native.restore_media_placeholder(observed)], images=None,
        videos=[np.stack([np.asarray(i) for i in images])], fps=2.,
        images_kwargs={'size':dict(shortest_edge=MIN_PIXELS,longest_edge=MAX_PIXELS)},
        padding=False, truncation=False, return_tensors='pt')
    if x['input_ids'][0].tolist() != observed['input_token_ids'] or x['video_grid_thw'].tolist() != observed['video_grid_thw']:
        raise ValueError('Blank changes text or video token layout')
    ids = [processor.tokenizer.encode(label,add_special_tokens=False) for label in 'ABCD']
    if any(len(i)!=1 for i in ids): raise ValueError('Letters must be single tokens')
    ids = [i[0] for i in ids]
    pixel_hash = tensor_hash(x['pixel_values_videos']);x=x.to('cuda:0');n=x['input_ids'].shape[1]
    prepared=model.prepare_inputs_for_generation(**x,cache_position=torch.arange(n,device='cuda:0'),use_cache=True)
    prepared['return_dict']=True
    # 步骤2：前向前持久记账，禁止为了取得更有利分布重复调用。
    durable(path,dict(status='started_forward',fingerprint=fingerprint,pixel_sha256=pixel_hash,utc=utc()))
    with torch.inference_mode(): out=model(**prepared)
    torch.cuda.synchronize();logits=out.logits[0,-1,:].float()
    if not torch.isfinite(logits).all(): raise ValueError('Nonfinite logits')
    result=dict(status='completed',fingerprint=fingerprint,letter_logits=logits[ids].cpu().tolist(),
        token_ids=ids,pixel_sha256=pixel_hash,input_ids=observed['input_token_ids'],
        video_grid_thw=observed['video_grid_thw'],forward_calls=1,generation_calls=0)
    durable(path,result);return result


def main():
    """接口：冻结两组修复与两组反例，4次空白prefill＋1次合成，无新增真实画面前向。"""
    from videoqa_budget32.run import verify
    with exclusive(RUN/'run.lock'):
        verify(native.SOURCE)
        jobs=[]
        for q in ['409-3','556-1']:
            jobs.append(dict(q=q,conditions=['original','repair'],expected='repair',
                results=[native.SOURCE/'results'/c/(q+'.json') for c in ['original','repair']],
                logits=[OLD/'calls'/c/(q+'.json') for c in ['original','repair']]))
        for q,changed in [('383-2','exp19'),('629-3','exp12')]:
            jobs.append(dict(q=q,conditions=['exp07',changed],expected='exp07',
                results=[next(p for qq,c,p in controls.JOBS if qq==q and c==cond) for cond in ['exp07',changed]],
                logits=[CTRL/'calls'/c/(q+'.json') for c in ['exp07',changed]]))
        warm=sorted((native.SOURCE/'synthetic').glob('*.json'))[0]
        sources=[Path(__file__),Path(native.__file__),Path(controls.__file__),warm,
                 ROOT/'src/videoqa_audit/visual_prior_probe.py',native.SOURCE/'protocol.json']
        sources += [p for j in jobs for p in j['results']+j['logits']]
        spec=dict(sources={str(p):sha256(p) for p in sources},questions=[j['q'] for j in jobs],
            max_blank_forwards=4,max_synthetic_forwards=1,new_real_image_forwards=0,new_generations=0,
            metric='KL(real four-letter softmax || blank four-letter softmax), natural logarithm',
            hypothesis='Distribution influence must distinguish repaired evidence and fixed wrong controls before selector use',
            limitation='Blank video is an intervention, not pure language prior; distribution shift is not correctness')
        pp=RUN/'protocol.json'
        if pp.exists():
            if read_json(pp)!=spec: raise ValueError('Protocol changed')
        else: durable(pp,spec)
        fp=sha256(pp)
        for j in jobs:
            values=[read_json(p) for p in j['results']]
            if values[0]['answer']['observed']['input_token_ids'] != values[1]['answer']['observed']['input_token_ids']:
                raise ValueError('Paired text or input layout differs')
            for p in j['logits']:
                if read_json(p)['status']!='completed':raise ValueError('Prior logits incomplete')
        pending=[j for j in jobs if not (RUN/'calls'/(j['q']+'.json')).exists()]
        # 步骤1：只在未完成空白前向存在时加载后端，所有真实画面概率来自旧资产。
        if pending:
            from videoqa_runtime.baseline_run import available_gpus
            free,_=available_gpus(['1'])
            if '1' not in free:raise RuntimeError('GPU1 not idle')
            offline_environment()
            import torch
            from PIL import Image
            from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            durable(RUN/'status.json',dict(status='loading',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            backend=QwenFA2VideoBackend()
            synthetic=[Image.new('RGB',(896,672),(40,i*7,150)) for i in range(32)]
            native.score(backend,synthetic,read_json(warm)['result']['observed'],RUN/'synthetic.json',dict(protocol_sha256=fp,kind='synthetic'))
            for image in synthetic:image.close()
            # 步骤2：精确读取一张源图只确定尺寸，随后替换为纯白，未送入模型。
            for j in pending:
                result=read_json(j['results'][0]);selection=dict(result['selection'])
                selection['selected_frames']=selection['selected_frames'][:1]
                image,_=FrameProvider(selection).decode();size=image[0].size;image[0].close()
                blanks=[Image.new('RGB',size,'white') for _ in range(32)]
                blank_score(backend,blanks,result['answer']['observed'],RUN/'calls'/(j['q']+'.json'),fp)
                for image in blanks:image.close()
        # 步骤3：先计算与标签无关的KL，再在输出层对照事前冻结的正确/错误条件。
        records=[]
        for j in jobs:
            b=read_json(RUN/'calls'/(j['q']+'.json'))
            if b['status']!='completed' or b['fingerprint']!=fp:raise ValueError('Unsafe completed blank call')
            values={c:distribution_shift(read_json(p)['letter_logits'],b['letter_logits']) for c,p in zip(j['conditions'],j['logits'])}
            other=next(c for c in j['conditions'] if c!=j['expected'])
            records.append(dict(question_id=j['q'],kl_nats=values,expected=j['expected'],
                                expected_has_larger_shift=values[j['expected']]>values[other]))
        durable(RUN/'summary.json',dict(records=records,new_blank_forwards=4,synthetic_forwards=1,new_generations=0))
        durable(RUN/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        print(records,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        durable(RUN/'status.json',dict(status='paused',error=repr(exc),utc=utc()));raise
