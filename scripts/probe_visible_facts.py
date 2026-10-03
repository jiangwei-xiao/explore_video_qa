"""六张固定原帧的可见事实描述探针；不是数据集问答或正式选帧模块。"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.query_input_probe import BASE
from videoqa_budget32.frames import FrameProvider
from videoqa_budget32.observe import tensor_hash
RUN=ROOT/'outputs/diagnostics/visible_facts6_20261003_r1'
SOURCE=ROOT/'outputs/diagnostics/evidence_repairs_qwen32_20261003_r1'
CASES=[('409-3',80),('409-3',94),('556-1',42),('556-1',47),('383-2',1118),('383-2',1124)]
PROMPT=('Describe only what is directly visible in this image in one concise sentence. '
        'Mention visible objects, actions, and spatial relationships. '
        'Transcribe a short label only if clearly readable. '
        'Do not infer intentions, unseen events, causes, or names of people.')


def describe(backend,image,path,identity):
    """接口：单图固定描述，输入不含问题/选项/答案；生成开始落盘，未知状态不重试。"""
    if path.exists():
        r=read_json(path)
        if r['status']!='completed' or r['identity']!=identity:raise ValueError('Unsafe caption resume')
        return r
    from videoqa_full.qwen_backend import MIN_PIXELS,MAX_PIXELS
    torch=backend.torch;processor=backend.processor;model=backend.model
    # 步骤1：独立辅助单图协议，不冒充正式视频32帧Tensor；原RGB来自精确PTS读取。
    messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text=PROMPT)])]
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    data=processor(text=[prompt],images=[image],images_kwargs={'size':dict(shortest_edge=MIN_PIXELS,longest_edge=MAX_PIXELS)},
        padding=False,truncation=False,return_tensors='pt')
    ids=data['input_ids'][0];assert int((ids==model.config.video_token_id).sum())==0
    assert int((ids==model.config.image_token_id).sum())==int(data['image_grid_thw'].prod())//4
    observed=dict(prompt=prompt,input_ids=ids.tolist(),image_grid_thw=data['image_grid_thw'].tolist(),
        pixel_sha256=tensor_hash(data['pixel_values']),max_new_tokens=96,media_type='single_image_auxiliary',source_size=list(image.size))
    data=data.to('cuda:0');n=data['input_ids'].shape[1]
    durable(path,dict(status='started_generation',identity=identity,observed=observed,pid=os.getpid(),utc=utc()))
    # 步骤2：固定贪心96Token，不因遗漏关键物体或描述错误而重跑。
    torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode():
        output=model.generate(**data,do_sample=False,num_beams=1,temperature=None,top_p=None,use_cache=True,
            max_new_tokens=96,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id)
    torch.cuda.synchronize();generated=output[0,n:].tolist()
    assert output[0,:n].tolist()==observed['input_ids'] and 0<len(generated)<=96
    result=dict(status='completed',identity=identity,observed=observed,generated_token_ids=generated,
        raw_text=processor.tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False),
        normal_eos=generated[-1]==processor.tokenizer.eos_token_id,seconds=time.perf_counter()-start,generation_calls=1)
    durable(path,result);return result


def main():
    """接口：固定六张帧加一次合成，描述失败只记录，不扩充样本或改提示词。"""
    from videoqa_budget32.run import verify
    with exclusive(RUN/'run.lock'):
        verify(SOURCE)
        paths={q:BASE/'pools'/(q+'.json') for q,_ in CASES};pools={q:read_json(p) for q,p in paths.items()}
        spec=dict(sources={str(p):sha256(p) for p in [Path(__file__),SOURCE/'protocol.json',*paths.values()]},
            cases=[dict(question_id=q,candidate_index=i,source_frame=pools[q]['candidates'][i]) for q,i in CASES],
            prompt=PROMPT,max_real_descriptions=6,max_synthetic_descriptions=1,max_dataset_qa_generations=0,
            question_conditioned=False,answer_options_provided=False,max_new_tokens=96,
            scope='Fixed posthoc evidence/distractor frames; no automatic ranking or semantic sufficiency claim')
        pp=RUN/'protocol.json'
        if pp.exists():assert read_json(pp)==spec
        else:durable(pp,spec)
        fp=sha256(pp);files=list((RUN/'calls').glob('*.json'));allowed={q+'_'+str(i) for q,i in CASES}
        assert len(files)<=6 and all(p.stem in allowed for p in files)
        for p in files:
            r=read_json(p);assert r['status']=='completed' and r['identity']['protocol_sha256']==fp
        pending=[(q,i) for q,i in CASES if not (RUN/'calls'/f'{q}_{i}.json').exists()]
        # 步骤1：恢复已完成描述不加载模型；只用启动时空闲的GPU1。
        if pending:
            from videoqa_runtime.baseline_run import available_gpus
            free,_=available_gpus(['1']);assert '1' in free
            offline_environment()
            import torch
            from PIL import Image
            from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            durable(RUN/'status.json',dict(status='loading',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            backend=QwenFA2VideoBackend();synthetic=Image.new('RGB',(640,360),(255,0,0))
            control=describe(backend,synthetic,RUN/'synthetic.json',dict(kind='synthetic',protocol_sha256=fp));synthetic.close()
            assert control['raw_text'].strip() and control['normal_eos']
            # 步骤2：源帧身份和RGB逐项记录；辅助输入不包含历史预测或描述目标标签。
            for q,i in pending:
                pool=pools[q];images,hashes=FrameProvider(dict(video=pool['video'],selected_frames=[pool['candidates'][i]])).decode()
                durable(RUN/'status.json',dict(status='running',question_id=q,candidate_index=i,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                describe(backend,images[0],RUN/'calls'/f'{q}_{i}.json',dict(question_id=q,candidate_index=i,
                    source_frame=pool['candidates'][i],rgb_sha256=hashes[0],protocol_sha256=fp));images[0].close()
        # 步骤3：只汇总原文，真假与证据作用必须另做画面审核。
        results=[read_json(RUN/'calls'/f'{q}_{i}.json') for q,i in CASES]
        durable(RUN/'summary.json',dict(real_description_calls=6,synthetic_description_calls=1,new_dataset_answers=0,
            descriptions=[dict(question_id=r['identity']['question_id'],candidate_index=r['identity']['candidate_index'],
                               raw_text=r['raw_text'],normal_eos=r['normal_eos']) for r in results]))
        durable(RUN/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        print([(r['identity']['question_id'],r['identity']['candidate_index'],r['raw_text']) for r in results],flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        durable(RUN/'status.json',dict(status='paused',error=repr(exc),pid=os.getpid(),utc=utc()));raise
