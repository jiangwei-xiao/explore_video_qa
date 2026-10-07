"""固定八题纯文本物体查询提取；无视频输入、无答题生成，不选正确备选。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys
import argparse
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,rows,exclusive,utc
from videoqa_audit.grounded_object_queries import object_extraction_prompt,parse_object_queries
QUESTIONS=('409-3','556-1','383-2','103-1','170-2','872-2','844-1','541-1')


def prepare(root):
    """接口：固定原题/原选项与8次文本生成上限；标准答案不保存到提取输入。"""
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():return verify(root)
    data={r['question_id']:r for r in rows()}
    sources={str(Path(__file__)):sha256(__file__),str(ROOT/'src/videoqa_audit/grounded_object_queries.py'):sha256(ROOT/'src/videoqa_audit/grounded_object_queries.py')}
    inputs={q:dict(question=data[q]['question'],options=data[q]['options']) for q in QUESTIONS}
    # 步骤1：复用已验收Qwen模型和环境，权重/任务模板不修改。
    model=Path('/home/models/Qwen/Qwen2.5-VL-7B-Instruct')
    for p in model.iterdir():
        if p.is_file() and p.suffix in ('.safetensors','.json','.txt'):sources[str(p)]=sha256(p)
    for p in (ROOT/'src/videoqa_full/qwen_backend.py',ROOT/'src/videoqa_full/qwen_fa2_backend.py'):
        sources[str(p)]=sha256(p)
    spec=dict(inputs=inputs,sources=sources,max_auxiliary_text_generations=8,max_new_tokens=128,
        qa_calls=0,video_input_calls=0,scope='Query extraction capability, not answer or automated selector; lexical fidelity is insufficient without semantic review',utc=utc())
    import shutil
    (root/'code_snapshot').mkdir(exist_ok=True)
    shutil.copyfile(__file__,root/'code_snapshot'/Path(__file__).name)
    shutil.copyfile(ROOT/'src/videoqa_audit/grounded_object_queries.py',root/'code_snapshot/grounded_object_queries.py')
    durable(root/'protocol.json',spec);return spec


def verify(root):
    """接口：模板与模型字节变化时暂停，已开始但不确定的提取不再次生成。"""
    spec=read_json(root/'protocol.json')
    for p,h in spec['sources'].items():
        if sha256(p)!=h:raise ValueError('Object extraction source changed: '+p)
    return spec


def invoke(root,q,item,backend,spec):
    """接口：原题纯文本贪心JSON提取；返回原始Token、EOS、校验/回退及耗时。"""
    import torch
    model=backend.model;processor=backend.processor;path=root/'calls'/(q+'.json');rawpath=root/'raw'/(q+'.json')
    if path.exists():
        record=read_json(path)
        if record['status']!='completed' or record['protocol_sha256']!=sha256(root/'protocol.json') or sha256(rawpath)!=record['raw_sha256']:
            raise ValueError('Uncertain auxiliary generation; repetition forbidden')
        return record
    context=object_extraction_prompt(item['question'],item['options'])
    prompt=processor.apply_chat_template([dict(role='system',content='You are a careful text parser.'),dict(role='user',content=context)],tokenize=False,add_generation_prompt=True)
    encoded=processor.tokenizer(prompt,return_tensors='pt',truncation=False).to('cuda')
    if encoded['input_ids'].shape[1]+128>model.config.max_position_embeddings:raise ValueError('Extraction context exceeds model')
    # 步骤1：调用前持久化，文本配置局部传入，不改正式QA16Token或模型默认配置。
    durable(path,dict(status='started_may_generate',protocol_sha256=sha256(root/'protocol.json'),prompt=prompt,input_tokens=encoded['input_ids'][0].tolist(),utc=utc()))
    started=time.perf_counter()
    with torch.inference_mode():out=model.generate(**encoded,max_new_tokens=128,do_sample=False,num_beams=1,use_cache=True)
    torch.cuda.synchronize();generated=out[0,encoded['input_ids'].shape[1]:].tolist()
    text=processor.tokenizer.decode(generated,skip_special_tokens=True)
    # 步骤2：返回先落原始字节，再校验JSON；解析/语义失败不重试或逐题润色。
    durable(rawpath,dict(output_token_ids=generated,raw_output=text,returned=True,prompt=prompt))
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    ended=bool(generated and generated[-1] in eos)
    parsed=parse_object_queries(text,item['question'],item['options']) if ended else dict(known=[],choices=[[],[],[],[]],valid=False,fallback=True,reason='generation_not_normally_ended')
    record=dict(status='completed',protocol_sha256=sha256(root/'protocol.json'),raw_sha256=sha256(rawpath),raw_output=text,
        output_token_ids=generated,normal_eos=ended,parsed=parsed,seconds=time.perf_counter()-started,qa_calls=0,utc=utc())
    durable(path,record);return record


def execute(root,gpu):
    """接口：一卡常驻后端八次纯文本，不用预留题或标准答案，也不生成开发题答案。"""
    offline_environment();spec=verify(root)
    if (root/'summary.json').exists():
        saved=read_json(root/'summary.json')
        for p,h in saved['artifacts'].items():
            if sha256(p)!=h:raise ValueError('Completed extraction artifact changed')
        return saved
    import torch
    from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
    torch.set_num_threads(2);torch.manual_seed(2027)
    with exclusive(root/'run.lock'):
        backend=QwenFA2VideoBackend()
        durable(root/'environment.json',dict(gpu=gpu,pid=os.getpid(),sid=os.getsid(0),load_report=backend.load_report,utc=utc()))
        results={}
        # 步骤1：固定顺序只提取文本，不读取任何视频、不决定哪个备选正确。
        for q,item in spec['inputs'].items():
            results[q]=invoke(root,q,item,backend,spec);print(q,results[q]['raw_output'],results[q]['parsed'],flush=True)
        # 步骤2：忠实性规则通过也不能替代语义审核，暂不接入检测或选帧。
        summary=dict(status='completed',results=results,auxiliary_text_generations=8,qa_calls=0,
            valid=sum(r['parsed']['valid'] for r in results.values()),artifacts={str(p):sha256(p) for d in ('calls','raw') for p in (root/d).iterdir()},utc=utc())
        durable(root/'summary.json',summary);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--stage',choices=('prepare','run'),required=True);parser.add_argument('--gpu',default='7')
    args=parser.parse_args()
    if not args.run_id.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run ID')
    root=ROOT/'outputs/diagnostics'/args.run_id
    try:prepare(root) if args.stage=='prepare' else execute(root,args.gpu)
    except BaseException as exc:
        durable(root/('failure_'+str(time.time_ns())+'.json'),dict(error=repr(exc),utc=utc()));raise
