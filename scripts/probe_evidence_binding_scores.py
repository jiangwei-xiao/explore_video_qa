"""最后四次诊断：同提示词prefill分数的交换对称化，不新增生成或修改标签模板。"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json, sha256, offline_environment
from videoqa_full.state import durable, exclusive, utc
from videoqa_budget32.frames import FrameProvider
from videoqa_budget32.observe import tensor_hash

SOURCE=ROOT/'outputs/diagnostics/evidence_binding_probe4_20261003_r2'
RUN=ROOT/'outputs/diagnostics/evidence_binding_scores2_20261003_r1'


def decompose(forward_difference, reversed_difference):
    """接口：反对称内容项与对称位置项；这是代数分解，不自动证明模型存在真实效用函数。"""
    return dict(antisymmetric_content=(forward_difference-reversed_difference)/2,
                symmetric_position=(forward_difference+reversed_difference)/2)


def score_once(backend, pictures, raw, path, identity):
    """接口：一次真实prefill，生成次数0；未完成账本禁止重复，原像素/Token必须一致。"""
    import time
    from videoqa_full.qwen_backend import MIN_PIXELS, MAX_PIXELS
    if path.exists():
        value=read_json(path)
        assert value['status']=='completed' and value['identity']==identity
        assert value['matches_previous_generated_first_token'], '前向与原生成不可比，暂停归因'
        return value
    processor, model, torch=backend.processor,backend.model,backend.torch
    tokens={label:processor.tokenizer.encode(label,add_special_tokens=False) for label in ['FIRST','SECOND']}
    assert all(len(t)==1 for t in tokens.values())
    # 步骤1：重建原始输入，既不加换行也不改图组或任务文本。
    inputs=processor(text=[raw['prompt']],images=pictures,
        images_kwargs={'size':{'shortest_edge':MIN_PIXELS,'longest_edge':MAX_PIXELS}},
        padding=False,truncation=False,return_tensors='pt')
    assert inputs['input_ids'][0].tolist()==raw['input_token_ids']
    assert inputs['image_grid_thw'].tolist()==raw['image_grid_thw']
    assert tensor_hash(inputs['pixel_values'])==raw['processor_pixel_sha256']
    inputs=inputs.to('cuda:0'); n=inputs['input_ids'].shape[1]
    prepared=model.prepare_inputs_for_generation(**inputs,cache_position=torch.arange(n,device='cuda:0'),use_cache=True)
    prepared['return_dict']=True
    durable(path,dict(status='started_forward',identity=identity,pid=os.getpid(),utc=utc()))
    torch.cuda.synchronize(); started=time.perf_counter()
    # 步骤2：保留与generate相同的准备路径，只读取末位logits；不采样、不请求第二个Token。
    try:
        with torch.inference_mode():output=model(**prepared)
        torch.cuda.synchronize()
        values=output.logits[0,-1,:]
        assert torch.isfinite(values).all()
        first,second=[float(values[tokens[label][0]].float().cpu()) for label in ['FIRST','SECOND']]
        best=int(values.argmax().cpu())
        result=dict(status='completed',identity=identity,first_logit=first,second_logit=second,
                    difference=first-second,top_token=best,previous_token=raw['generated_token_ids'][0],
                    matches_previous_generated_first_token=best==raw['generated_token_ids'][0],
                    processor_pixel_sha256=raw['processor_pixel_sha256'],prefill_tokens=int(n),
                    seconds=time.perf_counter()-started,forward_calls=1,generation_calls=0)
        durable(path,result)
    except BaseException as exc:
        durable(path,dict(status='failed_or_uncertain',identity=identity,error=repr(exc),utc=utc()))
        raise
    assert result['matches_previous_generated_first_token'], '前向与原生成不一致，停止，不重试'
    return result


def main():
    """接口：仅原固定题序前两题×两种呈现，合计4前向；保守计入16次诊断上限。"""
    with exclusive(RUN/'run.lock'):
        source=read_json(SOURCE/'protocol.json'); ids=source['question_ids'][:2]
        assert ids==['409-3','556-1'] and read_json(SOURCE/'input_audit.json')['status']=='passed'
        assert len(list((SOURCE/'calls').glob('*/*.json')))==8
        paths=[Path(__file__).resolve(),SOURCE/'protocol.json',SOURCE/'input_audit.json']
        paths += [SOURCE/'raw'/o/(q+'.json') for q in ids for o in ['forward','reversed']]
        sources={str(p):sha256(p) for p in paths}
        spec=dict(question_ids=ids,sources=sources,max_real_forward_calls=4,prior_diagnostic_calls=12,
                  total_diagnostic_ceiling=16,rule='u=(d_forward-d_reversed)/2; b=(d_forward+d_reversed)/2',
                  generation_calls=0,scope='No prompt change; first two pre-existing cases; not algorithm accuracy')
        protocol=RUN/'protocol.json'
        if protocol.exists():assert read_json(protocol)==spec
        else:durable(protocol,spec)
        fp=sha256(protocol)
        allowed={(q,o) for q in ids for o in ['forward','reversed']}
        calls=list((RUN/'calls').glob('*/*.json'))
        assert len(calls)<=4 and all((p.stem,p.parent.name) in allowed for p in calls)
        for path in calls:
            c=read_json(path);assert c['status']=='completed' and c['identity']['protocol_sha256']==fp
            assert c['matches_previous_generated_first_token']
        pending=[(q,o) for q in ids for o in ['forward','reversed'] if not (RUN/'calls'/o/(q+'.json')).exists()]
        # 步骤1：无需新前向时直接汇总，恢复不加载模型。
        if pending:
            from videoqa_runtime.baseline_run import available_gpus
            from videoqa_budget32.run import verify as verify_reference
            verify_reference(ROOT/'outputs/diagnostics/rd12_exp19__qwen32__dev200__20261003_r2')
            free,_=available_gpus(['0']);assert '0' in free
            offline_environment()
            import torch
            from PIL import Image
            from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            backend=QwenFA2VideoBackend()
            # 步骤2：先用原合成输入检验直接prefill与generate首Token一致，不用真实题调接口。
            pictures=[Image.new('RGB',(640,360),c) for c in [(200,0,0)]*2+[(0,0,200)]*2]
            score_once(backend,pictures,read_json(SOURCE/'synthetic/raw.json'),RUN/'synthetic.json',dict(kind='synthetic',protocol_sha256=fp))
            for im in pictures:im.close()
            for q,o in pending:
                r=source['records'][q];groups=['evidence','contrast'] if o=='forward' else ['contrast','evidence']
                frames=[f for g in groups for f in r['groups'][g]];ordered=sorted(frames,key=lambda f:f['source_pts'])
                images,hashes=FrameProvider(dict(video=r['video'],selected_frames=ordered)).decode()
                for f,h in zip(ordered,hashes):assert h==r['source_rgb_sha256'][str(f['candidate_index'])]
                lookup={f['source_pts']:im for f,im in zip(ordered,images)}
                durable(RUN/'status.json',dict(status='running',question_id=q,orientation=o,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                score_once(backend,[lookup[f['source_pts']] for f in frames],read_json(SOURCE/'raw'/o/(q+'.json')),
                           RUN/'calls'/o/(q+'.json'),dict(question_id=q,orientation=o,protocol_sha256=fp))
                for im in images:im.close()
        # 步骤3：对称性由定义保证，只有内容项方向是否匹配人工偏好是观测结果。
        records=[]
        for q in ids:
            forward=read_json(RUN/'calls/forward'/(q+'.json'));reverse=read_json(RUN/'calls/reversed'/(q+'.json'))
            value=decompose(forward['difference'],reverse['difference'])
            records.append(dict(question_id=q,forward=forward['difference'],reversed=reverse['difference'],**value,
                                prefers_curated_evidence=value['antisymmetric_content']>0))
        durable(RUN/'summary.json',dict(status='completed',records=records,real_forward_calls=4,new_generations=0,
            total_phase_diagnostic_calls=16,synthetic_forward_calls=1,limitation='Two curated cases; not calibrated utility; symmetry is constructed, not independent validation'))
        durable(RUN/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        print(records,flush=True)


if __name__=='__main__':main()
