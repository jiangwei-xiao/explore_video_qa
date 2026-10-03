"""固定四题证据组偏好诊断：8次镜像呈现，不产生方法准确率或自动选帧器。"""
import argparse
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from videoqa_runtime.common import read_json, sha256, offline_environment
from videoqa_full.state import durable, exclusive, rows, invoke_once, utc
from videoqa_audit.query_input_probe import BASE
from videoqa_budget32.frames import FrameProvider
from videoqa_budget32.observe import tensor_hash

RUN = ROOT/'outputs/diagnostics/evidence_binding_probe4_20261003_r2'
REFERENCE = ROOT/'outputs/diagnostics/rd12_exp19__qwen32__dev200__20261003_r2'
BANKS = {
    '409-3': {'evidence': [77, 80], 'contrast': [99, 122]},
    '556-1': {'evidence': [42, 47], 'contrast': [248, 254]},
    '383-2': {'evidence': [1118, 945], 'contrast': [249, 279]},
    '705-1': {'evidence': [555, 2383], 'contrast': [371, 1861]},
}
LABELS = {'FIRST', 'SECOND', 'TIE', 'UNCERTAIN'}
INSTRUCTION = '''Compare two groups of still frames from the same video. They are separate observations, not a continuous clip.
Choose the group with clearer visible evidence for answering the question as asked. Respect the question's object and event constraints. A picture merely resembling an answer option is not enough.
Do not answer the video question. Do not infer an unseen event or identify a person from their face. Use visible objects, actions, written text and context.
Return exactly FIRST or SECOND. Return TIE if equally useful, or UNCERTAIN if you cannot tell.'''


def parse_preference(raw):
    """接口：只接受四个完整标签；格式错误不重试，也不从解释中挑选有利词。"""
    value = raw.strip().upper()
    return value if value in LABELS else None


def messages(question, options, frame_groups):
    """接口：只接收已排序两组帧，不接收人工偏好标签、标准答案或历史模型预测。"""
    content = [dict(type='text', text=INSTRUCTION+'\n\nQuestion:\n'+question+'\nOptions:\n'+'\n'.join(options))]
    # 步骤1：模型只见FIRST/SECOND与实际时间，不见内部evidence/contrast标注。
    for name, frames in zip(['FIRST', 'SECOND'], frame_groups):
        for j, frame in enumerate(frames, 1):
            content.extend([dict(type='text', text=f'Group {name}, frame {j}, source time {frame["timestamp_seconds"]:.3f} seconds:'), dict(type='image')])
    content.append(dict(type='text', text='Which group has clearer visible evidence? Reply FIRST, SECOND, TIE, or UNCERTAIN only.'))
    return [dict(role='system', content='You are a helpful assistant.'), dict(role='user', content=content)]


def prepare():
    """接口：冻结4题16源帧及镜像条件；只CPU解码，先留待画面验收，不加载模型。"""
    with exclusive(RUN/'prepare.lock'):
        if (RUN/'protocol.json').exists():
            return verify()
        data = {r['question_id']: r for r in rows()}
        source = {str(Path(__file__).resolve()): sha256(__file__)}
        records = {}
        # 步骤1：本库由已审案例构造，仅作能力诊断；不宣称随机、盲审或定位精确率。
        from PIL import Image, ImageDraw
        for q, bank in BANKS.items():
            path = BASE/'pools'/(q+'.json'); pool = read_json(path); source[str(path)] = sha256(path)
            ids = sorted(set(bank['evidence']+bank['contrast']), key=lambda i: pool['candidates'][i]['source_pts'])
            frames = [pool['candidates'][i] for i in ids]
            images, hashes = FrameProvider(dict(video=pool['video'], selected_frames=frames)).decode()
            lookup = dict(zip(ids, images))
            board = Image.new('RGB', (1280, 210), 'white'); draw = ImageDraw.Draw(board)
            for j, i in enumerate(bank['evidence']+bank['contrast']):
                preview = lookup[i].copy(); preview.thumbnail((316, 178)); board.paste(preview, (j*320, 0)); preview.close()
                draw.text((j*320, 180), f'{"evidence" if j<2 else "contrast"}: c{i} {pool["candidates"][i]["timestamp_seconds"]:.3f}s', fill='black')
            dest = RUN/'review'/(q+'.jpg'); dest.parent.mkdir(parents=True, exist_ok=True); board.save(dest, quality=94); board.close()
            for image in images:
                image.close()
            records[q] = dict(question=data[q]['question'], options=data[q]['options'], video=pool['video'],
                              groups={k: [pool['candidates'][i] for i in indices] for k, indices in bank.items()},
                              source_rgb_sha256={str(i): h for i, h in zip(ids, hashes)}, board_sha256=sha256(dest))
        # 步骤2：诊断额度保守合并：此前4次修复＋本批最多8次，合成另列。
        prior = ROOT/'outputs/diagnostics/evidence_repairs_qwen32_20261003_r1'
        calls = list((prior/'calls').glob('*/*.json')); assert len(calls)==4
        source.update({str(p): sha256(p) for p in calls})
        spec = dict(question_ids=list(BANKS), records=records, sources=source,
                    max_real_calls=8, prior_diagnostic_calls=4, combined_diagnostic_ceiling=16,
                    labels=sorted(LABELS), media='four separately labelled still images, not benchmark video input',
                    max_new_tokens=8, seed=2027, expected_preference='evidence',
                    scope='Curated post-hoc capability probe; mirrored presentation; not algorithm accuracy')
        durable(RUN/'protocol.json', spec)
        print('Prepared four curated image banks; generation calls 0', flush=True)
        return spec


def verify():
    """接口：源数据或代码变化立即停止；不把新准备覆盖旧诊断。"""
    spec = read_json(RUN/'protocol.json')
    for path, digest in spec['sources'].items():
        assert sha256(path)==digest, path
    for q, record in spec['records'].items():
        assert sha256(RUN/'review'/(q+'.jpg'))==record['board_sha256']
    return spec


def finish_raw(raw, tokenizer):
    """接口：只解码已完成的原始返回，恢复不再次执行生成。"""
    assert raw['generation_state']=={'started': True, 'returned': True}
    assert raw['measurements']['prefill_calls']==1 and raw['measurements']['checked_logit_steps']>0
    assert 1 <= len(raw['generated_token_ids']) <= 8 and len(raw['image_grid_thw'])==4
    text = tokenizer.decode(raw['generated_token_ids'], skip_special_tokens=True).strip()
    return dict(raw_output=text, parsed_preference=parse_preference(text), observed=raw)


def generate(backend, images, frame_groups, question, options, state, raw_path, identity):
    """接口：独立多图能力诊断；不调用或改写正式视频QA模板与GenerationConfig。"""
    import numpy as np
    from transformers import GenerationConfig
    from videoqa_full.qwen_backend import MIN_PIXELS, MAX_PIXELS
    torch = backend.torch
    torch.cuda.synchronize(); start = time.perf_counter()
    prompt = backend.processor.apply_chat_template(messages(question, options, frame_groups), tokenize=False, add_generation_prompt=True)
    inputs = backend.processor(text=[prompt], images=images,
        images_kwargs={'size': {'shortest_edge': MIN_PIXELS, 'longest_edge': MAX_PIXELS}},
        padding=False, truncation=False, return_tensors='pt')
    grid = inputs['image_grid_thw'].tolist()
    assert len(grid)==len(images)==4 and 'pixel_values_videos' not in inputs
    assert int((inputs['input_ids']==backend.model.config.image_token_id).sum())==sum(int(np.prod(g)//4) for g in grid)
    assert int((inputs['input_ids']==backend.model.config.video_token_id).sum())==0
    n = int(inputs['input_ids'].shape[1]); assert n+8 <= backend.model.config.max_position_embeddings
    pixel_hash = tensor_hash(inputs['pixel_values'])
    ids = inputs['input_ids'][0].tolist(); inputs = inputs.to('cuda:0')
    config = GenerationConfig(max_new_tokens=8, do_sample=False, num_beams=1, use_cache=True,
        eos_token_id=backend.model.generation_config.eos_token_id,
        pad_token_id=backend.processor.tokenizer.pad_token_id, bos_token_id=backend.processor.tokenizer.bos_token_id)
    measured = {'prefill_calls': 0, 'checked_logit_steps': 0}
    def check_input(module, args, kwargs):
        """中文说明：仅首次带图前向核验实际像素，不把解码步当作额外视觉调用。"""
        if kwargs.get('pixel_values') is not None:
            assert tensor_hash(kwargs['pixel_values'])==pixel_hash
            measured['prefill_calls'] += 1
    def check_logits(module, args, output):
        """中文说明：逐生成步检查末位logits有限，不读取其值选择更有利结果。"""
        assert torch.isfinite(output[:,-1,:]).all(); measured['checked_logit_steps'] += 1
    hooks = [backend.model.register_forward_pre_hook(check_input, with_kwargs=True), backend.model.lm_head.register_forward_hook(check_logits)]
    try:
        # 步骤1：调用账本由外层先落盘；返回后的Token先检查点，再做标签解析。
        state['started']=True
        with torch.inference_mode():
            output = backend.model.generate(**inputs, generation_config=config)
        state['returned']=True; torch.cuda.synchronize()
        raw = dict(identity=identity, generation_state=dict(state), generated_token_ids=output[0,n:].tolist(),
                   input_token_ids=ids, prompt=prompt, image_grid_thw=grid, processor_pixel_sha256=pixel_hash,
                   generation_config=config.to_dict(), measurements=measured,
                   total_seconds=time.perf_counter()-start, load_report=backend.load_report)
        durable(raw_path, raw)
    finally:
        for hook in hooks:
            hook.remove()
    # 步骤2：不符合观测协议则暂停；已有返回仍保留，不自动重试。
    assert measured['prefill_calls']==1 and measured['checked_logit_steps']>0
    assert 1 <= len(raw['generated_token_ids']) <= 8
    return finish_raw(raw, backend.processor.tokenizer)


def run():
    """接口：固定8个调用ID，后台执行；不确定生成拒绝重复，镜像顺序不改变源帧内容。"""
    with exclusive(RUN/'run.lock'):
        spec = verify(); fp = sha256(RUN/'protocol.json')
        acceptance = read_json(RUN/'review_acceptance.json')
        assert acceptance['status']=='passed' and acceptance['protocol_sha256']==fp
        from videoqa_budget32.run import verify as verify_reference
        verify_reference(REFERENCE)
        allowed = {(q, o) for q in spec['question_ids'] for o in ['forward', 'reversed']}
        calls = list((RUN/'calls').glob('*/*.json'))
        assert len(calls)<=8 and all((p.stem,p.parent.name) in allowed for p in calls)
        # 步骤1：只恢复已有完整原始返回；没有返回检查点的未完成ID必须暂停。
        from transformers import AutoProcessor
        from videoqa_full.state import QWEN_PATH
        tokenizer = AutoProcessor.from_pretrained(QWEN_PATH, local_files_only=True, use_fast=False).tokenizer
        for path in calls:
            c = read_json(path); assert c['protocol_sha256']==fp
            if c['status']!='completed':
                raw = read_json(RUN/'raw'/path.parent.name/path.name)
                assert raw['identity']==dict(question_id=path.stem, orientation=path.parent.name, protocol_sha256=fp)
                c.update(status='completed', generation_state=raw['generation_state'], answer=finish_raw(raw,tokenizer), recovered_postprocessing=True)
                durable(path,c)
        pending = [(q,o) for q in spec['question_ids'] for o in ['forward','reversed'] if not (RUN/'calls'/o/(q+'.json')).exists()]
        if pending:
            from videoqa_runtime.baseline_run import available_gpus
            free, states = available_gpus(['0']); assert '0' in free, 'GPU0 unavailable; wait without changing device'
            os.environ['CUDA_VISIBLE_DEVICES']='0'; offline_environment()
            import torch
            from PIL import Image
            from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
            torch.set_num_threads(2); torch.set_num_interop_threads(1)
            durable(RUN/'status.json',dict(status='loading',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            backend = QwenFA2VideoBackend()
            # 步骤2：固定四色合成单独记账，不用真实题或合成标签准确率调配置。
            synth_images=[Image.new('RGB',(640,360),c) for c in [(200,0,0)]*2+[(0,0,200)]*2]
            synth_frames=[[dict(timestamp_seconds=t) for t in [0.,1.]], [dict(timestamp_seconds=t) for t in [10.,11.]]]
            invoke_once(RUN/'synthetic','warmup','images',fp,'0',lambda state:generate(backend,synth_images,synth_frames,
                'Which group shows red?', ['Red','Blue','Green','Black'],state,RUN/'synthetic/raw.json',dict(kind='synthetic',protocol_sha256=fp)))
            for im in synth_images: im.close()
            # 步骤3：同题两种呈现同卡同进程；图像与真PTS跟随组一起交换，不添加人工答案说明。
            for q, orientation in pending:
                r=spec['records'][q]; groups=['evidence','contrast'] if orientation=='forward' else ['contrast','evidence']
                frames=[f for g in groups for f in r['groups'][g]]
                ordered=sorted(frames,key=lambda f:f['source_pts'])
                images, hashes=FrameProvider(dict(video=r['video'],selected_frames=ordered)).decode()
                lookup={f['source_pts']:im for f,im in zip(ordered,images)}
                for f,h in zip(ordered,hashes): assert h==r['source_rgb_sha256'][str(f['candidate_index'])]
                pictures=[lookup[f['source_pts']] for f in frames]
                durable(RUN/'status.json',dict(status='running',question_id=q,orientation=orientation,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                identity=dict(question_id=q,orientation=orientation,protocol_sha256=fp)
                invoke_once(RUN,q,orientation,fp,'0',lambda state:generate(backend,pictures,[r['groups'][g] for g in groups],
                    r['question'],r['options'],state,RUN/'raw'/orientation/(q+'.json'),identity))
                for im in images: im.close()
        # 步骤4：按人工事前偏好评估顺序一致性；不是Video-MME答题正确率。
        results=[]
        for q in spec['question_ids']:
            pair={o:read_json(RUN/'calls'/o/(q+'.json'))['answer']['parsed_preference'] for o in ['forward','reversed']}
            results.append(dict(question_id=q,labels=pair,both_orders_prefer_evidence=pair=={'forward':'FIRST','reversed':'SECOND'}))
        assert len(list((RUN/'calls').glob('*/*.json')))==8
        durable(RUN/'summary.json',dict(status='completed',results=results,real_diagnostic_calls=8,prior_calls=4,
            total_current_phase_diagnostic_calls=12,synthetic_calls=1,scope='Curated preference probe, not algorithm accuracy'))
        durable(RUN/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        print(results,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','run'],required=True);args=parser.parse_args()
    if args.stage=='prepare': prepare()
    else: run()
