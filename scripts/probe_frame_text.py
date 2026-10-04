"""固定十二张完整原帧的OCR能力验证，不进行问题回答或自动选帧。"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
DEPS = ROOT / 'outputs/diagnostics/frame_text_dependencies_20261004/einops081'
sys.path.insert(0, str(DEPS))
TOKENIZER_DEPS = DEPS.parent / 'tokenizer'
sys.path.insert(0, str(TOKENIZER_DEPS))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive, utc

MODEL = Path('/home/models/PaddleOCR-VL')
CASES = [('584-3',520), ('584-3',582), ('807-2',1646), ('807-2',1511),
         ('705-1',2383), ('538-1',11), ('395-2',276), ('893-1',706),
         ('409-3',80), ('556-1',42), ('383-2',944), ('383-2',1124)]


def prepare(run):
    """接口：原环境按精确PTS解码完整画面，冻结输入、模型及源码，无模型调用。"""
    from PIL import Image, ImageDraw, ImageFont
    from videoqa_audit.query_input_probe import BASE
    from videoqa_budget32.frames import FrameProvider
    # 步骤1：保存整帧，禁止根据识别结果裁剪、改变问题或挑选有利输出。
    with exclusive(run / 'prepare.lock'):
        pp = run / 'protocol.json'
        if pp.exists():
            verify(run)
            return
        sources = {str(Path(__file__)): sha256(__file__)}
        for path in DEPS.parent.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                sources[str(path)] = sha256(path)
        for path in MODEL.iterdir():
            if path.is_file():
                sources[str(path)] = sha256(path)
        cases = []
        for q, index in CASES:
            path = BASE / 'pools' / (q + '.json')
            sources[str(path)] = sha256(path)
            pool = read_json(path)
            frame = pool['candidates'][index]
            images, hashes = FrameProvider(dict(video=pool['video'], selected_frames=[frame])).decode()
            output = run / 'images' / f'{q}_{index}.png'
            output.parent.mkdir(parents=True, exist_ok=True)
            images[0].save(output)
            cases.append(dict(id=f'{q}_{index}', question_id=q, candidate_index=index,
                              source_frame=frame, rgb_sha256=hashes[0], image=str(output),
                              png_sha256=sha256(output), size=list(images[0].size)))
            images[0].close()
        # 步骤2：固定合成文字只验收接口；与真实图像识别分开记账。
        font_path = Path('/App/conda/envs/explore_video_qa/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf')
        sources[str(font_path)] = sha256(font_path)
        synthetic = Image.new('RGB', (640, 160), 'white')
        ImageDraw.Draw(synthetic).text((20, 50), 'FRAME OCR 123', fill='black',
                                      font=ImageFont.truetype(str(font_path), 40))
        path = run / 'images/synthetic.png'
        synthetic.save(path)
        spec = dict(sources=sources, cases=cases, synthetic_image=str(path), synthetic_sha256=sha256(path),
                    prompt='OCR:', max_new_tokens=256, do_sample=False, num_beams=1,
                    dtype='bfloat16', attention='eager', model=str(MODEL), full_frame=True,
                    manual_crop=False, max_real_ocr=12, max_synthetic_ocr=1, max_dataset_answers=0,
                    scope='事后固定能力面板；不是自动检索、语义充分性或准确率实验')
        durable(pp, spec)


def verify(run):
    """接口：每次启动核验全部冻结文件，变化时停止而非更新实验指纹。"""
    spec = read_json(run / 'protocol.json')
    for path, digest in spec['sources'].items():
        if sha256(path) != digest:
            raise ValueError('冻结来源改变: ' + path)
    for case in spec['cases']:
        if sha256(case['image']) != case['png_sha256']:
            raise ValueError('输入图像改变')
    if sha256(spec['synthetic_image']) != spec['synthetic_sha256']:
        raise ValueError('合成输入改变')
    return spec


def recognize(model, processor, image_path, path, fingerprint):
    """接口：调用前持久化，原始返回先落盘；不确定生成禁止重试。"""
    import torch
    from PIL import Image
    if path.exists():
        record = read_json(path)
        if record['fingerprint'] != fingerprint or record['status'] not in ['returned', 'completed']:
            raise ValueError('未知生成状态，禁止重试')
        if record['status'] == 'completed':
            return record
    else:
        # 步骤1：官方单图模板只含OCR指令，不注入题目、选项、预期词或人工区域。
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        messages = [dict(role='user', content=[dict(type='image', image=image), dict(type='text', text='OCR:')])]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors='pt').to('cuda:0')
        input_ids = inputs['input_ids'][0].tolist()
        observed = {k: dict(shape=list(v.shape), dtype=str(v.dtype),
                    sha256=hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
                    for k,v in inputs.items() if isinstance(v, torch.Tensor)}
        record = dict(status='started', fingerprint=fingerprint, input_ids=input_ids,
                      tensors=observed, pid=os.getpid(), utc=utc())
        durable(path, record)
        torch.cuda.synchronize()
        start = time.perf_counter()
        # 步骤2：单次贪心生成；先保存完整返回，后处理失败不得重复生成。
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=256)
        torch.cuda.synchronize()
        record.update(status='returned', output_ids=output[0].tolist(), seconds=time.perf_counter()-start)
        durable(path, record)
        image.close()
    # 步骤3：只解码新增Token，不把用户指令作为识别结果；达到上限如实标记。
    n = len(record['input_ids'])
    if record['output_ids'][:n] != record['input_ids']:
        raise ValueError('生成输入前缀未复现')
    generated = record['output_ids'][n:]
    if not 0 < len(generated) <= 256:
        raise ValueError('输出Token数异常')
    eos = model.generation_config.eos_token_id
    record.update(status='completed', generated_ids=generated,
                  normal_eos=generated[-1] in (eos if isinstance(eos,list) else [eos]),
                  raw_text=processor.tokenizer.decode(generated, skip_special_tokens=True))
    durable(path, record)
    return record


def run_probe(run):
    """接口：辅助环境一次合成加十二次OCR，恢复完整结果不加载模型。"""
    with exclusive(run / 'run.lock'):
        spec = verify(run)
        fingerprint = sha256(run / 'protocol.json')
        ids = ['synthetic'] + [c['id'] for c in spec['cases']]
        paths = [run / 'calls' / (i + '.json') for i in ids]
        if set(p.stem for p in (run / 'calls').glob('*.json')) - set(ids):
            raise ValueError('调用超出冻结名单')
        for path in paths:
            if path.exists():
                r = read_json(path)
                if r['fingerprint'] != fingerprint or r['status'] not in ['returned','completed']:
                    raise ValueError('不安全的恢复状态')
        if not all(p.exists() and read_json(p)['status']=='completed' for p in paths):
            # 步骤1：不升级正式环境，加载辅助环境本地只读权重并固定实际依赖。
            from videoqa_runtime.baseline_run import available_gpus
            free, _ = available_gpus(['1'])
            if '1' not in free:
                raise RuntimeError('GPU1不空闲，未启动生成')
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            torch.set_num_threads(2)
            torch.manual_seed(2027)
            deps = {p: importlib.metadata.version(p) for p in ['torch','transformers','numpy','pillow','einops','sentencepiece','protobuf']}
            dp = run / 'dependencies.json'
            if dp.exists() and read_json(dp) != deps:
                raise ValueError('辅助环境依赖改变')
            if not dp.exists():
                durable(dp, deps)
            durable(run/'status.json', dict(status='loading', pid=os.getpid(), sid=os.getsid(0), utc=utc()))
            model = AutoModelForCausalLM.from_pretrained(str(MODEL), trust_remote_code=True,
                    local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='eager').to('cuda:0').eval()
            processor = AutoProcessor.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
            control = recognize(model, processor, spec['synthetic_image'], paths[0], fingerprint)
            if not control['normal_eos'] or 'FRAME OCR 123' not in control['raw_text']:
                raise ValueError('合成识别未通过，不启动真实帧')
            # 步骤2：固定顺序，每张图最多一次；真实OCR错误或截断不重跑。
            for case,path in zip(spec['cases'], paths[1:]):
                durable(run/'status.json', dict(status='running', current=case['id'],pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                result = recognize(model, processor, case['image'], path, fingerprint)
                print(case['id'], repr(result['raw_text']), flush=True)
        # 步骤3：只汇总文字识别，不计为问答成绩；无文字不等于无证据。
        durable(run/'summary.json', dict(real_ocr=12, synthetic_ocr=1, dataset_answers=0,
            results=[dict(id=i, **read_json(p)) for i,p in zip(ids[1:], paths[1:])]))
        durable(run/'status.json', dict(status='completed', pid=os.getpid(),sid=os.getsid(0),utc=utc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare','run'], required=True)
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else run_probe)(args.run)
