"""固定十二题当前输入与既有目标的文字信息面板，非自动候选选择或问答实验。"""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import argparse
import importlib.metadata
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive, utc

TARGETS = {'409-3':[77,80], '556-1':[42], '395-2':[276], '584-3':[582],
           '807-2':[1646], '383-2':[944], '538-1':[9,10,11], '705-1':[2383],
           '893-1':[699,704,706], '305-3':[198,199], '170-2':[10], '191-3':[6,17,28]}
PARENT = ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'


def prepare(run):
    """接口：复用固定面板，每题选中帧与既有目标并集；不读取标准答案。"""
    from videoqa_audit.query_input_probe import BASE
    from videoqa_budget32.frames import FrameProvider
    with exclusive(run/'prepare.lock'):
        if (run/'inputs.json').exists():
            check_inputs(run)
            return
        sources = {str(Path(__file__)):sha256(__file__)}
        entries = []
        # 步骤1：名单由上一阶段冻结目标确定，不因OCR输出调整输入图像。
        for q, targets in TARGETS.items():
            pool_path = BASE/'pools'/(q+'.json')
            selection_path = PARENT/'selections/RD-1.2-exp12'/(q+'.json')
            for p in [pool_path,selection_path]:
                sources[str(p)] = sha256(p)
            pool, selection = read_json(pool_path), read_json(selection_path)
            selected = set(selection['selected_indices'])
            ids = sorted(selected | set(targets), key=lambda i:pool['candidates'][i]['source_pts'])
            # 步骤2：提供器最多32帧，分块仅用于精确读取，所有并集帧都保留。
            for offset in range(0,len(ids),32):
                batch = ids[offset:offset+32]
                images, hashes = FrameProvider(dict(video=pool['video'], selected_frames=[pool['candidates'][i] for i in batch])).decode()
                for i,image,digest in zip(batch,images,hashes):
                    path = run/'images'/f'{q}_{i}.png'
                    path.parent.mkdir(parents=True,exist_ok=True)
                    image.save(path)
                    entries.append(dict(id=f'{q}_{i}', question_id=q,candidate_index=i,
                        selected=i in selected,diagnostic_target=i in targets,source_frame=pool['candidates'][i],
                        image=str(path),rgb_sha256=digest,png_sha256=sha256(path)))
                    image.close()
        if len(entries)>404 or len({r['id'] for r in entries})!=len(entries):
            raise ValueError('面板上限或唯一性异常')
        # 步骤3：只保存来源与展示角色；角色不传给OCR网络。
        durable(run/'inputs.json',dict(sources=sources, entries=entries, max_images=404,
            dataset_answers=0, limitation='事后固定面板，不能估计自动全池检索表现'))


def check_inputs(run):
    """接口：恢复时校验源码、上游选择及图像，禁止静默更新来源。"""
    data = read_json(run/'inputs.json')
    for path,digest in data['sources'].items():
        if sha256(path)!=digest:
            raise ValueError('冻结来源变化')
    for row in data['entries']:
        if sha256(row['image'])!=row['png_sha256']:
            raise ValueError('面板图像变化')
    return data


def recognize(run):
    """接口：独立CPU环境扫描固定图像，沿用前一能力检查参数，不做选帧。"""
    import cv2
    import numpy as np
    import yaml
    import rapidocr_onnxruntime
    from rapidocr_onnxruntime import RapidOCR
    from rapidocr_onnxruntime.ch_ppocr_det.text_detect import TextDetector
    from rapidocr_onnxruntime.ch_ppocr_rec.text_recognize import TextRecognizer
    cv2.setNumThreads(2)
    with exclusive(run/'run.lock'):
        data = check_inputs(run)
        package = Path(rapidocr_onnxruntime.__file__).parent
        cfg = yaml.safe_load((package/'config.yaml').read_text())
        for key in ['Det','Rec']:
            cfg[key]['model_path'] = str(package/cfg[key]['model_path'])
            cfg[key].update(intra_op_num_threads=2,inter_op_num_threads=2)
        sources = {str(p):sha256(p) for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        protocol = dict(input_sha256=sha256(run/'inputs.json'), sources=sources,
            config={k:cfg[k] for k in ['Det','Rec']}, display_score=0.5,
            dependencies={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
            max_detection_calls=len(data['entries']), max_regions_per_image=1000,
            new_qa_calls=0, new_blip_calls=0, manual_region_edits=False)
        pp=run/'protocol.json'
        if pp.exists() and read_json(pp)!=protocol:
            raise ValueError('协议或依赖变化')
        if not pp.exists():durable(pp,protocol)
        fingerprint=sha256(pp)
        detector=recognizer=None
        # 步骤1：逐帧检测和识别，模型不接收问题、答案或诊断目标标记。
        for ordinal,row in enumerate(data['entries']):
            path=run/'calls'/(row['id']+'.json')
            if path.exists():
                result=read_json(path)
                if result['status']!='completed' or result['fingerprint']!=fingerprint:
                    raise ValueError('调用未知，不重试')
                continue
            if detector is None:
                detector=TextDetector(cfg['Det']);recognizer=TextRecognizer(cfg['Rec'])
            image=cv2.imread(row['image'])
            if image is None:raise ValueError('图像读取失败')
            durable(run/'status.json',dict(status='running',completed=ordinal,total=len(data['entries']),pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            durable(path,dict(status='started',fingerprint=fingerprint))
            start=time.perf_counter()
            boxes,det_seconds=detector(image)
            boxes=[] if boxes is None else np.asarray(boxes).tolist()
            crops=RapidOCR.get_crop_img_list(None,image,np.asarray(boxes,dtype=np.float32))
            batches=[]
            original=recognizer.session

            def observe(batch):
                """记录实际识别批次，保持原CTC输入和输出。"""
                batches.append(list(batch.shape))
                return original(batch)

            recognizer.session=observe
            words,rec_seconds=recognizer(crops) if crops else ([],0.0)
            recognizer.session=original
            if len(words)!=len(boxes):raise ValueError('字框与识别不对应')
            # 步骤2：低分与误检均保留，不按预期关键词清洗结果。
            result=dict(status='completed',fingerprint=fingerprint,id=row['id'],boxes=boxes,
                words=[dict(text=str(w[0]),score=float(w[1])) for w in words],
                recognition_batches=batches,det_seconds=det_seconds,rec_seconds=rec_seconds,
                seconds=time.perf_counter()-start)
            durable(path,result)
        # 步骤3：完成恢复不加载模型，汇总真实调用和区域数量，不生成准确率。
        records=[read_json(run/'calls'/(r['id']+'.json')) for r in data['entries']]
        durable(run/'summary.json',dict(images=len(records),detection_calls=len(records),
            recognition_regions=sum(len(r['words']) for r in records),
            recognition_batches=sum(len(r['recognition_batches']) for r in records),dataset_answers=0,
            seconds=sum(r['seconds'] for r in records)))
        durable(run/'status.json',dict(status='completed',completed=len(records),total=len(records),pid=os.getpid(),sid=os.getsid(0),utc=utc()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','recognize'],required=True)
    args=parser.parse_args()
    (prepare if args.stage=='prepare' else recognize)(args.run)
