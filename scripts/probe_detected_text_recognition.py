"""对冻结自动框进行一次CTC文字识别；不重检、不人工筛框、不调用问答。"""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive

SOURCE = ROOT / 'outputs/diagnostics/scene_text_detection12_20261004_r1'


def main(run):
    """接口：检测与识别分开记账，保留全部原始低分及误识别输出供复核。"""
    import cv2
    import numpy as np
    import yaml
    import rapidocr_onnxruntime
    from rapidocr_onnxruntime import RapidOCR
    from rapidocr_onnxruntime.ch_ppocr_rec.text_recognize import TextRecognizer
    cv2.setNumThreads(2)
    with exclusive(run/'run.lock'):
        # 步骤1：冻结全部42个真实自动框及1个合成文字框，不手动修框。
        old = read_json(SOURCE/'protocol.json')
        records = read_json(SOURCE/'summary.json')['records']
        inputs = {i['id']: i for i in old['inputs']}
        real_boxes = sum(len(r['boxes']) for r in records if not r['id'].startswith('synthetic'))
        if real_boxes != 42:
            raise ValueError('检测框数量改变')
        package = Path(rapidocr_onnxruntime.__file__).parent
        config = yaml.safe_load((package/'config.yaml').read_text())['Rec']
        config['model_path'] = str(package/config['model_path'])
        config.update(intra_op_num_threads=2, inter_op_num_threads=2)
        sources = {str(p):sha256(p) for p in [Path(__file__), SOURCE/'protocol.json', SOURCE/'summary.json', Path(config['model_path'])]}
        for path,digest in old['sources'].items():
            if sha256(path) != digest:
                raise ValueError('检测来源改变')
        protocol = dict(sources=sources, recognizer_config=config, reporting_text_score=0.5,
                        max_real_regions=42, max_synthetic_regions=1, dataset_answers=0,
                        limitation='同时使用检测及CTC识别，与前一生成式整帧OCR不是单因素对照')
        pp = run/'protocol.json'
        if pp.exists() and read_json(pp) != protocol:
            raise ValueError('识别恢复指纹变化')
        if not pp.exists():
            durable(pp, protocol)
        fingerprint = sha256(pp)
        recognizer = None
        for record in records:
            path = run/'calls'/(record['id']+'.json')
            if path.exists():
                result = read_json(path)
                if result['status'] != 'completed' or result['fingerprint'] != fingerprint:
                    raise ValueError('识别状态未知，不重复前向')
            else:
                image_path = inputs[record['id']]['image']
                if sha256(image_path) != inputs[record['id']]['png_sha256']:
                    raise ValueError('原图改变')
                image = cv2.imread(image_path)
                boxes = np.asarray(record['boxes'], dtype=np.float32)
                crops = RapidOCR.get_crop_img_list(None, image, boxes)
                # 步骤2：包内原透视裁剪和CTC识别；记录实际前向批次，不将帧数当批数。
                batches = []
                if crops:
                    if recognizer is None:
                        recognizer = TextRecognizer(config)
                    original_session = recognizer.session

                    def observe(batch):
                        """记录真实识别前向批次形状，不改变输入或输出。"""
                        batches.append(list(batch.shape))
                        return original_session(batch)

                    recognizer.session = observe
                    durable(path, dict(status='started', fingerprint=fingerprint))
                    start = time.perf_counter()
                    words, model_seconds = recognizer(crops)
                    elapsed = time.perf_counter()-start
                    recognizer.session = original_session
                else:
                    words, model_seconds, elapsed = [], 0.0, 0.0
                result = dict(status='completed', fingerprint=fingerprint, id=record['id'],
                    regions=[dict(box=box, text=str(word[0]), score=float(word[1]),
                                  passes_default_score=float(word[1])>=0.5)
                             for box,word in zip(record['boxes'], words)],
                    forward_batches=batches, seconds=elapsed, model_seconds=model_seconds)
                if len(result['regions']) != len(record['boxes']):
                    raise ValueError('识别框对应数量不符')
                durable(path, result)
            print(record['id'], [(r['text'],round(r['score'],3)) for r in result['regions']], flush=True)
        # 步骤3：保存全部输出，低分不删除；置信度不是证据或答案概率。
        results = [read_json(run/'calls'/(r['id']+'.json')) for r in records]
        durable(run/'summary.json', dict(real_regions=42, synthetic_regions=1, dataset_answers=0,
                                        results=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run)
