"""固定十二图的自动文字框检测，不识别文字、不回答问题、不调整阈值。"""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import argparse
import importlib.metadata
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive, utc

SOURCE = ROOT / 'outputs/diagnostics/frame_text12_20261004_r3'


def main(run):
    """接口：独立CPU环境沿用默认DB检测参数，结果绑定模型、源码与原帧。

    字框不代表与问题相关；未检测到文字不代表该帧没有有效视觉信息。
    """
    import cv2
    import numpy as np
    import yaml
    from PIL import Image, ImageDraw
    import rapidocr_onnxruntime
    from rapidocr_onnxruntime.ch_ppocr_det.text_detect import TextDetector
    cv2.setNumThreads(2)
    with exclusive(run / 'run.lock'):
        # 步骤1：沿用既有12图；只增加已知文字、纯白两项合成检查。
        old = read_json(SOURCE / 'protocol.json')
        package = Path(rapidocr_onnxruntime.__file__).parent
        config_path = package / 'config.yaml'
        config = yaml.safe_load(config_path.read_text())['Det']
        config['model_path'] = str(package / config['model_path'])
        config.update(intra_op_num_threads=2, inter_op_num_threads=2)
        inputs = [dict(id='synthetic_text', image=old['synthetic_image'], png_sha256=old['synthetic_sha256'])]
        blank = run / 'blank.png'
        if not blank.exists():
            run.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (640,160), 'white').save(blank)
        inputs += [dict(id='synthetic_blank', image=str(blank), png_sha256=sha256(blank))] + old['cases']
        sources = {str(Path(__file__)): sha256(__file__), str(SOURCE/'protocol.json'): sha256(SOURCE/'protocol.json'),
                   str(config_path): sha256(config_path), config['model_path']: sha256(config['model_path'])}
        for p in package.rglob('*.py'):
            sources[str(p)] = sha256(p)
        deps = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
        protocol = dict(sources=sources, inputs=inputs, detector_config=config, dependencies=deps,
                        real_detection_limit=12, synthetic_detection_limit=2, recognition_calls=0,
                        dataset_answers=0, scope='自动文字框能力检查，不是证据相关性或算法精度')
        pp = run/'protocol.json'
        if pp.exists():
            if read_json(pp) != protocol:
                raise ValueError('恢复协议或依赖改变')
        else:
            durable(pp, protocol)
        fingerprint = sha256(pp)
        detector = None
        # 步骤2：每张图只调用检测网络一次，结果为空也是有效记录，不降低阈值重试。
        for item in inputs:
            if sha256(item['image']) != item['png_sha256']:
                raise ValueError('输入图像变化')
            path = run/'calls'/(item['id']+'.json')
            if path.exists():
                result = read_json(path)
                if result['status'] != 'completed' or result['fingerprint'] != fingerprint:
                    raise ValueError('未知调用状态，不重试')
            else:
                if detector is None:
                    detector = TextDetector(config)
                image = cv2.imread(item['image'])
                if image is None:
                    raise ValueError('图像读取失败')
                durable(path, dict(status='started', fingerprint=fingerprint, utc=utc()))
                start = time.perf_counter()
                boxes, elapsed = detector(image)
                result = dict(status='completed', fingerprint=fingerprint, id=item['id'],
                              boxes=[] if boxes is None else np.asarray(boxes).tolist(),
                              seconds=time.perf_counter()-start, detector_seconds=elapsed,
                              image_shape=list(image.shape))
                durable(path, result)
            if item['id']=='synthetic_text' and not result['boxes']:
                raise ValueError('合成文字未检测到')
            if item['id']=='synthetic_blank' and result['boxes']:
                raise ValueError('合成纯白出现文字框')
            # 步骤3：图板只展示自动框，不人工删改误检或补画漏检框。
            output = run/'views'/(item['id']+'.png')
            if not output.exists():
                output.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(item['image']) as im:
                    im = im.convert('RGB')
                    draw = ImageDraw.Draw(im)
                    for box in result['boxes']:
                        points = [tuple(p) for p in box]
                        draw.line(points+[points[0]], fill='red', width=3)
                    im.save(output)
            print(item['id'], len(result['boxes']), flush=True)
        durable(run/'summary.json', dict(real_detection_calls=12, synthetic_detection_calls=2,
            recognition_calls=0, dataset_answers=0,
            records=[read_json(run/'calls'/(item['id']+'.json')) for item in inputs]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run)
