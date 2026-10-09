"""隔离OCR CPU环境的持久二进制输入接口，不加载QA、BLIP或视频解码库。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import sys,json,struct,hashlib,time
from pathlib import Path


def read_exact(stream,count):
    """接口：读取一帧完整字节；部分消息终止，不猜测边界或重复前向。"""
    data=bytearray()
    while len(data)<count:
        part=stream.read(count-len(data))
        if not part:raise EOFError('Partial OCR frame message')
        data.extend(part)
    return bytes(data)


def main():
    """接口：完整RGB→自动检测/CTC原始结果；两种环境通过管道解耦且不升级依赖。"""
    import cv2,numpy as np,yaml,rapidocr_onnxruntime
    from rapidocr_onnxruntime import RapidOCR
    from rapidocr_onnxruntime.ch_ppocr_det.text_detect import TextDetector
    from rapidocr_onnxruntime.ch_ppocr_rec.text_recognize import TextRecognizer
    cv2.setNumThreads(2);package=Path(rapidocr_onnxruntime.__file__).parent
    cfg=yaml.safe_load((package/'config.yaml').read_text())
    for key in ('Det','Rec'):
        cfg[key]['model_path']=str(package/cfg[key]['model_path']);cfg[key].update(intra_op_num_threads=2,inter_op_num_threads=2,use_cuda=False,use_dml=False)
    detector,recognizer=TextDetector(cfg['Det']),TextRecognizer(cfg['Rec'])
    # 步骤1：权重和包由父协议哈希固定，仅CPU执行、不接收问题/选项/答案。
    while True:
        header=sys.stdin.buffer.read(4)
        if not header:break
        if len(header)!=4:raise EOFError('Incomplete OCR header')
        meta=json.loads(read_exact(sys.stdin.buffer,struct.unpack('!I',header)[0]));shape=meta['shape']
        raw=read_exact(sys.stdin.buffer,meta['bytes'])
        if hashlib.sha256(raw).hexdigest()!=meta['rgb_sha256']:raise ValueError('OCR transport pixels differ')
        if len(shape)!=3 or shape[2]!=3 or int(np.prod(shape))!=len(raw):raise ValueError('OCR input shape differs')
        image=np.ascontiguousarray(np.frombuffer(raw,dtype=np.uint8).reshape(shape)[:,:,::-1]);began=time.perf_counter()
        # 步骤2：沿用默认检测与透视裁剪，不手工指定答题区域或补文字框。
        boxes,det_seconds=detector(image);boxes=[] if boxes is None else np.asarray(boxes).tolist()
        crops=RapidOCR.get_crop_img_list(None,image,np.asarray(boxes,dtype=np.float32));batches=[]
        original=recognizer.session
        def observe(batch):
            batches.append(list(batch.shape));return original(batch)
        recognizer.session=observe
        try:words,rec_seconds=recognizer(crops) if crops else ([],0.0)
        finally:recognizer.session=original
        if len(words)!=len(boxes):raise ValueError('OCR box and text count differs')
        value=dict(rgb_sha256=meta['rgb_sha256'],regions=[dict(box=b,text=str(w[0]),score=float(w[1])) for b,w in zip(boxes,words)],
            detection_forwards=1,recognition_batches=batches,seconds=time.perf_counter()-began,det_seconds=det_seconds,rec_seconds=rec_seconds)
        # 步骤3：父进程收到完整JSON后才能记完成，stdout只用于协议消息。
        print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()
