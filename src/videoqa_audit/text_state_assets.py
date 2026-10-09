"""OCR辅助资产：独立环境RPC、精确PTS有界解码、单次CPU视觉账本。"""
from pathlib import Path
import subprocess,json,struct,hashlib,time,os
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import durable,exclusive,digest,utc
from videoqa_audit.framework_contraction import BASE

OCRPY=ROOT/'outputs/diagnostics/scene_text_env_20261004/bin/python'
CHILD=ROOT/'scripts/text_state_ocr_worker.py'


def ocr_identity():
    """接口：冻结已有OCR包/两权重及依赖，不升级原QA或OCR环境。"""
    code='''import rapidocr_onnxruntime,yaml,pathlib,hashlib,importlib.metadata,json
p=pathlib.Path(rapidocr_onnxruntime.__file__).parent
c=yaml.safe_load((p/'config.yaml').read_text())
paths=[p/'config.yaml',p/c['Det']['model_path'],p/c['Rec']['model_path']]+list(p.rglob('*.py'))
print(json.dumps(dict(files={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},config={k:c[k] for k in ('Det','Rec')},dependencies={n:importlib.metadata.version(n) for n in ('rapidocr-onnxruntime','onnxruntime','numpy','opencv-python')})))'''
    return json.loads(subprocess.check_output([str(OCRPY),'-c',code],text=True))


class OCRProcess:
    """接口：一个长期CPU识别进程，一次仅传一帧，stdout返回完整原始识别结果。"""
    def __init__(self,log):
        self.log=Path(log);self.log.parent.mkdir(parents=True,exist_ok=True)
        self.handle=self.log.open('ab')
        self.process=subprocess.Popen([str(OCRPY),str(CHILD)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.handle,cwd=ROOT)

    def infer(self,rgb):
        """接口：连续uint8 RGB→自动框/文本/批次；没有任何题目或答案字段。"""
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Invalid OCR RGB')
        raw=np.ascontiguousarray(rgb).tobytes();h=hashlib.sha256(raw).hexdigest()
        header=json.dumps(dict(shape=list(rgb.shape),bytes=len(raw),rgb_sha256=h)).encode()
        # 步骤1：完整发送形状与无损像素，识别结果须绑定同一RGB摘要。
        self.process.stdin.write(struct.pack('!I',len(header)));self.process.stdin.write(header);self.process.stdin.write(raw);self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('OCR process returned no complete output; inspect '+str(self.log))
        result=json.loads(line)
        if result['rgb_sha256']!=h:raise ValueError('OCR returned another image identity')
        return result

    def close(self):
        """接口：关闭输入后正常退出；只终止本任务OCR子进程，不影响其他计算。"""
        self.process.stdin.close()
        try:code=self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:self.process.terminate();self.process.wait();code=-1
        self.process.stdout.close();self.handle.close()
        if code!=0:raise RuntimeError('OCR subprocess failed; log '+str(self.log))


def extract_question(job):
    """接口：一题M32冻结粗细池→逐源OCR缓存；K8通过原全局行映射复用。

    全视频顺序解码但只转换目标RGB，源帧号/PTS均核对。没有全候选PNG或RGB堆积。
    """
    import av
    from videoqa_methods.text_state_redundancy import normalize_text_state
    root,q=Path(job[0]),job[1];pp=BASE/'pools'/(q+'.json');pool=read_json(pp)
    spec=read_json(root/'protocol.json');fingerprint=sha256(root/'protocol.json');dest=root/'ocr_banks'/(q+'.json')
    parent_check=read_json(BASE/'selection_checkpoints'/(q+'.json'))
    if sha256(pp)!=parent_check['files'][str(pp.relative_to(BASE))]:raise ValueError('Frozen M32 pool changed')
    video=pool['video'];path=Path(video['path']);stat=path.stat()
    video_id=spec['question_video_ids'][q]
    old=read_json(BASE/'protocol.json');expected=old['video_hashes'][video_id]
    check=dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=expected)
    check_path=root/'video_checks'/(video_id+'.json')
    if check_path.exists():
        if read_json(check_path)!=check:raise ValueError('Source video changed')
    else:
        if sha256(path)!=expected:raise ValueError('Video hash differs')
        durable(check_path,check)
    if dest.exists():
        saved=read_json(dest)
        if saved['pool_sha256']!=sha256(pp) or saved['protocol_sha256']!=fingerprint:raise ValueError('OCR bank recovery differs')
        for p,h in saved['calls'].items():
            if sha256(p)!=h:raise ValueError('OCR result changed')
        return saved
    ordered=sorted(pool['candidates'],key=lambda f:f['source_pts']);lookup={f['source_pts']:f for f in ordered}
    cache=root/'frame_calls'/video_id;cache.mkdir(parents=True,exist_ok=True);done={};process=None;began=time.perf_counter()
    # 步骤1：开始标记的未知前向禁止重发，完成缓存按真实源身份复用。
    for frame in ordered:
        p=cache/(str(frame['source_pts'])+'.json')
        if p.exists():
            r=read_json(p)
            if r['status']!='completed' or r['protocol_sha256']!=fingerprint or r['source_frame_index']!=frame['source_frame_index']:raise ValueError('Uncertain OCR call; repetition forbidden')
            done[frame['source_pts']]=r
    try:
        if len(done)<len(ordered):
            process=OCRProcess(root/'logs'/('ocr_'+q+'.log'))
            # 步骤2：从文件起点逐帧计数，绝不按平均FPS反推；一帧只保留到返回结果。
            with av.open(str(path)) as container:
                stream=container.streams.video[0];stream.codec_context.thread_count=2
                from fractions import Fraction
                if Fraction(stream.time_base)!=Fraction(video['time_base']) or (stream.start_time or 0)!=video['start_pts']:raise ValueError('Video clock differs')
                for index,frame in enumerate(container.decode(stream)):
                    pts=frame.pts
                    if pts not in lookup or pts in done:continue
                    wanted=lookup[pts]
                    if index!=wanted['source_frame_index']:raise ValueError('Full decoder source index differs')
                    rgb=frame.to_ndarray(format='rgb24');p=cache/(str(pts)+'.json')
                    with exclusive(p.with_suffix('.lock')):
                        if p.exists():raise ValueError('Concurrent OCR source appeared unexpectedly')
                        durable(p,dict(status='started',protocol_sha256=fingerprint,source_pts=pts,source_frame_index=index,utc=utc()))
                        result=process.infer(rgb)
                        value=dict(status='completed',protocol_sha256=fingerprint,source_pts=pts,source_frame_index=index,
                            rgb_shape=list(rgb.shape),ocr=result,tokens=normalize_text_state(result['regions']),utc=utc())
                        durable(p,value);done[pts]=value
                    del rgb
                    if len(done)%20==0:durable(root/'ocr_status'/(q+'.json'),dict(completed=len(done),total=len(ordered),pid=os.getpid(),utc=utc()))
                    if len(done)==len(ordered):break
        if len(done)!=len(ordered):raise ValueError('Frozen OCR source frame missing')
    finally:
        if process is not None:process.close()
    # 步骤3：银行只保存紧凑词集与源身份，完整低分输出/坐标在逐源账本中保留。
    items=[done[f['source_pts']] for f in pool['candidates']]
    saved=dict(question_id=q,pool_sha256=sha256(pp),protocol_sha256=fingerprint,
        states=[r['tokens'] for r in items],source_keys=[[f['source_frame_index'],f['source_pts']] for f in pool['candidates']],
        calls={str(cache/(str(f['source_pts'])+'.json')):sha256(cache/(str(f['source_pts'])+'.json')) for f in pool['candidates']},
        seconds=time.perf_counter()-began,frames=len(items),text_frames=sum(bool(r['tokens']) for r in items))
    durable(dest,saved);return saved
