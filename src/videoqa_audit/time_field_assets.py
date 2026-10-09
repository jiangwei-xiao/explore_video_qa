"""显式字段只读资产：仅初始粗候选，复用完成OCR；不重算BLIP或视频视觉特征。"""
from pathlib import Path
import os,time
from fractions import Fraction
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.framework_contraction import BASE
from videoqa_audit.text_state_assets import OCRProcess


def extract_coarse_question(job):
    """接口：冻结原视频与完整粗PTS→一次OCR账本；有界逐源解码、未知返回不重发。"""
    import av
    root,q=Path(job[0]),job[1]; spec=read_json(root/'protocol.json'); fp=sha256(root/'protocol.json')
    pool_path=BASE/'pools'/(q+'.json'); pool=read_json(pool_path)
    if sha256(pool_path)!=spec['pool_sources'][str(pool_path)]: raise ValueError('Frozen coarse pool changed')
    coarse=pool['candidates'][:pool['initial_count']]; video=pool['video']; vid=spec['question_video_ids'][q]
    dest=root/'ocr_banks'/(q+'.json'); cache=root/'frame_calls'/vid; done={}; engine=None; began=time.perf_counter()
    # 步骤1：历史完成资产与新的前向严格区分，不能重发未知OCR或改写旧记录。
    if dest.exists():
        bank=read_json(dest)
        if bank['protocol_sha256']!=fp or bank['source_keys']!=[[c['source_frame_index'],c['source_pts']] for c in coarse]:
            raise ValueError('Recovered coarse bank changed')
        for p,h in bank['calls'].items():
            if sha256(p)!=h: raise ValueError('Recovered OCR return changed')
        return bank
    path=Path(video['path']); st=path.stat(); identity=dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=spec['video_hashes'][vid])
    check=root/'video_checks'/(vid+'.json')
    if check.exists():
        if read_json(check)!=identity: raise ValueError('Video identity changed')
    else:
        if sha256(path)!=identity['sha256']: raise ValueError('Video bytes changed')
        durable(check,identity)
    for c in coarse:
        out=cache/(str(c['source_pts'])+'.json')
        if out.exists():
            r=read_json(out)
            if r['status']!='completed' or r['protocol_sha256']!=fp or r['source_frame_index']!=c['source_frame_index']:
                raise ValueError('Uncertain OCR; repeat forbidden')
            done[c['source_pts']]=r
        elif str(c['source_pts']) in spec['imports'].get(vid,{}):
            info=spec['imports'][vid][str(c['source_pts'])]; old=read_json(info['path'])
            if sha256(info['path'])!=info['sha256'] or old['status']!='completed' or old['source_frame_index']!=c['source_frame_index']:
                raise ValueError('Imported OCR source changed')
            value=dict(status='completed',protocol_sha256=fp,source_pts=c['source_pts'],source_frame_index=c['source_frame_index'],
                ocr=old['ocr'],rgb_shape=old['rgb_shape'],reused=True,import_source=info,utc=utc())
            durable(out,value); done[c['source_pts']]=value
    lookup={c['source_pts']:c for c in coarse}
    try:
        if len(done)<len(coarse):
            engine=OCRProcess(root/'logs'/('ocr_'+q+'.log'))
            # 步骤2：文件起点的真实源编号与PTS逐项核验；不保存全候选RGB/PNG。
            with av.open(str(path)) as container:
                stream=container.streams.video[0]; stream.codec_context.thread_count=2
                if Fraction(stream.time_base)!=Fraction(video['time_base']) or (stream.start_time or 0)!=video['start_pts']:
                    raise ValueError('Video clock changed')
                for index,image in enumerate(container.decode(stream)):
                    pts=image.pts
                    if pts not in lookup or pts in done: continue
                    if index!=lookup[pts]['source_frame_index']: raise ValueError('Coarse source index differs')
                    rgb=image.to_ndarray(format='rgb24'); out=cache/(str(pts)+'.json')
                    with exclusive(out.with_suffix('.lock')):
                        if out.exists(): raise ValueError('Concurrent source call appeared')
                        durable(out,dict(status='started',protocol_sha256=fp,source_pts=pts,source_frame_index=index,utc=utc()))
                        result=engine.infer(rgb)
                        value=dict(status='completed',protocol_sha256=fp,source_pts=pts,source_frame_index=index,
                            ocr=result,rgb_shape=list(rgb.shape),reused=False,utc=utc())
                        durable(out,value); done[pts]=value
                    del rgb
                    if len(done)%20==0: durable(root/'progress'/(q+'.json'),dict(completed=len(done),total=len(coarse),pid=os.getpid(),utc=utc()))
                    if len(done)==len(coarse): break
        if len(done)!=len(coarse): raise ValueError('A coarse source was not decoded')
    finally:
        if engine is not None: engine.close()
    # 步骤3：保存原始返回来源，不把缓存成本算作新模型前向。
    value=dict(question_id=q,protocol_sha256=fp,source_keys=[[c['source_frame_index'],c['source_pts']] for c in coarse],
        calls={str(cache/(str(c['source_pts'])+'.json')):sha256(cache/(str(c['source_pts'])+'.json')) for c in coarse},
        frames=len(coarse),new_frames=sum(not r['reused'] for r in done.values()),seconds=time.perf_counter()-began)
    durable(dest,value); return value
