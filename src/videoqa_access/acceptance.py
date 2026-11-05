"""12视频无模型原片验收与实际HTTP链路校准；不借用旧模型输出。"""
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request,urlopen

from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,utc
from .trial import prepare_manifest,build_preview,freeze
from .transport import AccessService,AccessClient,start_service,PROFILES,rgb_hash


def engineering_rows(manifest):
    """接口：Video-MME每时长首2；MLVU前6种任务各首1，不依赖标签。"""
    rows=manifest['rows'];out=[]
    for group in ('short','medium','long'):
        out.extend([r for r in rows if r['dataset']=='videomme' and r['stratum']==group][:2])
    seen=set()
    for row in rows:
        if row['dataset']=='mlvu' and row['task_type'] not in seen:
            seen.add(row['task_type']);out.append(row)
            if len(seen)==6:break
    if len(out)!=12 or len({r['video_id'] for r in out})!=12:raise ValueError('Engineering panel differs')
    return out


def calibrate(root,url):
    """接口：以已知4 MiB实际响应校准两限制链路，不把解码或模型时间混入。"""
    root=Path(root);data=b'controlled-video-access-calibration'*((4*1024**2)//34+1);data=data[:4*1024**2]
    key=hashlib.sha256(data).hexdigest();p=root/'transport_payloads'/(key+'.bin');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    records=[]
    for profile,(rate,latency) in PROFILES.items():
        for repetition in range(2):
            request=json.dumps(dict(payload_sha256=key,link_profile=profile)).encode();began=time.perf_counter()
            with urlopen(Request(url+'/replay',request,{'Content-Type':'application/json'}),timeout=120) as response:out=response.read()
            seconds=time.perf_counter()-began
            if out!=data:raise ValueError('Calibration bytes changed')
            ideal=latency+len(data)*8/rate if rate else 0.
            if rate and not .95*ideal<=seconds<=ideal*1.20+.10:raise ValueError('Link calibration outside tolerance')
            records.append(dict(profile=profile,repetition=repetition,payload_bytes=len(data),seconds=seconds,ideal_seconds=ideal))
    freeze(root/'link_calibration.json',dict(status='passed',records=records,new_model_calls=0));return records


def engineering(root):
    """接口：先真实验收12视频；恢复只验证完成导出，不重解码或重发模型。"""
    from videoqa_methods.region_context import read_exact_frames
    root=Path(root);m=prepare_manifest(root);panel=engineering_rows(m);dest=root/'engineering_acceptance.json'
    if dest.exists():
        old=read_json(dest)
        for p,h in old['exports'].items():
            if sha256(root/p)!=h:raise ValueError('Engineering export changed')
        return old
    records=[];indices={}
    # 步骤1逐视频完成真实预览及完整索引核验；阶段心跳不意味着任务通过。
    for ordinal,row in enumerate(panel):
        vid=row['video_id'];index=build_preview((str(root),vid,m['indices'][vid]));indices[vid]=index
        durable(root/'engineering_status.json',dict(status='indexing',completed=ordinal+1,total=12,active_question=row['question_id'],utc=utc()))
    service=AccessService(root,indices);server,thread=start_service(service);url='http://127.0.0.1:'+str(server.server_port)
    try:
        for row in panel:
            vid=row['video_id'];index=indices[vid];c=index['coarse'];frames=[c[i] for i in sorted(set([0,len(c)//2,len(c)-1]))]
            client=AccessClient(url,vid,'engineering/'+row['question_id']);images=client.fetch(frames)
            direct=read_exact_frames(index['video'],frames)
            try:
                actual=[rgb_hash(im) for im in images];expected=[rgb_hash(im) for im in direct]
                if actual!=expected:raise ValueError('Service/direct RGB differs')
                first=len(client.requests);cached=client.fetch(frames)
                for im in cached:im.close()
                if len(client.requests)!=first:raise ValueError('Per-method cache did not close')
                records.append(dict(question_id=row['question_id'],dataset=row['dataset'],video_id=vid,
                    requested_sources=[(f['source_pts'],f['source_frame_index']) for f in frames],rgb_sha256=actual,requests=client.requests))
            finally:
                for im in images+direct:im.close()
                client.close()
        calibrate(root,url)
    finally:server.shutdown();server.server_close();thread.join()
    exports={str(p.relative_to(root)):sha256(p) for p in [root/'manifest.json',root/'labels.json',root/'link_calibration.json',*[root/'indices'/(r['video_id']+'.json') for r in panel]]}
    value=dict(status='passed',videos=12,source_frames=sum(len(r['requested_sources']) for r in records),records=records,exports=exports,new_blip_calls=0,new_visual_calls=0,new_auxiliary_generations=0,new_final_qa=0)
    freeze(dest,value);durable(root/'engineering_status.json',dict(status='completed',videos=12,utc=utc()));return value


def representation_acceptance(root):
    """接口：12真实视频验证细预览等于原图原变换，且预览缓存不能充当原图缓存。"""
    from videoqa_access.transport import preview_bytes
    import io
    from PIL import Image
    root=Path(root);dest=root/'representation_acceptance.json'
    if dest.exists():
        value=read_json(dest)
        if value['protocol_sha256']!=sha256(root/'protocol.json'):raise ValueError('Representation acceptance fingerprint differs')
        return value
    panel=engineering_rows(read_json(root/'manifest.json'));indices={r['video_id']:read_json(root/'indices'/(r['video_id']+'.json')) for r in panel}
    service=AccessService(root,indices);server,thread=start_service(service);url='http://127.0.0.1:'+str(server.server_port);records=[]
    try:
        for row in panel:
            index=indices[row['video_id']];f=index['coarse'][len(index['coarse'])//2]
            client=AccessClient(url,row['video_id'],'representations/'+row['question_id'])
            low=client.fetch([f],representation='preview')
            if client.cache:raise ValueError('Preview incorrectly populated original cache')
            full=client.fetch([f])
            try:
                with Image.open(io.BytesIO(preview_bytes(full[0]))) as expected:
                    if rgb_hash(low[0])!=rgb_hash(expected) or rgb_hash(low[0])!=f['preview_rgb_sha256']:
                        raise ValueError('On-demand JPEG differs from original preview transform')
                records.append(dict(video_id=row['video_id'],question_id=row['question_id'],source_pts=f['source_pts'],
                    preview_rgb_sha256=rgb_hash(low[0]),original_rgb_sha256=rgb_hash(full[0]),requests=client.requests))
            finally:
                for im in low+full:im.close()
                client.close()
    finally:server.shutdown();server.server_close();thread.join()
    value=dict(status='passed',videos=12,protocol_sha256=sha256(root/'protocol.json'),records=records,
        equivalent_selector_pixels=True,original_and_preview_cache_isolated=True,new_model_calls=0)
    freeze(dest,value);return value
