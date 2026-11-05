"""独立原片服务、真实解码/字节计账与受控链路重放。"""
import hashlib
import io
import json
from pathlib import Path
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

PROFILES = {'local': (0, 0), 'fast': (100_000_000, .010), 'limited': (20_000_000, .050)}


def rgb_hash(image):
    """接口：返回实际RGB像素指纹，不把文件压缩字节当作模型输入身份。"""
    return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()


def preview_bytes(image):
    """接口：完整画面等比例缩至长边320、JPEG80，禁止放大或裁剪。"""
    from PIL import Image
    # 步骤1缩小完整RGB；步骤2固定编码，所有检索前端消费解码后的相同像素。
    value=image.convert('RGB'); value.thumbnail((320, 320), Image.Resampling.LANCZOS)
    output=io.BytesIO(); value.save(output, format='JPEG', quality=80, subsampling=2)
    value.close(); return output.getvalue()


class CountingReader:
    """接口：统计libavformat实际读取的文件字节和seek；不声称统计物理磁盘读。"""
    def __init__(self, path):
        self.stream=open(path, 'rb'); self.bytes_read=0; self.read_calls=0; self.seek_calls=0
        self.name=str(path)
    def read(self, size=-1):
        data=self.stream.read(size); self.bytes_read+=len(data); self.read_calls+=1; return data
    def seek(self, offset, whence=0):
        self.seek_calls+=1; return self.stream.seek(offset, whence)
    def tell(self):return self.stream.tell()
    def readable(self):return True
    def seekable(self):return True
    def close(self):self.stream.close()


def pack_frames(metadata, encoded):
    """接口：用显式长度封装JSON与无损PNG，保持帧顺序且能精确计量响应体。"""
    header=json.dumps(dict(metadata, image_lengths=[len(x) for x in encoded]), sort_keys=True).encode()
    return struct.pack('!Q', len(header))+header+b''.join(encoded)


def unpack_frames(payload):
    """接口：拒绝截断、多余字节或源身份/像素错配，不自动修复媒体。"""
    from PIL import Image
    if len(payload)<8:raise ValueError('Truncated transport header')
    size=struct.unpack('!Q', payload[:8])[0]
    metadata=json.loads(payload[8:8+size]); offset=8+size; images=[]
    try:
        if len(metadata['frames'])!=len(metadata['image_lengths']):raise ValueError('Frame length count')
        for record,n in zip(metadata['frames'],metadata['image_lengths']):
            if n<1 or offset+n>len(payload):raise ValueError('Truncated image')
            with Image.open(io.BytesIO(payload[offset:offset+n])) as im:image=im.convert('RGB')
            offset+=n
            if rgb_hash(image)!=record['rgb_sha256']:image.close();raise ValueError('Transport RGB differs')
            images.append(image)
        if offset!=len(payload):raise ValueError('Trailing transport data')
        return metadata,images
    except BaseException:
        for im in images:im.close()
        raise


def decode_exact(index, frames):
    """接口：按共同关键帧组批量解码；返回无损画面及实际文件读取/解码成本。"""
    import av
    import numpy as np
    # 步骤1验证源编号与完整PTS；索引只含时钟/关键帧元数据，不含旧视觉特征。
    pts=np.load(index['pts_path'], allow_pickle=False)
    wanted={int(f['source_pts']): f for f in frames}
    if len(wanted)!=len(frames):raise ValueError('Duplicate requested source')
    for f in frames:
        if not 0<=f['source_frame_index']<len(pts) or int(pts[f['source_frame_index']])!=f['source_pts']:
            raise ValueError('Requested source index/PTS mismatch')
    key_pts=np.asarray(index['keyframe_pts'], dtype=np.int64); groups={}
    for p in sorted(wanted):
        j=max(0,int(np.searchsorted(key_pts,p,side='right'))-1)
        groups.setdefault(int(key_pts[j]), []).append(p)
    # 步骤2同一关键帧组仅seek一次；额外解码帧只计成本，不进入候选特征。
    reader=CountingReader(index['video']['path']); images={}; decoded=0; began=time.perf_counter()
    try:
        with av.open(reader) as container:
            stream=container.streams.video[0];stream.codec_context.thread_count=2
            for anchor, targets in groups.items():
                container.seek(anchor,stream=stream,backward=True)
                remaining=set(targets); end=targets[-1]
                for frame in container.decode(stream):
                    decoded+=1
                    if frame.pts in remaining:
                        images[frame.pts]=frame.to_image(); remaining.remove(frame.pts)
                    if frame.pts is not None and frame.pts>=end:break
                if remaining:raise ValueError('Exact source frame missing')
        timing=dict(decode_seconds=time.perf_counter()-began, decoded_frames=decoded,
                    file_read_bytes=reader.bytes_read, file_read_calls=reader.read_calls,
                    file_seek_calls=reader.seek_calls, keyframe_groups=len(groups))
        return [images[int(f['source_pts'])] for f in frames], timing
    except BaseException:
        for im in images.values():im.close()
        raise
    finally:reader.close()


class AccessService:
    """接口：只允许冻结索引中的视频，生成可审计响应；客户端不接触原片路径。"""
    def __init__(self, root, indices):
        self.root=Path(root); self.indices=indices; self.lock=threading.Lock()
        self.payload_dir=self.root/'transport_payloads';self.payload_dir.mkdir(parents=True,exist_ok=True)

    def fetch(self, request):
        """步骤1核验来源；步骤2真实解码/PNG编码；步骤3保存响应供链路重放。"""
        from videoqa_full.state import durable, digest
        if request['video_id'] not in self.indices:raise ValueError('Unknown video')
        index=self.indices[request['video_id']]; frames=request['frames'];representation=request.get('representation','original')
        if representation not in ('original','preview'):raise ValueError('Unknown representation')
        if not 1<=len(frames)<=4096:raise ValueError('Unbounded source request')
        images,timing=decode_exact(index, frames);encoded=[];records=[];began=time.perf_counter()
        try:
            for f,image in zip(frames,images):
                if representation=='preview':
                    from PIL import Image
                    data=preview_bytes(image)
                    with Image.open(io.BytesIO(data)) as decoded:h=rgb_hash(decoded)
                else:
                    output=io.BytesIO();image.save(output,format='PNG');data=output.getvalue();h=rgb_hash(image)
                encoded.append(data);records.append(dict(f,rgb_sha256=h))
        finally:
            for image in images:image.close()
        timing['image_encode_seconds']=time.perf_counter()-began
        metadata=dict(video_id=request['video_id'],frames=records,timing=timing,representation=representation)
        payload=pack_frames(metadata,encoded);key=hashlib.sha256(payload).hexdigest()
        path=self.payload_dir/(key+'.bin')
        with self.lock:
            if not path.exists():
                tmp=path.with_suffix('.part');tmp.write_bytes(payload);tmp.replace(path)
            durable(self.root/'server_requests'/(digest(request)+'.json'),dict(request=request,
                payload_sha256=key,payload_bytes=len(payload),**timing))
        return payload

    def replay(self, key):
        """接口：只重放本轮已保存响应，不重新解码或把缓存耗时说成原片成本。"""
        if len(key)!=64 or any(x not in '0123456789abcdef' for x in key):raise ValueError('Invalid payload ID')
        data=(self.payload_dir/(key+'.bin')).read_bytes()
        if hashlib.sha256(data).hexdigest()!=key:raise ValueError('Replay payload changed')
        return data


def start_service(service):
    """接口：仅绑定回环地址；实际响应体按给定带宽整形，解码开销另外记录。"""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                profile=request.pop('link_profile','local'); rate,delay=PROFILES[profile]
                data=service.fetch(request) if self.path=='/frames' else service.replay(request['payload_sha256']) if self.path=='/replay' else None
                if data is None:raise ValueError('Unknown endpoint')
                self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers()
                # 步骤1固定请求延迟；步骤2无突发分块发送，多个并发请求共享同一模拟链路。
                with service.lock:
                    start=time.perf_counter()
                    if delay:time.sleep(delay)
                    sent=0
                    for offset in range(0,len(data),65536):
                        block=data[offset:offset+65536]
                        if rate:
                            due=start+delay+(sent+len(block))*8/rate
                            wait=due-time.perf_counter()
                            if wait>0:time.sleep(wait)
                        self.wfile.write(block);self.wfile.flush();sent+=len(block)
            except Exception as exc:
                if not self.wfile.closed:
                    try:self.send_error(500,str(exc))
                    except (BrokenPipeError,ConnectionResetError):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    return server,thread


class AccessClient:
    """接口：每题每方法拥有独立原图缓存；没有服务器返回即没有该帧像素。"""
    def __init__(self, url, video_id, scope):
        self.url=url;self.video_id=video_id;self.scope=scope;self.cache={};self.preview_cache={};self.requests=[]

    def fetch(self, frames, representation='original'):
        """步骤1仅请求缺失源；步骤2校验原图身份；步骤3保存实际网络与客户端解包耗时。"""
        from videoqa_full.state import digest
        if representation not in ('original','preview'):raise ValueError('Unknown representation')
        cache=self.cache if representation=='original' else self.preview_cache
        missing=[f for f in frames if f['source_pts'] not in cache]
        if missing:
            body=json.dumps(dict(video_id=self.video_id,frames=missing,scope=self.scope,representation=representation)).encode()
            began=time.perf_counter()
            with urlopen(Request(self.url+'/frames',body,{'Content-Type':'application/json'}),timeout=600) as response:
                payload=response.read()
            transport=time.perf_counter()-began; began=time.perf_counter();meta,images=unpack_frames(payload)
            if meta['video_id']!=self.video_id or meta['representation']!=representation or [(f['source_pts'],f['source_frame_index']) for f in meta['frames']]!=[(f['source_pts'],f['source_frame_index']) for f in missing]:
                for im in images:im.close()
                raise ValueError('Service returned different sources')
            for f,im in zip(missing,images):cache[f['source_pts']]=im
            self.requests.append(dict(payload_sha256=hashlib.sha256(payload).hexdigest(),payload_bytes=len(payload),
                representation=representation,
                network_and_server_seconds=transport,client_unpack_seconds=time.perf_counter()-began,
                requested_sources=[(f['source_pts'],f['source_frame_index']) for f in missing],**meta['timing']))
        return [cache[f['source_pts']].copy() for f in frames]

    def close(self):
        """接口：退出只释放本题本方法原图，禁止把缓存特权传给下一种方法。"""
        for im in self.cache.values():im.close()
        for im in self.preview_cache.values():im.close()
        self.cache.clear()
        self.preview_cache.clear()
