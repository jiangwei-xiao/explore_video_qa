"""独立网络传输修订：4MiB可恢复子段组成原64MiB接收块，不改变任何实验输入。"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import http.client
import json
import os
import shlex
import subprocess
import time
import urllib.request

from videoqa_set_value import longvideo_assets as a

SEGMENT_BYTES=4*2**20
HTTP_WORKERS=64
ACTIVE_CHUNKS=4
TEMPORARY_LIMIT=8*2**30


def segment_specs(entry,chunk_offset):
    """接口：细分仅作用于HTTP，不改变原接收块偏移、长度或官方LFS身份。"""
    length=min(a.CHUNK_BYTES,entry['bytes']-chunk_offset)
    if chunk_offset<0 or chunk_offset%a.CHUNK_BYTES or length<=0:raise ValueError('Invalid original chunk')
    return [dict(offset=p,bytes=min(SEGMENT_BYTES,chunk_offset+length-p))
        for p in range(chunk_offset,chunk_offset+length,SEGMENT_BYTES)]


def cache_prefix(state_dir,entry,chunk_offset):
    """接口：本机子段命名绑定固定源与真实字节偏移，不接受归档内路径。"""
    a.source_entry(entry['name'])
    return Path(state_dir)/'segments'/entry['name']/str(chunk_offset)


def fetch_segment(entry,spec,directory):
    """接口：已完成4MiB子段校验复用；失败仅重读本段，不丢弃同块其他完成段。"""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    offset,length=spec['offset'],spec['bytes'];path=directory/(str(offset)+'.bin');cp=path.with_suffix('.json')
    identity=dict(archive=entry['name'],source_sha256=entry['sha256'],offset=offset,bytes=length)
    if cp.exists():
        record=json.loads(cp.read_text())
        if record['identity']!=identity or not path.exists() or path.stat().st_size!=length or a.file_sha(path)!=record['sha256']:
            raise ValueError('Completed HTTP segment changed')
        return record
    end=offset+length-1
    # 步骤1：每段独立HTTP206/总长度验收；不缓存或打印重定向签名。
    for attempt in range(4):
        try:
            request=urllib.request.Request(entry['url']+f'?download=true&segment={offset}-{end}',
                headers={'Range':f'bytes={offset}-{end}'})
            temp=path.with_suffix('.part');h=hashlib.sha256();count=0
            with urllib.request.urlopen(request,timeout=120) as response,temp.open('wb') as out:
                if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{end}/{entry["bytes"]}':
                    raise ValueError('Segment response range differs')
                while True:
                    block=response.read(min(2**20,length+1-count))
                    if not block:break
                    count+=len(block)
                    if count>length:raise ValueError('Oversized HTTP segment')
                    out.write(block);h.update(block)
                out.flush();os.fsync(out.fileno())
            if count!=length:raise ValueError('Truncated HTTP segment')
            temp.replace(path)
            record=dict(identity=identity,sha256=h.hexdigest(),path=str(path),completed_unix=time.time())
            a.atomic_json(cp,record);return record
        except (OSError,ValueError,http.client.HTTPException):
            if attempt==3:raise
            time.sleep(attempt+1)


def assemble_chunk(entry,chunk_offset,directory):
    """接口：顺序拼接已验收子段，计算原64MiB块SHA；禁止静默跳过缺段。"""
    directory=Path(directory);specs=segment_specs(entry,chunk_offset);dest=directory/'assembled.bin'
    temp=directory/'assembled.part';h=hashlib.sha256();count=0
    # 步骤2：重新校验每段身份/字节后组装，输出兼容原接收API。
    with temp.open('wb') as out:
        for spec in specs:
            path=directory/(str(spec['offset'])+'.bin');record=json.loads(path.with_suffix('.json').read_text())
            expected=dict(archive=entry['name'],source_sha256=entry['sha256'],**spec)
            if record['identity']!=expected or path.stat().st_size!=spec['bytes'] or a.file_sha(path)!=record['sha256']:
                raise ValueError('Cannot assemble unverified/mutated segment')
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(2**20),b''):out.write(block);h.update(block);count+=len(block)
        out.flush();os.fsync(out.fileno())
    if count!=min(a.CHUNK_BYTES,entry['bytes']-chunk_offset):raise ValueError('Original chunk length differs')
    temp.replace(dest);return dict(path=str(dest),bytes=count,sha256=h.hexdigest())


def disk_usage(state_dir):
    """接口：包含历史中断残留计算实际暂存；新一批预留4块子段及拼接峰值空间。"""
    total=0
    for path in Path(state_dir).rglob('*'):
        try:
            if path.is_file():total+=path.stat().st_size
        except FileNotFoundError:continue
    return total


def download_batch(entries,state_dir):
    """接口：最多四原块、64 HTTP并发；完整子段可恢复，最后仍产出原粒度块。"""
    if not 1<=len(entries)<=ACTIVE_CHUNKS:raise ValueError('Too many active original chunks')
    state_dir=Path(state_dir)
    if disk_usage(state_dir)+2*len(entries)*a.CHUNK_BYTES>TEMPORARY_LIMIT:
        raise ValueError('Temporary storage would exceed 8GiB including legacy residuals')
    jobs=[]
    for entry,offset in entries:
        directory=cache_prefix(state_dir,entry,offset)
        jobs.extend((entry,spec,directory) for spec in segment_specs(entry,offset))
    # 步骤3：失败取消未启动网络任务；在途小段正常落盘，恢复保留成功段。
    with ThreadPoolExecutor(max_workers=HTTP_WORKERS) as pool:
        futures=[pool.submit(fetch_segment,*job) for job in jobs]
        try:
            for future in as_completed(futures):future.result()
        except BaseException:
            for future in futures:future.cancel()
            raise
    return [(entry,offset,assemble_chunk(entry,offset,cache_prefix(state_dir,entry,offset))) for entry,offset in entries]


def cleanup_uploaded(entry,offset,directory):
    """接口：远端确认接收后，只清理本模块有明确定义的本地子段副本。"""
    directory=Path(directory)
    for spec in segment_specs(entry,offset):
        path=directory/(str(spec['offset'])+'.bin')
        path.unlink();path.with_suffix('.json').unlink()
    (directory/'assembled.bin').unlink()


class Remote:
    """接口：只使用已有SSH主连接和固定远端接收器，不触碰模型或实验协调器。"""
    def __init__(self,root,socket,remote,port):
        self.root=str(root);self.base=['ssh','-S',socket,'-o','BatchMode=yes','-o','ConnectTimeout=15','-p',str(port),remote]

    def command(self,stage,entry=None,extra=()):
        """接口：参数经shlex封装，只有固定文件白名单可以进入64MiB接收API。"""
        args=['/App/conda/envs/explore_video_qa_lmms_qwen/bin/python',
            '/home/admin/projects/explore_video_qa/scripts/run_longvideo_set_value_trial.py',
            '--stage',stage,'--asset-root',self.root]
        if entry is not None:a.source_entry(entry['name']);args.extend(['--file',entry['name']])
        args.extend(map(str,extra));return self.base+[shlex.join(args)]

    def query(self,stage,entry=None,extra=()):
        """接口：远端只返回规范JSON；观察失败不猜测传输完成，也不重启协调器。"""
        return json.loads(subprocess.check_output(self.command(stage,entry,extra),text=True,timeout=1800))

    def send(self,entry,offset,block):
        """接口：一次64MiB块上传与原接收器完全兼容；原包最终SHA继续由旧入口负责。"""
        with Path(block['path']).open('rb') as stream:
            subprocess.run(self.command('asset-receive',entry,('--offset',offset,'--length',block['bytes'],
                '--sha256',block['sha256'])),stdin=stream,stdout=subprocess.DEVNULL,check=True,timeout=600)


def run(remote,state_dir):
    """接口：复用全部既有远端块，完成固定八包；只改变网络请求粒度，不改变数据或采样。"""
    state_dir=Path(state_dir);state_dir.mkdir(parents=True,exist_ok=True)
    with a.lock(state_dir/'relay.lock'):
        remote.query('asset-relay-status',extra=('--relay-status','running'))
        entries=a.inventory();states={e['name']:remote.query('asset-state',e) for e in entries}
        if not all(states[e['name']]['complete'] for e in entries[:3]):raise ValueError('Pinned annotation precondition missing')
        entries=entries[3:];completed={e['name']:set(map(int,states[e['name']]['chunks'])) for e in entries}
        final={e['name'] for e in entries if states[e['name']]['complete']};began=time.monotonic();new_bytes=0
        jobs=[]
        for offset in range(0,max(e['bytes'] for e in entries),a.CHUNK_BYTES):
            for entry in entries:
                if entry['name'] not in final and offset<entry['bytes'] and offset not in completed[entry['name']]:
                    jobs.append((entry,offset))
        a.atomic_json(state_dir/'segmented_transport_identity.json',dict(implementation_sha256=a.file_sha(__file__),
            entry_sha256=a.file_sha(Path(__file__).resolve().parents[2]/'scripts/relay_longvideo_trial_assets.py'),
            segment_bytes=SEGMENT_BYTES,original_chunk_bytes=a.CHUNK_BYTES,http_workers=HTTP_WORKERS,
            active_chunks=ACTIVE_CHUNKS,started_unix=time.time(),planned_missing_chunks=len(jobs),
            source_sha256={e['name']:e['sha256'] for e in entries},model_calls=0))
        try:
            for start in range(0,len(jobs),ACTIVE_CHUNKS):
                blocks=download_batch(jobs[start:start+ACTIVE_CHUNKS],state_dir)
                for entry,offset,block in blocks:
                    remote.send(entry,offset,block);completed[entry['name']].add(offset);new_bytes+=block['bytes']
                    cleanup_uploaded(entry,offset,cache_prefix(state_dir,entry,offset))
                    elapsed=time.monotonic()-began
                    a.atomic_json(state_dir/'segmented_progress.json',dict(status='running',uploaded_new_bytes=new_bytes,
                        uploaded_missing_chunks=sum(len(v)-len(states[k]['chunks']) for k,v in completed.items()),
                        seconds=elapsed,bytes_per_second=new_bytes/max(elapsed,1e-6),heartbeat_unix=time.time(),
                        temporary_bytes=disk_usage(state_dir)))
                    if len(completed[entry['name']])==len(range(0,entry['bytes'],a.CHUNK_BYTES)):
                        remote.query('asset-finalize',entry);final.add(entry['name'])
            for entry in entries:
                if entry['name'] not in final:remote.query('asset-finalize',entry)
            remote.query('asset-relay-status',extra=('--relay-status','completed'))
            value=dict(status='completed',uploaded_new_bytes=new_bytes,seconds=time.monotonic()-began,model_calls=0)
            a.atomic_json(state_dir/'segmented_completed.json',value);return value
        except BaseException:
            remote.query('asset-relay-status',extra=('--relay-status','failed'))
            raise
