"""长视频固定资产：标准库分块中转、SHA验收与安全解包；不导入模型或标签。"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request

ANNOTATION_REVISION = '311eccbd25e9646db8492fb6e728068828ee0684'
VIDEO_REVISION = '3a290aa4592a8a10351f2be86293fdcf735970b5'
CHUNK_BYTES = 64 * 2**20
SHARDS = (
    ('n', 18886638703, '499a2375b749cd29b656e643e618cee3fc826170eb790dcb3a681a61e7404cc1'),
    ('h', 24221458362, '4a00ddc4f3b2e11dcf587751b0c30089ae8ef28a7eba9beb2c6ff032eae5c127'),
    ('Z', 18116088594, '163fb4f34c6741ccc38e751d5c9d05288738c02d0257026aab4e008200376e99'),
    ('d', 34783317328, '1e896bf36e615f739509dfe8fc75f257a94bacf4c370230ee136d2a7afe9b1cf'),
    ('q', 16126169114, '72e31b664b77858bc882b4c278a590c60e7ecc22be2edc322b2c8b56edaedbf2'),
    ('L', 26922780991, '2d244c52fa9f4239175078202f20538d2e07b704c2f3b63ca2023d83fcda1728'),
    ('0', 32295691470, '2f85d2f1cd3cf3fdc6e5887b485dc8b921d8a31108502da52e0cf6b202fa7507'),
    ('X', 18169633173, '2ade6690bfc8c8af0883b10cc7e90adc0f3546a686248d68e5654de511682ca1'))
ANNOTATIONS = (
    ('train', 60365808, '6d18ff2aba368b16814c74eeb1be9005c456ff2b0d394fd5c18771d257e57558'),
    ('validation', 79164, '4de558500ca0e7f5ca7c1c0067d8ec49dc7133e7af0de2b0830b035ba04128a0'),
    ('test', 1098656, 'ca01f20741667ffe69f01860346f4be9a970a02e47399237c5c1022e7a863a55'))


def inventory():
    """接口：只允许已确认的三个Parquet与八分片，大小和LFS内容SHA均为固定值。"""
    result = []
    for split, size, sha in ANNOTATIONS:
        path = f'data/{split}-00000-of-00001.parquet'
        result.append(dict(name=Path(path).name, kind='annotation', bytes=size, sha256=sha,
            url=f'https://huggingface.co/datasets/LongVideo-Reason/longvideo-reason/resolve/{ANNOTATION_REVISION}/{path}'))
    for prefix, size, sha in SHARDS:
        name = f'longvila_videos_{prefix}.tar.gz'
        result.append(dict(name=name, kind='archive', bytes=size, sha256=sha,
            url=f'https://huggingface.co/datasets/Perflow-Shuai/movie_mp4/resolve/{VIDEO_REVISION}/{name}'))
    return result


def file_sha(path):
    """接口：流式完整字节摘要，不把源视频装入内存。"""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''): h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    """接口：中转脚本可在无项目依赖的本机运行；状态替换前同步文件。"""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    with temp.open('w') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False); out.flush(); os.fsync(out.fileno())
    temp.replace(path)


@contextmanager
def lock(path):
    """接口：同文件接收/发布排他，连接断开不导致多进程写同一分块。"""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as out:
        fcntl.flock(out, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def source_entry(name):
    """接口：命令行只能指向固定资产，不接受任意URL、路径或模型文件。"""
    found = [r for r in inventory() if r['name'] == name]
    if len(found) != 1: raise ValueError('Asset is outside the fixed inventory')
    return found[0]


def receive_chunk(root, name, offset, length, expected_sha, stream):
    """接口：远端接收一个至多64MiB分块；先校验再发布，重传只可复用相同字节。"""
    root = Path(root); entry = source_entry(name)
    if offset < 0 or offset % CHUNK_BYTES or length != min(CHUNK_BYTES, entry['bytes'] - offset) or length <= 0:
        raise ValueError('Invalid fixed chunk boundary')
    path = root/'chunks'/name/str(offset); path.parent.mkdir(parents=True, exist_ok=True)
    with lock(path.with_suffix('.lock')):
        # 步骤1：网络输入先到独立暂存；断线留下的暂存不得作为完成分块。
        h = hashlib.sha256(); count = 0; temp = path.with_suffix('.part')
        with temp.open('wb') as out:
            while count < length:
                block = stream.read(min(2**20, length - count))
                if not block: raise ValueError('Incomplete network chunk')
                out.write(block); h.update(block); count += len(block)
            if stream.read(1): raise ValueError('Oversized network chunk')
            out.flush(); os.fsync(out.fileno())
        if h.hexdigest() != expected_sha: raise ValueError('Chunk SHA differs')
        if path.exists() and file_sha(path) != expected_sha: raise ValueError('Existing chunk changed')
        temp.replace(path)
        # 步骤2：完成标记记录逐块摘要；整包仍须再次匹配官方LFS摘要。
        atomic_json(path.with_suffix('.json'), dict(offset=offset, bytes=length, sha256=expected_sha))
    return dict(offset=offset, bytes=length, sha256=expected_sha)


def asset_state(root, name):
    """接口：向本机返回已验收分块和整包状态，不暴露HF签名地址或凭据。"""
    root = Path(root); entry = source_entry(name); complete = root/'verified'/(name+'.json')
    if complete.exists():
        data = json.loads(complete.read_text()); path = Path(data['path'])
        if data['sha256'] != entry['sha256'] or not path.is_file() or path.stat().st_size != entry['bytes']:
            raise ValueError('Verified source identity changed')
        return dict(complete=True, chunks={}, **data)
    chunks = {}
    for cp in (root/'chunks'/name).glob('*.json'):
        item = json.loads(cp.read_text()); path = cp.with_suffix('')
        if not path.is_file() or path.stat().st_size != item['bytes'] or file_sha(path) != item['sha256']:
            raise ValueError('Completed chunk mutated')
        chunks[str(item['offset'])] = item
    return dict(complete=False, chunks=chunks)


def finalize_asset(root, name):
    """接口：按偏移拼接、完整SHA验收后发布；只清理本任务的已核验分块副本。"""
    root = Path(root); entry = source_entry(name)
    with lock(root/'locks'/(name+'.finalize.lock')):
        state = asset_state(root, name)
        if state['complete']:
            if file_sha(state['path']) != entry['sha256']: raise ValueError('Complete source changed')
            return state
        offsets = list(range(0, entry['bytes'], CHUNK_BYTES))
        if set(state['chunks']) != set(map(str, offsets)): raise ValueError('Source chunks incomplete')
        path = root/('official' if entry['kind']=='annotation' else 'archives')/name
        path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(path.suffix+'.part')
        h = hashlib.sha256()
        # 步骤1：仅按冻结偏移拼接，不允许缺段或把另一revision的分块拼入。
        with temp.open('wb') as out:
            for offset in offsets:
                part = root/'chunks'/name/str(offset)
                with part.open('rb') as source:
                    for block in iter(lambda:source.read(8*2**20), b''): h.update(block); out.write(block)
            out.flush(); os.fsync(out.fileno())
        if h.hexdigest()!=entry['sha256'] or temp.stat().st_size!=entry['bytes']:
            raise ValueError('Pinned LFS identity differs')
        temp.replace(path)
        value = dict(path=str(path), **entry, verified_unix=time.time())
        atomic_json(root/'verified'/(name+'.json'), value)
        # 步骤2：完整原包已保留，可从该包恢复；不清理用户数据或其他运行目录。
        for offset in offsets:
            part = root/'chunks'/name/str(offset); part.unlink(); part.with_suffix('.json').unlink()
        return value


def range_to_file(entry, offset, path):
    """接口：本机只取一个精确HTTP206区间，验证长度/总大小，不记录重定向签名。"""
    length = min(CHUNK_BYTES, entry['bytes']-offset); end = offset+length-1
    for attempt in range(4):
        try:
            # 步骤1：区间查询键避免代理缓存把首块返回给后续Range请求。
            url = entry['url'] + f'?download=true&range={offset}-{end}'
            request = urllib.request.Request(url, headers={'Range':f'bytes={offset}-{end}'})
            h = hashlib.sha256(); count=0
            with urllib.request.urlopen(request, timeout=120) as response, Path(path).open('wb') as out:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{entry["bytes"]}':
                    raise ValueError('Server did not honor the exact source range')
                while True:
                    block=response.read(2**20)
                    if not block: break
                    count+=len(block)
                    if count>length: raise ValueError('Oversized HTTP range')
                    out.write(block);h.update(block)
            if count!=length: raise ValueError('Truncated HTTP range')
            return length,h.hexdigest()
        except (OSError, ValueError):
            if attempt==3: raise
            time.sleep(1+attempt)


def relay(root, remote, socket, port, state_dir, workers=32, metadata_only=False):
    """接口：本机至多32路64MiB下载/四路传输，活跃临时文件≤2GiB，含残留总计≤8GiB。"""
    if not 1<=workers<=32: raise ValueError('Relay concurrency exceeds the fixed temporary-space bound')
    state_dir=Path(state_dir);state_dir.mkdir(parents=True,exist_ok=True)
    existing=sum(p.stat().st_size for p in state_dir.glob('longvideo-relay-*/chunk'))
    if existing+workers*CHUNK_BYTES>8*2**30:raise ValueError('Relay temporary disk would exceed 8GiB')
    script='/home/admin/projects/explore_video_qa/scripts/run_longvideo_set_value_trial.py'
    python='/App/conda/envs/explore_video_qa_lmms_qwen/bin/python'
    base=['ssh','-S',socket,'-o','BatchMode=yes','-o','ConnectTimeout=15','-p',str(port),remote]
    def command(stage, entry=None, extra=()):
        """内部接口：远端只接收固定入口参数，公开临时URL不进命令日志或Git。"""
        args=[python,script,'--stage',stage,'--asset-root',str(root)]
        if entry is not None:args.extend(['--file',entry['name']])
        args.extend(map(str,extra))
        return base+[shlex.join(args)]
    def transfer(entry):
        """内部接口：小标注顺序中转，整文件SHA验收后才可在远端解析。"""
        began=time.monotonic(); state=json.loads(subprocess.check_output(command('asset-state',entry),text=True))
        if state['complete']: return dict(name=entry['name'],status='verified',bytes=entry['bytes'],seconds=0)
        with tempfile.TemporaryDirectory(prefix='longvideo-relay-',dir=state_dir) as work:
            temp=Path(work)/'chunk'
            for offset in range(0,entry['bytes'],CHUNK_BYTES):
                if str(offset) in state['chunks']: continue
                length,sha=range_to_file(entry,offset,temp)
                # 步骤2：源段落盘后经独立SSH传输；接收端必须验证本段SHA才能返回成功。
                with temp.open('rb') as source:
                    subprocess.run(command('asset-receive',entry,('--offset',offset,'--length',length,'--sha256',sha)),
                        stdin=source,stdout=subprocess.DEVNULL,check=True,timeout=600)
                temp.unlink()
                elapsed=time.monotonic()-began
                atomic_json(state_dir/(entry['name']+'.progress.json'),dict(name=entry['name'],status='transferring',
                    completed_bytes=offset+length,total_bytes=entry['bytes'],seconds=elapsed,
                    bytes_per_second=(offset+length)/max(elapsed,1e-6),heartbeat_unix=time.time()))
            verified=json.loads(subprocess.check_output(command('asset-finalize',entry),text=True,timeout=1800))
        return dict(name=entry['name'],status='verified',sha256=verified['sha256'],bytes=entry['bytes'],seconds=time.monotonic()-began)
    with lock(state_dir/'relay.lock'):
        subprocess.run(command('asset-relay-status',extra=('--relay-status','running')),check=True,stdout=subprocess.DEVNULL)
        # 步骤3：先传小标注，远端可以先解析资格；视频八包不扩大固定来源范围。
        result=[]
        for entry in inventory()[:3]: result.append(transfer(entry))
        if not metadata_only:
            # 步骤4：下载按分块并行，避免单连接限速；SSH最多四路，不突破服务端会话限制。
            remote_slots=threading.Semaphore(4);progress_lock=threading.Lock();stop=threading.Event();states={};began=time.monotonic()
            entries=inventory()[3:]
            for entry in entries:
                states[entry['name']]=json.loads(subprocess.check_output(command('asset-state',entry),text=True))
            completed={e['name']:(e['bytes'] if states[e['name']]['complete'] else
                sum(c['bytes'] for c in states[e['name']]['chunks'].values())) for e in entries}
            initial=dict(completed)
            def chunk_job(job):
                """内部接口：任何分块失败后停止排队任务，保留已经完整验收的远端字节。"""
                if stop.is_set():return
                entry,offset=job
                with tempfile.TemporaryDirectory(prefix='longvideo-relay-',dir=state_dir) as work:
                    temp=Path(work)/'chunk';length,sha=range_to_file(entry,offset,temp)
                    with remote_slots, temp.open('rb') as source:
                        subprocess.run(command('asset-receive',entry,('--offset',offset,'--length',length,'--sha256',sha)),
                            stdin=source,stdout=subprocess.DEVNULL,check=True,timeout=600)
                with progress_lock:
                    completed[entry['name']]+=length;elapsed=time.monotonic()-began
                    atomic_json(state_dir/(entry['name']+'.progress.json'),dict(name=entry['name'],status='transferring',
                        completed_bytes=completed[entry['name']],total_bytes=entry['bytes'],seconds=elapsed,
                        bytes_per_second=(completed[entry['name']]-initial[entry['name']])/max(elapsed,1e-6),
                        total_new_bytes=sum(completed.values())-sum(initial.values()),workers=workers,heartbeat_unix=time.time()))
            jobs=[]
            for offset in range(0,max(e['bytes'] for e in entries),CHUNK_BYTES):
                for entry in entries:
                    state=states[entry['name']]
                    if offset<entry['bytes'] and not state['complete'] and str(offset) not in state['chunks']:
                        jobs.append((entry,offset))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures=[pool.submit(chunk_job,job) for job in jobs]
                try:
                    for future in as_completed(futures):future.result()
                except BaseException:
                    stop.set()
                    for future in futures:future.cancel()
                    subprocess.run(command('asset-relay-status',extra=('--relay-status','failed')),stdout=subprocess.DEVNULL)
                    raise
            for entry in entries:
                verified=json.loads(subprocess.check_output(command('asset-finalize',entry),text=True,timeout=1800))
                result.append(dict(name=entry['name'],status='verified',sha256=verified['sha256'],bytes=entry['bytes']))
        atomic_json(state_dir/'relay_completed.json',dict(files=result,completed_unix=time.time()))
        subprocess.run(command('asset-relay-status',extra=('--relay-status','completed')),check=True,stdout=subprocess.DEVNULL)
    return result


def extract_archive(root, name, needed_ids):
    """接口：只提取官方train引用的正规MP4；拒绝目录穿越/链接，完整原片SHA旁路记录。"""
    root=Path(root);entry=source_entry(name)
    if entry['kind']!='archive': raise ValueError('Only fixed video archives can be extracted')
    cp=root/'extraction'/(name+'.json')
    if cp.exists():
        value=json.loads(cp.read_text())
        for record in value['members']:
            if file_sha(record['path'])!=record['sha256']: raise ValueError('Extracted original video changed')
        return value
    state=asset_state(root,name)
    if not state['complete'] or file_sha(state['path'])!=entry['sha256']: raise ValueError('Unverified archive')
    records=[];seen=set()
    # 步骤1：流式读取gzip，绝不用extractall或信任归档提供的目标路径。
    with tarfile.open(state['path'],'r|gz') as archive:
        for item in archive:
            path=PurePosixPath(item.name)
            if path.is_absolute() or '..' in path.parts or item.issym() or item.islnk(): raise ValueError('Unsafe archive member')
            if not item.isfile() or path.suffix.lower()!='.mp4' or path.stem not in needed_ids: continue
            vid=path.stem
            if vid in seen: raise ValueError('Repeated source ID within fixed archive')
            seen.add(vid);dest=root/'videos'/(vid+'.mp4');dest.parent.mkdir(parents=True,exist_ok=True)
            member_cp=root/'members'/(vid+'.json')
            if member_cp.exists():
                previous=json.loads(member_cp.read_text())
                if previous['archive']!=name or file_sha(dest)!=previous['sha256']: raise ValueError('Ambiguous/mutated source ID')
                records.append(previous);continue
            source=archive.extractfile(item);temp=dest.with_suffix('.mp4.part');h=hashlib.sha256();count=0
            with source, temp.open('wb') as out:
                for block in iter(lambda:source.read(8*2**20),b''): out.write(block);h.update(block);count+=len(block)
                out.flush();os.fsync(out.fileno())
            if count!=item.size: raise ValueError('Truncated archive video')
            temp.replace(dest)
            record=dict(video_id=vid,path=str(dest),sha256=h.hexdigest(),bytes=count,archive=name,
                archive_sha256=entry['sha256'],member=item.name)
            atomic_json(member_cp,record);records.append(record)
    # 步骤2：已完成成员单独可恢复，整包完成记录只在读到合法EOF后发布。
    value=dict(archive=name,archive_sha256=entry['sha256'],members=records)
    atomic_json(cp,value);return value
