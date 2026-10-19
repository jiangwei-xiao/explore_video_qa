"""NExT-QA官方资产分段获取：只按身份选择视频，不读取模型对错。"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import csv
import hashlib
import io
import json
import struct
import threading
import time
import zipfile
import zlib

import requests

from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, exclusive, utc

REVISION='2432e9724f88ed9f40010e2989f104570a91de4e'
RAW='https://raw.githubusercontent.com/doc-doc/NExT-QA/'+REVISION+'/dataset/nextqa/'
ARCHIVE='https://drive.usercontent.google.com/download?id=1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH&export=download&confirm=t'
VIDEO_ROOT=ROOT/'data/videos/nextqa'
_LOCAL=threading.local()
_ACCESS_FILE=None


def select_archive_source(root):
    """接口：只允许研究目录内的公开镜像临时下载地址；日志记录规范URL而非签名。"""
    global _ACCESS_FILE
    candidate=Path(root)/'mirror_access.json'
    _ACCESS_FILE=candidate if candidate.exists() else None


def source_descriptor():
    """接口：下载签名可更新，规范源身份固定；公开签名不写入Git或实验展示。"""
    if _ACCESS_FILE is None:return dict(url=ARCHIVE,source_id='official_google')
    return read_json(_ACCESS_FILE)


def stable_key(scope,value):
    """接口：仅种子、作用域和身份决定稳定顺序，答案和模型结果不能作为输入。"""
    return hashlib.sha256(('2027/'+scope+'/'+str(value)).encode()).hexdigest()


def ordered_videos(rows,scope):
    """接口：投影视频ID/问题ID后再排序，每视频至少两题；不按答案或题型筛选。"""
    groups=defaultdict(list)
    for row in rows:groups[row['video']].append((row['type'],row['qid'],row['_row_index']))
    return sorted((v for v,qs in groups.items() if len(set(qs))>=2),key=lambda v:stable_key(scope,v))


def _session():
    """接口：下载线程独享HTTP连接池，最多16个纯网络线程；不记录环境凭据。"""
    if not hasattr(_LOCAL,'session'):_LOCAL.session=requests.Session()
    return _LOCAL.session


def fetch_range(start,length,total=None):
    """接口：严格HTTP206与精确Content-Range，拒绝HTML、整包回退及缓存错段。"""
    end=start+length-1
    for attempt in range(3):
        try:
            # 每段有稳定查询键，避免上游缓存忽略Range头而复用另一个区间。
            source=source_descriptor();url=source['url']
            if _ACCESS_FILE is None:url+='&rg='+str(start)+'_'+str(end)
            response=_session().get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(15,90),stream=True)
            expected=f'bytes {start}-{end}/'
            content_range=response.headers.get('Content-Range','')
            if response.status_code!=206 or not content_range.startswith(expected):
                status=response.status_code;response.close()
                raise ValueError('Exact archive range unavailable: '+str(status)+' '+content_range)
            size=int(content_range.split('/')[-1]);data=response.raw.read(length+1);response.close()
            if len(data)!=length or (total is not None and total!=size):raise ValueError('Archive range length/identity differs')
            return data,size
        except (requests.RequestException,ValueError):
            if attempt==2:raise
            time.sleep(attempt+1)


class RemoteArchive(io.RawIOBase):
    """接口：只读可seek的HTTP文件，标准zipfile负责ZIP64目录解析。"""
    def __init__(self):
        _,self.size=fetch_range(0,1);self.position=0

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position

    def seek(self,offset,whence=0):
        """步骤1按文件端点计算偏移，步骤2拒绝负位置。"""
        p=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if p<0:raise ValueError('Negative archive seek')
        self.position=p;return p

    def read(self,size=-1):
        """接口：仅取请求范围；ZIP目录读取不触发24GB整包下载。"""
        n=self.size-self.position if size<0 else min(size,self.size-self.position)
        if n<=0:return b''
        data,_=fetch_range(self.position,n,self.size);self.position+=n;return data


def official_metadata(root):
    """接口：固定官方revision，仅下载train/val及视频映射，校验完整选择题结构。"""
    root=Path(root);dest=root/'official';dest.mkdir(parents=True,exist_ok=True)
    result={};fingerprints={}
    # 步骤1：固定字节快照；已有文件恢复只能保持一致，不随main更新。
    for name in ('train.csv','val.csv','map_vid_vidorID.json'):
        p=dest/name
        if not p.exists():
            response=_session().get(RAW+name,timeout=(15,90));response.raise_for_status()
            temp=p.with_suffix(p.suffix+'.part');temp.write_bytes(response.content);temp.replace(p)
        fingerprints[name]=sha256(p)
        if name.endswith('.csv'):
            with p.open(newline='') as stream:rows=list(csv.DictReader(stream))
            if len(rows)!={'train.csv':34132,'val.csv':4996}[name]:raise ValueError('Official MCQ count differs')
            for ordinal,row in enumerate(rows):
                # 官方存在同视频同qid但type/问题不同的记录，使用源行作为完整身份，不删改原文。
                row['_row_index']=ordinal
                if not row['question'].strip() or not 0<=int(row['answer'])<5 or any(not row['a'+str(i)].strip() for i in range(5)):
                    raise ValueError('Invalid official MCQ row')
            result[name[:-4]]=rows
    manifest=dict(revision=REVISION,files=fingerprints,official_raw_base=RAW,archive_url=ARCHIVE,
        row_identity_policy='video+type+qid+source_csv_row',utc=utc())
    identity=root/'official_identity.json'
    if identity.exists():
        old=read_json(identity)
        if old['revision']!=REVISION or old['files']!=fingerprints:raise ValueError('Official metadata changed')
    else:durable(identity,manifest)
    return result


def archive_directory(root):
    """接口：一次读取官方ZIP64目录，按文件名ID映射合法MP4成员。"""
    root=Path(root);select_archive_source(root)
    dest=root/('mirror_archive_directory.json' if _ACCESS_FILE else 'archive_directory.json')
    if dest.exists():return read_json(dest)
    remote=RemoteArchive();members={}
    with zipfile.ZipFile(remote) as archive:
        for info in archive.infolist():
            p=Path(info.filename)
            if p.suffix.lower()!='.mp4' or p.parts[0]!='NExTVideo':continue
            if '..' in p.parts or info.flag_bits&1 or info.compress_type not in (0,8):raise ValueError('Unsupported/unsafe video ZIP entry')
            if p.stem in members:raise ValueError('Ambiguous archive video ID')
            members[p.stem]=dict(filename=info.filename,offset=info.header_offset,
                compressed_bytes=info.compress_size,bytes=info.file_size,crc32=info.CRC,compression=info.compress_type)
    result=dict(archive_bytes=remote.size,members=members,utc=utc())
    if _ACCESS_FILE:
        official=read_json(root/'archive_directory.json')['members']
        if set(members)!=set(official) or any(any(members[v][k]!=official[v][k] for k in ('filename','bytes','crc32')) for v in members):
            raise ValueError('Public mirror differs from official raw video member identities')
        result['official_raw_member_identity_matches']=len(members)
        result['source_id']=source_descriptor()['source_id']
    durable(dest,result);return result


def download_member(root,video_id,directory):
    """接口：只获取一个视频的压缩数据，CRC与长度验证后原子发布，可恢复已完成下载。"""
    root=Path(root);entry=directory['members'][video_id];dest=VIDEO_ROOT/entry['filename']
    dest.parent.mkdir(parents=True,exist_ok=True);record=root/'downloads'/(video_id+'.json')
    with exclusive(root/'download_locks'/(video_id+'.lock')):
        if record.exists():
            old=read_json(record)
            if any(old['entry'][k]!=entry[k] for k in ('filename','bytes','crc32')) or not dest.exists() or sha256(dest)!=old['sha256']:
                raise ValueError('Completed video bytes changed')
            return old
        began=time.perf_counter();header,_=fetch_range(entry['offset'],30,directory['archive_bytes'])
        if header[:4]!=b'PK\x03\x04':raise ValueError('Invalid ZIP local header')
        name_size,extra_size=struct.unpack('<HH',header[26:30]);start=entry['offset']+30+name_size+extra_size
        filename,_=fetch_range(entry['offset']+30,name_size,directory['archive_bytes'])
        if filename.decode('utf-8')!=entry['filename']:raise ValueError('ZIP source filename mismatch')
        part=dest.with_suffix('.mp4.part');compressed=root/'download_parts'/source_descriptor()['source_id']/(video_id+'.compressed')
        compressed.parent.mkdir(parents=True,exist_ok=True)
        # 步骤1：压缩数据按4MiB区间取得；既有前缀可从当前位置接续网络下载。
        offset=compressed.stat().st_size if compressed.exists() else 0
        if offset>entry['compressed_bytes']:raise ValueError('Oversized partial video')
        with compressed.open('ab') as out:
            while offset<entry['compressed_bytes']:
                n=min(4*2**20,entry['compressed_bytes']-offset)
                block,_=fetch_range(start+offset,n,directory['archive_bytes']);out.write(block);out.flush();offset+=n
        # 步骤2：CRC验证解压结果；不解包其他成员，不信任ZIP路径创建任意目录。
        crc=0;size=0;decompress=zlib.decompressobj(-15) if entry['compression']==8 else None
        with compressed.open('rb') as source,part.open('wb') as out:
            while True:
                block=source.read(2**20)
                if not block:break
                data=decompress.decompress(block) if decompress else block
                out.write(data);crc=zlib.crc32(data,crc);size+=len(data)
            if decompress:
                data=decompress.flush();out.write(data);crc=zlib.crc32(data,crc);size+=len(data)
        if size!=entry['bytes'] or crc&0xffffffff!=entry['crc32']:raise ValueError('Downloaded video CRC/length differs')
        part.replace(dest)
        result=dict(video_id=video_id,path=str(dest),entry=entry,sha256=sha256(dest),
            downloaded_bytes=entry['compressed_bytes'],archive_source_id=source_descriptor()['source_id'],seconds=time.perf_counter()-began,utc=utc())
        durable(record,result);return result


def acquire(root,train_count=320,val_count=120):
    """接口：按固定身份顺序预取候选，供后续400视频隔离/资格检查；本阶段模型调用0。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'acquisition.lock'):
        metadata=official_metadata(root);directory=archive_directory(root)
        chosen={split:ordered_videos(metadata[split],split)[:n] for split,n in [('train',train_count),('val',val_count)]}
        if any(v not in directory['members'] for ids in chosen.values() for v in ids):raise ValueError('Official video absent from raw archive')
        identities=set(chosen['train'])&set(chosen['val'])
        if identities:raise ValueError('Official train/val video IDs overlap')
        durable(root/'candidate_video_order.json',dict(selected=chosen,all_orders={s:ordered_videos(metadata[s],s) for s in chosen},
            answer_information_used_for_selection=False,seed=2027,utc=utc()))
        ids=chosen['train']+chosen['val'];failures=[];completed=0
        # 步骤1：线程仅做可恢复下载，不能启动BLIP/问答或读取历史对错。
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures={pool.submit(download_member,root,v,directory):v for v in ids}
            for future in as_completed(futures):
                v=futures[future]
                try:future.result();completed+=1
                except BaseException as error:failures.append(dict(video_id=v,error=repr(error)))
                durable(root/'acquisition_status.json',dict(status='downloading',completed=completed,total=len(ids),failures=failures,utc=utc()))
        durable(root/'acquisition_status.json',dict(status='failed' if failures else 'download_completed',completed=completed,total=len(ids),failures=failures,utc=utc()))
        if failures:raise RuntimeError('Some official videos failed to download; inspect acquisition_status')
        return chosen
