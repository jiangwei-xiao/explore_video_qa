"""传输子段独立验收：不调用模型、不依赖真实网络或已冻结算法。"""
import hashlib
import io
import json
from pathlib import Path

import pytest

from videoqa_set_value import longvideo_transport as t


class Response(io.BytesIO):
    """测试接口：显式模拟精确HTTP206，不允许整包/错段回退。"""
    status=206

    def __init__(self,data,start,end,total):
        super().__init__(data);self.headers={'Content-Range':f'bytes {start}-{end}/{total}'}


@pytest.fixture
def small(monkeypatch):
    data=b'abcdefghijk';entry=dict(name='unit.tar.gz',bytes=len(data),sha256='f'*64,url='https://fixed.invalid/source')
    monkeypatch.setattr(t.a,'CHUNK_BYTES',8);monkeypatch.setattr(t,'SEGMENT_BYTES',3)
    monkeypatch.setattr(t.a,'source_entry',lambda name:entry if name==entry['name'] else (_ for _ in ()).throw(ValueError('whitelist')))
    def open_url(request,timeout):
        assert timeout==120
        start,end=map(int,request.headers['Range'].removeprefix('bytes=').split('-'))
        return Response(data[start:end+1],start,end,len(data))
    monkeypatch.setattr(t.urllib.request,'urlopen',open_url)
    return entry,data


def test_original_chunk_and_last_segment_boundaries(small):
    entry,_=small
    assert t.segment_specs(entry,0)==[dict(offset=0,bytes=3),dict(offset=3,bytes=3),dict(offset=6,bytes=2)]
    assert t.segment_specs(entry,8)==[dict(offset=8,bytes=3)]
    for bad in (-1,1,16):
        with pytest.raises(ValueError):t.segment_specs(entry,bad)


def test_fetch_resume_and_identity(small,tmp_path,monkeypatch):
    entry,data=small;spec=t.segment_specs(entry,0)[0]
    record=t.fetch_segment(entry,spec,tmp_path)
    assert Path(record['path']).read_bytes()==data[:3]
    monkeypatch.setattr(t.urllib.request,'urlopen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('network repeated')))
    assert t.fetch_segment(entry,spec,tmp_path)==record
    Path(record['path']).write_bytes(b'BAD')
    with pytest.raises(ValueError,match='changed'):t.fetch_segment(entry,spec,tmp_path)


def test_batch_equals_original_bytes_and_cleanup(small,tmp_path):
    entry,data=small;blocks=t.download_batch([(entry,0),(entry,8)],tmp_path)
    assert len(blocks)==2
    assert Path(blocks[0][2]['path']).read_bytes()==data[:8]
    assert blocks[0][2]['sha256']==hashlib.sha256(data[:8]).hexdigest()
    assert Path(blocks[1][2]['path']).read_bytes()==data[8:]
    for e,offset,b in blocks:t.cleanup_uploaded(e,offset,Path(b['path']).parent)
    assert not list((tmp_path/'segments').rglob('*.bin'))


def test_partial_failure_preserves_other_complete_segments(small,tmp_path,monkeypatch):
    entry,_=small;directory=t.cache_prefix(tmp_path,entry,0)
    first=t.fetch_segment(entry,t.segment_specs(entry,0)[0],directory)
    saved=Path(first['path']).read_bytes()
    monkeypatch.setattr(t.urllib.request,'urlopen',lambda *a,**k:(_ for _ in ()).throw(OSError('network unavailable')))
    monkeypatch.setattr(t.time,'sleep',lambda _:None)
    with pytest.raises(OSError):t.download_batch([(entry,0)],tmp_path)
    assert Path(first['path']).read_bytes()==saved


def test_wrong_range_no_completion_record(small,tmp_path,monkeypatch):
    entry,_=small
    monkeypatch.setattr(t.urllib.request,'urlopen',lambda *a,**k:Response(b'abc',1,3,11))
    monkeypatch.setattr(t.time,'sleep',lambda _:None)
    with pytest.raises(ValueError,match='range differs'):t.fetch_segment(entry,dict(offset=0,bytes=3),tmp_path)
    assert not list(tmp_path.glob('*.json'))


def test_incomplete_http_retries_only_the_small_segment(small,tmp_path,monkeypatch):
    import http.client
    entry,_=small;original=t.urllib.request.urlopen;calls=[]
    def flaky(*args,**kwargs):
        calls.append(1)
        if len(calls)<3:raise http.client.IncompleteRead(b'a',2)
        return original(*args,**kwargs)
    monkeypatch.setattr(t.urllib.request,'urlopen',flaky);monkeypatch.setattr(t.time,'sleep',lambda _:None)
    record=t.fetch_segment(entry,dict(offset=0,bytes=3),tmp_path)
    assert len(calls)==3 and record['identity']['bytes']==3


def test_assembled_source_identity_cannot_change(small,tmp_path):
    entry,_=small;directory=t.cache_prefix(tmp_path,entry,0)
    for spec in t.segment_specs(entry,0):t.fetch_segment(entry,spec,directory)
    p=directory/'0.json';value=json.loads(p.read_text());value['identity']['source_sha256']='0'*64;p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='unverified'):t.assemble_chunk(entry,0,directory)


def test_temp_limit_includes_legacy_files(small,tmp_path,monkeypatch):
    entry,_=small;(tmp_path/'legacy').write_bytes(b'12345678')
    monkeypatch.setattr(t,'TEMPORARY_LIMIT',20)
    with pytest.raises(ValueError,match='8GiB'):t.download_batch([(entry,0)],tmp_path)


def test_active_original_chunks_are_bounded(small,tmp_path):
    entry,_=small
    with pytest.raises(ValueError,match='active'):t.download_batch([(entry,0)]*5,tmp_path)


def test_temp_reservation_includes_every_assembled_original_chunk(small,tmp_path,monkeypatch):
    entry,_=small
    monkeypatch.setattr(t,'TEMPORARY_LIMIT',50)
    with pytest.raises(ValueError,match='8GiB'):t.download_batch([(entry,0)]*4,tmp_path)


def test_remote_api_is_original_compatible(small):
    entry,_=small;remote=t.Remote('/safe asset','/control.sock','admin@example.invalid',10087)
    command=remote.command('asset-receive',entry,('--offset',0,'--length',8,'--sha256','a'*64))
    assert command[:3]==['ssh','-S','/control.sock']
    assert '--file unit.tar.gz' in command[-1] and "'/safe asset'" in command[-1]
    with pytest.raises(ValueError):remote.command('asset-state',dict(name='outside.tar.gz'))


def test_run_keeps_original_remote_blocks_and_resumes_after_upload_error(small,tmp_path,monkeypatch):
    entry,_=small
    annotations=[dict(name='annotation'+str(i)) for i in range(3)]
    monkeypatch.setattr(t.a,'inventory',lambda:annotations+[entry])
    class Remote:
        """测试接口：模拟原64MiB块已确认，上传失败后仅恢复缺失块，不重复旧块。"""
        def __init__(self):self.existing={0};self.sent=[];self.status=[];self.fail=True

        def query(self,stage,record=None,extra=()):
            if stage=='asset-relay-status':self.status.append(extra[-1]);return dict(status=extra[-1])
            if stage=='asset-state':
                if record['name']!='unit.tar.gz':return dict(complete=True,chunks={})
                return dict(complete=False,chunks={str(i):{} for i in self.existing})
            assert stage=='asset-finalize' and self.existing=={0,8}
            return dict(complete=True,sha256=entry['sha256'])

        def send(self,entry,offset,block):
            assert offset==8 and block['bytes']==3
            if self.fail:raise OSError('upload observation failed')
            self.existing.add(offset);self.sent.append(offset)
    remote=Remote()
    with pytest.raises(OSError):t.run(remote,tmp_path)
    assert remote.status[-1]=='failed' and remote.existing=={0}
    # 子段/拼接字节保留，恢复不再发HTTP读取，也不改原接收API粒度。
    remote.fail=False
    monkeypatch.setattr(t.urllib.request,'urlopen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('HTTP repeated')))
    result=t.run(remote,tmp_path)
    assert result['status']=='completed' and result['uploaded_new_bytes']==3
    assert remote.sent==[8] and remote.status[-1]=='completed'
