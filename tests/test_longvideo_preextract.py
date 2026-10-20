"""预提取验收：只发布完整包、路径隔离、原子竞争及原流程缓存复用。"""
import hashlib
import io
import json
import tarfile

import pytest

from videoqa_set_value import longvideo_assets as assets
from videoqa_set_value import longvideo_preextract as module


@pytest.fixture
def package(tmp_path,monkeypatch):
    """测试接口：模拟一个完整固定包和一个尚未完整的包，不使用真实研究数据。"""
    root=tmp_path/'assets';path=root/'archives/unit.tar.gz';path.parent.mkdir(parents=True)
    with tarfile.open(path,'w:gz') as out:
        for vid,data in [('wanted',b'original-video'),('extra',b'unreferenced')]:
            item=tarfile.TarInfo('longvila_videos/'+vid+'.mp4');item.size=len(data)
            out.addfile(item,io.BytesIO(data))
    entry=dict(name='unit.tar.gz',kind='archive',bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    missing=dict(entry,name='missing.tar.gz')
    monkeypatch.setattr(assets,'inventory',lambda:[dict(kind='annotation'),entry,missing])
    monkeypatch.setattr(assets,'source_entry',lambda name:entry if name==entry['name'] else missing)
    assets.atomic_json(root/'verified'/(entry['name']+'.json'),dict(entry,path=str(path)))
    return root,tmp_path/'stage',entry


def test_verified_only_and_original_reader_reuses(package):
    root,stage,entry=package
    result=module.preextract(root,stage,{'wanted'})
    assert result['model_calls']==0 and len(result['records'])==1
    assert result['records'][0]['videos']==1 and result['records'][0]['published']
    cached=assets.extract_archive(root,entry['name'],{'wanted'})
    assert cached['members'][0]['video_id']=='wanted'
    assert cached['members'][0]['path'].startswith(str(stage))
    assert not (root/'videos').exists()
    assert module.preextract(root,stage,{'wanted'})['records']==[]


def test_mutated_archive_not_published(package):
    root,stage,entry=package
    with (root/'archives'/entry['name']).open('r+b') as out:out.write(b'BAD')
    with pytest.raises(ValueError):module.preextract(root,stage,{'wanted'})
    assert not (root/'extraction'/(entry['name']+'.json')).exists()


def test_coordinator_publishes_first_no_overwrite(package,monkeypatch):
    root,stage,entry=package;original=assets.extract_archive
    def concurrent(staging,name,ids):
        result=original(staging,name,ids)
        original(root,name,ids)
        return result
    monkeypatch.setattr(assets,'extract_archive',concurrent)
    result=module.preextract(root,stage,{'wanted'})
    assert not result['records'][0]['published']
    saved=json.loads((root/'extraction'/(entry['name']+'.json')).read_text())
    assert saved['members'][0]['path'].startswith(str(root/'videos'))


def test_conflicting_receipt_rejected(package,monkeypatch):
    root,stage,entry=package;original=assets.extract_archive
    def conflict(staging,name,ids):
        result=original(staging,name,ids)
        assets.atomic_json(root/'extraction'/(name+'.json'),dict(archive_sha256='bad',members=[]))
        return result
    monkeypatch.setattr(assets,'extract_archive',conflict)
    with pytest.raises(ValueError,match='Concurrent'):module.preextract(root,stage,{'wanted'})
