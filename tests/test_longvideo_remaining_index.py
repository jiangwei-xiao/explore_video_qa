"""剩余源CPU调度验收：同一总体、反向调度、已完成跳过，无模型调用。"""
import importlib.util
import json
from pathlib import Path

import pytest


def load():
    """测试接口：只加载独立调度器，不运行真实全解码或协调器。"""
    path=Path(__file__).resolve().parents[1]/'scripts/finish_longvideo_source_index.py'
    spec=importlib.util.spec_from_file_location('remaining_index_under_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def setup_sources(tmp_path,monkeypatch):
    """测试接口：构造两个有资格源及一个超时长源，标签字段不在输入中。"""
    m=load();monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'inventory',lambda:[None,None,None,dict(name='unit.tar.gz',sha256='archive')])
    root=tmp_path/'run';members=[]
    for i in range(3):
        x=dict(video_id=str(i),path='/original/'+str(i)+'.mp4',sha256='source'+str(i),bytes=10)
        members.append(x);p=root/'duration_checks'/(str(i)+'.json');p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(dict(x,eligible=i!=2,reason=None if i!=2 else 'duration')))
    p=tmp_path/'data/videos/longvideo_reason/extraction/unit.tar.gz.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(archive_sha256='archive',members=members)))
    return m,root,p


def test_reverse_pending_only(tmp_path,monkeypatch):
    m,root,_=setup_sources(tmp_path,monkeypatch)
    assert [x['video_id'] for x in m.pending_sources(root)[0]]==['1','0']
    (root/'identity_checks').mkdir();(root/'identity_checks/1.json').write_text('{}')
    assert [x['video_id'] for x in m.pending_sources(root)[0]]==['0']


def test_source_identity_mismatch_rejected(tmp_path,monkeypatch):
    m,root,p=setup_sources(tmp_path,monkeypatch);v=json.loads(p.read_text());v['members'][0]['sha256']='changed'
    p.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='identity'):m.pending_sources(root)


def test_cached_source_uses_original_reader_without_redecoding(tmp_path,monkeypatch):
    m=load();root=tmp_path/'run';stage=tmp_path/'isolated';p=root/'identity_checks/one.json'
    p.parent.mkdir(parents=True);p.write_text('{}');seen=[]
    monkeypatch.setattr(m,'index_fingerprint',lambda job:seen.append(job) or dict(video_id='one'))
    result,path=m.check_one((str(root),str(stage),dict(video_id='one')))
    assert path is None and seen[0][0]==str(root) and not stage.exists()


def test_paused_trial_cannot_start_extra_cpu_jobs(tmp_path,monkeypatch):
    m=load();(tmp_path/'PAUSE.json').write_text('{}')
    monkeypatch.setattr(m,'pending_sources',lambda *a:pytest.fail('Must not read/start new work'))
    assert m.run(tmp_path)['status']=='cancelled_before_start'
