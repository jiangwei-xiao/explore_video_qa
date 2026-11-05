"""受控访问的真实媒体、预算与输入隔离验收。"""
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np
import pytest
from PIL import Image
from videoqa_access.transport import (AccessService,AccessClient,CountingReader,decode_exact,
    pack_frames,unpack_frames,preview_bytes,rgb_hash,start_service)
from videoqa_access.trial import coarse_frames,targets_to_frames,subset


@pytest.fixture
def media(tmp_path):
    """接口：生成有真实GOP的短视频，再从解码结果建立完整PTS真值。"""
    import av
    path=tmp_path/'source.mp4'
    with av.open(str(path),'w') as c:
        stream=c.add_stream('mpeg4',rate=4);stream.width=96;stream.height=64;stream.pix_fmt='yuv420p';stream.gop_size=8
        for i in range(24):
            arr=np.full((64,96,3),80,dtype=np.uint8)
            arr[16:32,i:i+16]=[180,60,40]
            for packet in stream.encode(av.VideoFrame.from_ndarray(arr,format='rgb24')):c.mux(packet)
        for packet in stream.encode():c.mux(packet)
    pts=[];keys=[];hashes={}
    with av.open(str(path)) as c:
        s=c.streams.video[0];video=dict(path=str(path),time_base=str(s.time_base),start_pts=s.start_time or 0,
            duration_fraction=str(s.duration*s.time_base),duration_seconds=float(s.duration*s.time_base))
        for frame in c.decode(s):
            pts.append(frame.pts);hashes[frame.pts]=rgb_hash(frame.to_image())
            if frame.key_frame:keys.append(frame.pts)
    pp=tmp_path/'pts.npy';np.save(pp,np.asarray(pts,dtype=np.int64))
    return dict(video=video,pts_path=str(pp),keyframe_pts=keys),pts,hashes


def frame(index,pts):return dict(source_frame_index=index,source_pts=pts)


def test_exact_gop_access_and_payload(media,tmp_path):
    index,pts,hashes=media;wanted=[frame(i,pts[i]) for i in (1,2,3,19)]
    ims,stats=decode_exact(index,wanted)
    try:
        assert [rgb_hash(im) for im in ims]==[hashes[f['source_pts']] for f in wanted]
        assert stats['decoded_frames']>=len(ims)
        assert stats['file_read_bytes']>0 and stats['keyframe_groups']<len(ims)
    finally:
        for im in ims:im.close()
    svc=AccessService(tmp_path/'run',{'v':index});server,thread=start_service(svc)
    url='http://127.0.0.1:'+str(server.server_port);client=AccessClient(url,'v','question/method')
    try:
        ims=client.fetch(wanted)
        for im in ims:im.close()
        assert len(client.requests)==1 and client.requests[0]['payload_bytes']>0
        ims=client.fetch(wanted)
        for im in ims:im.close()
        assert len(client.requests)==1
        second=AccessClient(url,'v','question/other')
        ims=second.fetch(wanted)
        for im in ims:im.close()
        assert len(second.requests)==1;second.close()
    finally:client.close();server.shutdown();server.server_close();thread.join()


def test_service_rejects_identity_and_unknown_source(media,tmp_path):
    index,pts,_=media;svc=AccessService(tmp_path,{'v':index})
    with pytest.raises(ValueError):svc.fetch(dict(video_id='other',frames=[frame(0,pts[0])]))
    with pytest.raises(ValueError):svc.fetch(dict(video_id='v',frames=[frame(1,pts[0])]))
    with pytest.raises(ValueError):svc.fetch(dict(video_id='v',frames=[frame(0,pts[0])]*2))
    with pytest.raises(ValueError):svc.replay('../'*20)


def test_packet_corruption_and_order():
    image=Image.new('RGB',(20,30),'red');data=io.BytesIO();image.save(data,format='PNG')
    value=pack_frames(dict(frames=[dict(rgb_sha256=rgb_hash(image))]),[data.getvalue()])
    meta,ims=unpack_frames(value)
    assert rgb_hash(ims[0])==rgb_hash(image)
    for im in ims:im.close()
    for broken in [value[:7],value[:-1],value+b'x']:
        with pytest.raises((ValueError,json.JSONDecodeError)):unpack_frames(broken)
    wrong=pack_frames(dict(frames=[dict(rgb_sha256='0'*64)]),[data.getvalue()])
    with pytest.raises(ValueError):unpack_frames(wrong)
    image.close()


def test_preview_fixed_geometry_and_stability():
    image=Image.new('RGB',(1920,1080),'blue')
    a=preview_bytes(image);b=preview_bytes(image)
    assert a==b
    with Image.open(io.BytesIO(a)) as im:assert im.size==(320,180)
    small=Image.new('RGB',(80,50),'red')
    with Image.open(io.BytesIO(preview_bytes(small))) as im:assert im.size==(80,50)
    image.close();small.close()


def test_vfr_phase_mapping_and_64bit_pts(tmp_path):
    origin=2**34;pts=np.array([origin+x for x in (0,200,270,750,1000,1280,1750,2000)],dtype=np.int64)
    v=dict(time_base='1/1000',start_pts=origin,duration_fraction='9/4')
    coarse=coarse_frames(pts,v)
    assert [f['source_frame_index'] for f in coarse]==[2,5]
    pp=tmp_path/'pts.npy';np.save(pp,pts);idx=dict(video=v,pts_path=str(pp))
    from fractions import Fraction as F
    fine=targets_to_frames([(F(3,4),F(0),F(3,4),0),(F(7,4),F(0),F(17,10),0)],idx,[])
    assert [f['source_frame_index'] for f in fine]==[3]
    assert targets_to_frames([(F(3,4),F(0),F(1),0)],idx,[int(pts[3])])==[]


def test_subset_ignores_labels_and_preserves_order():
    data=[dict(question_id=str(i),video_id=str(i//2),group='a' if i<4 else 'b',answer_index=i%4) for i in range(8)]
    a=subset(data,'group',('a','b'),2)
    assert [x['question_id'] for x in a]==['0','2','4','6']
    for r in data:r['answer_index']=3-r['answer_index']
    assert [x['question_id'] for x in subset(data,'group',('a','b'),2)]==['0','2','4','6']
    with pytest.raises(ValueError):subset(data,'group',('a','b'),3)


def test_real_http_link_calibration(tmp_path):
    svc=AccessService(tmp_path,{});data=b'x'*(1024*1024);key=hashlib.sha256(data).hexdigest()
    (svc.payload_dir/(key+'.bin')).write_bytes(data);server,thread=start_service(svc)
    try:
        body=json.dumps(dict(payload_sha256=key,link_profile='limited')).encode();begin=time.perf_counter()
        with urlopen(Request('http://127.0.0.1:'+str(server.server_port)+'/replay',body,{'Content-Type':'application/json'})) as response:actual=response.read()
        elapsed=time.perf_counter()-begin;ideal=.05+len(data)*8/20_000_000
        assert actual==data and ideal*.95<=elapsed<=ideal+1.
    finally:server.shutdown();server.server_close();thread.join()


def test_six_methods_real_access_and_safe_preparation_recovery(media,tmp_path,monkeypatch):
    from videoqa_access import trial
    index,pts,_=media;root=tmp_path/'trial';source=Path(index['video']['path']);st=source.stat()
    source_index=dict(index,video_stat=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns),
        video_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),pts_sha256=hashlib.sha256(Path(index['pts_path']).read_bytes()).hexdigest())
    idx=trial.build_preview((str(root),'v',source_index))
    monkeypatch.setattr(trial,'wfs_indices',lambda scores,features:list(range(min(8,len(scores)))))
    class FakeScorer:
        calls=0
        def tokenize(self,q):assert q=='What is visible?';return {}
        def visual_batch(self,images):
            self.calls+=1;f=np.zeros((len(images),768),np.float32);f[:,0]=1
            return f,.001,.001
        def score_batch_features(self,encoded,images):
            f,a,b=self.visual_batch(images);return [.5]*len(images),f,a,b
    scorer=FakeScorer();svc=AccessService(root,{'v':idx});server,thread=start_service(svc)
    row=dict(dataset='videomme',question_id='q',video_id='v',question='What is visible?')
    try:
        result=trial.prepare_question(root,row,scorer,'http://127.0.0.1:'+str(server.server_port))
        assert set(result['items'])==set(trial.METHODS)
        assert result['items']['CTRL-WFS-preview']['requests']==[]
        rd=result['items']['RD-1.2-exp12'];pool=json.loads((root/'pools/videomme/q.json').read_text())
        acquired={p for r in rd['requests'] for p,_ in r['requested_sources']}
        assert all(f['source_pts'] in acquired for f in pool['candidates'][pool['initial_count']:])
        assert rd['extra_visual']['itm_frames']==0
        assert result['items']['CTRL-TopK-local']['extra_visual']['itm_frames']>0
        calls=scorer.calls
        assert trial.prepare_question(root,row,scorer,'unused')['exports']==result['exports']
        assert scorer.calls==calls
        path=root/result['items']['BASE-Uniform']['frames'][0]['image'];path.write_bytes(b'changed')
        with pytest.raises(ValueError):trial.prepare_question(root,row,scorer,'unused')
    finally:server.shutdown();server.server_close();thread.join()


def test_budget_gate_does_not_ignore_preview_baseline():
    from videoqa_access.report import budget_gate
    from videoqa_access.trial import METHODS
    table={m:dict(successful_correct=30,mean_raw_bytes=100.,mean_component_seconds=10.) for m in METHODS}
    table['CTRL-WFS-preview'].update(successful_correct=40,mean_raw_bytes=0.,mean_component_seconds=5.)
    table['RD-1.2-exp12'].update(successful_correct=35,mean_raw_bytes=200.,mean_component_seconds=20.)
    tables={p:{b:table for b in ('8','32')} for p in ('fast','limited')}
    assert not budget_gate(tables)['passed']
    table['RD-1.2-exp12'].update(successful_correct=42,mean_raw_bytes=0.,mean_component_seconds=5.)
    assert budget_gate(tables)['passed']


def test_stratified_pair_bootstrap_reproducible():
    from videoqa_access.report import paired_bootstrap
    a=[1,0,1,1];b=[0,1,0,1];groups=['x','x','y','y']
    assert paired_bootstrap(a,b,groups)==paired_bootstrap(a,b,groups)
    assert paired_bootstrap(a,a,groups)['ci95_pp']==[0.,0.]


def test_qa_completed_recovery_does_not_load_model_or_generate(tmp_path,monkeypatch):
    import types,sys
    from videoqa_access import runtime
    from videoqa_full.state import durable
    from videoqa_runtime.common import sha256
    root=tmp_path/'run';run=root/'qa/videomme';q='q';method='BASE-Uniform'
    durable(root/'protocol.json',dict(test=True));fp=sha256(root/'protocol.json')
    durable(root/'labels.json',{'videomme/q':1});durable(run/'plan.json',dict(calls=[[q,method]]))
    answer=dict(parsed_answer='B',raw_output='B',observed=dict(visual_tokens=1680,max_new_tokens=16,identity=dict(pixels='same')))
    identity=dict(dataset='videomme',question_id=q,method=method,protocol_sha256=fp,gpu='4')
    durable(run/'raw_generations'/method/'q.json',dict(identity=identity))
    durable(run/'calls'/method/'q.json',dict(status='completed',answer=answer,gpu='4'))
    durable(root/'prepared/videomme/q.json',dict(test=True))
    durable(run/'inputs/q.json',dict(prepared_sha256=sha256(root/'prepared/videomme/q.json'),groups=[dict(representative=method,methods=[method])],
        items={method:dict(processed=dict(pixels='same'),frames=[{}]*8,identity=dict(frozen=True))}))
    fake=types.ModuleType('videoqa_audit.frame_group_diagnostic')
    for fn in ['task_backend','observe_answer','synthetic_input']:
        setattr(fake,fn,lambda *a,**k:(_ for _ in ()).throw(AssertionError('model/QA must not run')))
    fake.restore_answer=lambda *a,**k:answer;fake.observed_input=lambda profile,a:a['observed']['identity'];fake.canonical_option=lambda case,a:a['parsed_answer']
    monkeypatch.setitem(sys.modules,'videoqa_audit.frame_group_diagnostic',fake)
    monkeypatch.setattr(runtime,'verify',lambda root:None)
    monkeypatch.setattr(runtime,'data_rows',lambda root,dataset:[dict(question_id=q)])
    args=types.SimpleNamespace(dataset='videomme',slot=0,workers=1,gpu='4')
    runtime.qa_worker(root,args);path=run/'results'/method/'q.json';before=(path.read_bytes(),path.stat().st_mtime_ns)
    runtime.qa_worker(root,args)
    assert (path.read_bytes(),path.stat().st_mtime_ns)==before
    assert json.loads(path.read_text())['correct']
    (run/'raw_generations'/method/'q.json').unlink()
    with pytest.raises(RuntimeError,match='repetition forbidden'):runtime.qa_worker(root,args)
