"""独立数据集合价值学习：无标签泄漏、偏好损失、共享、恢复和大文件区间边界。"""
import io
import inspect
import json
import os
import zipfile
from pathlib import Path
import numpy as np
import pytest
import torch

from videoqa_set_value.learner import SetValueHead,extract_set_features,preference_loss,build_preference_labels,predict_validation
from videoqa_set_value.task import strict_option,context,public_doc
from videoqa_set_value import data,data_acquisition as acquisition,runtime
from videoqa_full.state import durable


@pytest.mark.parametrize('letter',list('ABCDE'))
@pytest.mark.parametrize('form',['{}','{}.','({})'])
def test_five_option_deterministic_parser(letter,form):
    assert strict_option(' \n'+form.format(letter)+' \n')==letter


@pytest.mark.parametrize('raw',['','The answer is A','A B','F','a','(A).','A because it is visible'])
def test_ambiguous_answer_does_not_get_random_label(raw):
    assert strict_option(raw) is None


def test_prompt_excludes_gold_and_keeps_five_original_options():
    row=dict(question='What is visible?',options=['one','two','three','four','five'],answer_index=0)
    first=context(row);row['answer_index']=4
    assert context(row)==first and 'A. one' in first and 'E. five' in first
    assert set(public_doc(row))=={'question','a0','a1','a2','a3','a4'}


def test_feature_api_has_no_labels_options_or_method_source():
    assert list(inspect.signature(extract_set_features).parameters)==['fused','probabilities','times','duration']
    assert list(inspect.signature(runtime.cpu_identity).parameters)==['tools','frames','row']
    x=extract_set_features(np.ones((8,768)),np.linspace(0,1,8),np.arange(8),8.)
    assert x.shape==(8,770) and x.dtype==np.float32
    np.testing.assert_allclose(np.linalg.norm(x[:,:768],axis=1),1,atol=1e-6)
    np.testing.assert_allclose(x[:,769],np.arange(8)/8.)


@pytest.mark.parametrize('value',[float('nan'),-1,2])
def test_bad_itm_feature_rejected(value):
    with pytest.raises(ValueError):extract_set_features(np.ones((1,768)),[value],[0],1.)


def test_set_score_is_order_invariant_and_padding_masked():
    torch.set_num_threads(2);torch.manual_seed(2027);head=SetValueHead().eval();x=torch.randn(2,8,770)
    mask=torch.tensor([[True]*8,[True]*5+[False]*3]);order=torch.tensor([7,1,5,0,4,6,2,3])
    with torch.inference_mode():
        a=head(x,mask);b=head(x[:,order],mask[:,order]);torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        x[1,5:]=999;torch.testing.assert_close(head(x,mask),a)
    assert sum(p.numel() for p in head.parameters())==107073
    with pytest.raises(ValueError):head(x,torch.zeros((2,8),dtype=torch.bool))


def test_preference_loss_cancels_question_bias_and_has_correct_gradient():
    s=torch.tensor([[.2,.3,.1,.0],[.8,.1,.4,.2]],requires_grad=True)
    y=torch.tensor([[1.,1.,0.,0.],[1.,0.,1.,0.]])
    a=preference_loss(s,y,[[0,1,2,3],[0,1,2,3]])
    b=preference_loss(s+torch.tensor([[7.],[-3.]]),y,[[0,1,2,3],[0,1,2,3]])
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6);a.backward()
    assert torch.all(s.grad[y==1]<0) and torch.all(s.grad[y==0]>0)


def test_no_preference_signal_is_not_fabricated():
    with pytest.raises(ValueError,match='No within-question'):
        preference_loss(torch.zeros((1,4)),torch.ones((1,4)),[[0,1,2,3]])


def test_duplicate_inputs_do_not_add_preference_votes():
    s=torch.tensor([[.5,.5,.1,-.2]]);y=torch.tensor([[1.,1.,0.,0.]])
    a=preference_loss(s,y,[[0,2,3]])
    expected=torch.nn.functional.softplus(-torch.tensor([.4,.7])).mean()
    torch.testing.assert_close(a,expected)


def test_training_loader_refuses_validation_before_reading_files(tmp_path):
    with pytest.raises(ValueError,match='Validation labels'):
        build_preference_labels(tmp_path,[dict(split='val')])


def test_video_order_ignores_answer_and_preserves_composite_question_ids():
    rows=[dict(video='v1',qid='2',type='TC',_row_index=0,answer='0'),
          dict(video='v1',qid='2',type='TN',_row_index=1,answer='4'),
          dict(video='v2',qid='0',type='DC',_row_index=2,answer='1')]
    before=acquisition.ordered_videos(rows,'train')
    for row in rows:row['answer']='3'
    assert acquisition.ordered_videos(rows,'train')==before==['v1']


def test_temporal_duplicate_matching_does_not_reuse_one_frame():
    rng=np.random.default_rng(2027);h=[format(int(x),'016x') for x in rng.integers(0,2**63,size=8,dtype=np.uint64)]
    assert data.temporal_matches(h,h)==8
    assert data.temporal_matches(h,list(reversed(h)))<6
    assert data.temporal_matches(h,[h[0]]*8)<6
    bank=[dict(source_id='same',fingerprint=dict(phash=h)),dict(source_id='reverse',fingerprint=dict(phash=list(reversed(h))))]
    assert data.near_duplicates(h,bank)==['same']


def test_http_archive_seek_preserves_offsets_above_32_bits(monkeypatch):
    calls=[]
    def fetch(start,length,total=None):calls.append((start,length));return b'x'*length,2**34
    monkeypatch.setattr(acquisition,'fetch_range',fetch)
    remote=acquisition.RemoteArchive();remote.seek(2**33+17)
    assert remote.read(3)==b'xxx' and calls[-1]==(2**33+17,3)


def test_zip_member_crc_and_resume_do_not_rewrite_completed_file(tmp_path,monkeypatch):
    payload=b'fake-video-payload'*100;buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:archive.writestr('NExTVideo/0001/v1.mp4',payload)
    raw=buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:i=archive.infolist()[0]
    entry=dict(filename=i.filename,offset=i.header_offset,compressed_bytes=i.compress_size,bytes=i.file_size,crc32=i.CRC,compression=i.compress_type)
    monkeypatch.setattr(acquisition,'VIDEO_ROOT',tmp_path/'videos')
    monkeypatch.setattr(acquisition,'_ACCESS_FILE',None)
    monkeypatch.setattr(acquisition,'fetch_range',lambda start,length,total=None:(raw[start:start+length],len(raw)))
    directory=dict(members={'v1':entry},archive_bytes=len(raw))
    first=acquisition.download_member(tmp_path/'run','v1',directory);path=Path(first['path']);stamp=path.stat().st_mtime_ns
    assert path.read_bytes()==payload
    second=acquisition.download_member(tmp_path/'run','v1',directory)
    assert first==second and path.stat().st_mtime_ns==stamp


def test_unknown_teacher_call_does_not_retry(tmp_path):
    durable(tmp_path/'protocol.json',{});fp=runtime.sha256(tmp_path/'protocol.json')
    durable(tmp_path/'calls/BASE-Uniform/q.json',dict(status='started_may_generate',gpu='1',protocol_sha256=fp))
    with pytest.raises(RuntimeError,match='no retry'):runtime.restore_raw(tmp_path,'q','BASE-Uniform',{},'1')


def test_validation_cannot_release_with_changed_head(tmp_path):
    durable(tmp_path/'validation_predictions.json',dict(checkpoint_sha256='not-current'))
    (tmp_path/'head').mkdir();(tmp_path/'head/checkpoint.pt').write_bytes(b'weights')
    with pytest.raises(ValueError,match='head changed'):runtime.phase_protocol(tmp_path,'val')


def test_data_transition_refuses_missing_live_process(tmp_path):
    with pytest.raises(RuntimeError,match='Data process ended'):
        runtime.wait_data_and_run(tmp_path,99999999)


def test_sequential_media_preserves_exact_source_and_never_seeks(monkeypatch):
    import av
    from PIL import Image
    from fractions import Fraction
    from types import SimpleNamespace
    from videoqa_set_value.media import SequentialFrameProvider
    class Container:
        def __init__(self):self.streams=SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1,10),start_time=0,codec_context=SimpleNamespace(thread_count=None))])
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def seek(self,*args,**kwargs):raise AssertionError('Random seek must not be called')
        def decode(self,stream):
            for i in range(5):yield SimpleNamespace(pts=i,to_image=lambda v=i:Image.new('RGB',(4,4),(v,0,0)))
    monkeypatch.setattr(av,'open',lambda path:Container())
    video=dict(path='fake.mp4',time_base='1/10',start_pts=0,duration_fraction='1',duration_seconds=1.)
    frames=[dict(source_pts=i,source_frame_index=i,timestamp_seconds=i/10) for i in (1,4)]
    provider=SequentialFrameProvider(dict(video=video,selected_frames=frames));ims,hashes=provider.decode()
    try:assert [im.getpixel((0,0))[0] for im in ims]==[1,4] and len(hashes)==2
    finally:
        for im in ims:im.close()
    frames[0]['source_frame_index']=0
    with pytest.raises(ValueError,match='source index changed'):
        SequentialFrameProvider(dict(video=video,selected_frames=frames)).decode()


def test_fixed_training_then_validation_prediction_needs_no_validation_answers(tmp_path):
    """只用人工数值fixture验收数据流，不能当作真实学习收益或模型调用。"""
    from videoqa_set_value.learner import fit
    from videoqa_set_value.data import METHODS
    rows=[dict(question_id=f'q{i}',split='train' if i<600 else 'val',video_id=f'v{i//2}') for i in range(800)]
    durable(tmp_path/'manifest.json',dict(rows=rows));durable(tmp_path/'inputs_complete.json',{})
    durable(tmp_path/'qa/train/completion_audit.json',{})
    values={m:np.full((8,770),1. if j in (0,3) else -1.,np.float32) for j,m in enumerate(METHODS)}
    template=tmp_path/'feature_fixture.npz';np.savez(template,**values)
    (tmp_path/'value_features').mkdir();(tmp_path/'inputs').mkdir()
    plan=dict(groups=[dict(representative_method_index=j) for j in range(4)],
        methods={m:dict(source_order=[[j,0]]) for j,m in enumerate(METHODS)})
    for m in METHODS:(tmp_path/'qa/train/results'/m).mkdir(parents=True)
    for row in rows:
        q=row['question_id'];os.link(template,tmp_path/'value_features'/(q+'.npz'))
        (tmp_path/'inputs'/(q+'.json')).write_text(json.dumps(plan))
        if row['split']=='train':
            for j,m in enumerate(METHODS):
                (tmp_path/'qa/train/results'/m/(q+'.json')).write_text(json.dumps(dict(correct=j in (0,3))))
    training=fit(tmp_path)
    assert training['epochs']==20 and training['usable_questions']==600
    assert training['history'][-1]['loss']<training['history'][0]['loss']
    assert training['train_preference_accuracy']==1.
    checkpoint=runtime.sha256(tmp_path/'head/checkpoint.pt')
    predictions=predict_validation(tmp_path)
    assert len(predictions['predictions'])==200 and not list((tmp_path/'qa/val').glob('**/*'))
    assert all(p['chosen_method']==METHODS[0] for p in predictions['predictions'])
    assert fit(tmp_path)==training and runtime.sha256(tmp_path/'head/checkpoint.pt')==checkpoint
    assert predict_validation(tmp_path)==predictions
